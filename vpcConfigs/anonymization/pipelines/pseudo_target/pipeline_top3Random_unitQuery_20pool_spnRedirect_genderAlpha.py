"""pipeline_top3Random_unitQuery_20pool_spnRedirect_genderAlpha.py

在 spnRedirect 版本基础上的两处增量改动：

  改动 1 —— gender-aware dynAlpha（性别分治 α）
    主检索 α（分支 B 标准音素）的随机采样区间可按性别分开配置。
    动机：α-cliff 性别不对称——男性声道在 L6 更集中，dynAlpha[0.5,0.6]
    这种偏 L6 的工作点下男性 EER 破防（~37），女性安全（~50）。
    把男性区间压回 [0.3,0.4]（偏 L24）可把男性推回高 EER 平台，
    女性维持原区间不动。
    优先级：use_gender_alpha(固定) > gender-aware dynAlpha > 全局 dynAlpha。

  改动 2 —— enable_spn_redirect（SPN 模块总开关，用于消融）
    一个布尔开关。False 时所有帧（含 phone 0/1）走分支 B 标准检索，
    等价于「关闭 SPN 重定向」的 baseline。
    用途：补齐 v7 缺失的 SPN-off 对照，测量 SPN 模块的真实净效用，
    无需维护第二份 pipeline 文件。

  两处改动均向后兼容：不配置新参数时行为与原版完全一致。
"""

import sys
import io
import json
import random
import subprocess
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, '/root/autodl-tmp/anon_test')

CKPT_DIR  = Path('/root/autodl-tmp/anon_test/checkpoints')
BANKS_DIR = CKPT_DIR / 'banks_v2'
DATA_DIR  = CKPT_DIR / 'trainother500_200spk'
N_POOLS   = 20

# SIL / SPN phone id
SIL_PHONES = {0, 1}

# 音素级自适应Alpha映射
PHONE_ALPHA_MAP = {
    # 静音/噪声: 由 SPN 重定向逻辑接管，此处保留 0.0 作为回退
    0: 0.0, 1: 0.0,
    # 元音: α=0.20
    3: 0.20, 9: 0.20, 10: 0.20, 14: 0.20, 17: 0.20,
    22: 0.20, 25: 0.20, 26: 0.20, 27: 0.20, 31: 0.20,
    33: 0.20, 34: 0.20, 36: 0.20, 39: 0.20, 40: 0.20,
    41: 0.20,
    # 半元音/近音: α=0.18
    7: 0.18, 12: 0.18, 30: 0.18, 37: 0.18,
    # 浊辅音: α=0.12
    8: 0.12, 15: 0.12, 19: 0.12, 21: 0.12,
    24: 0.12, 28: 0.12, 29: 0.12,
    # 清辅音/摩擦: α=0.08
    2: 0.08, 4: 0.08, 5: 0.08, 6: 0.08, 11: 0.08, 13: 0.08,
    16: 0.08, 18: 0.08, 20: 0.08, 23: 0.08, 32: 0.08,
    35: 0.08, 38: 0.08,
}


class PseudoTargetPipelineTop3:
    def __init__(self, config, force_compute=False, devices=None):
        self.config = config
        self.force_compute = force_compute
        self.device = devices[0] if devices else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.anon_suffix = config['modules']['anon_suffix']
        self.mode    = config['modules'].get('mode', 'same')
        self.top_k   = config['modules'].get('top_k', 3)

        # ── alpha 配置 ──
        self.use_gender_alpha = config['modules'].get('use_gender_alpha', False)
        if self.use_gender_alpha:
            self.alpha_m = config['modules'].get('alpha_m', 0.2)
            self.alpha_f = config['modules'].get('alpha_f', 0.3)
            self.alpha = None
        else:
            self.alpha = config['modules'].get('alpha', 0.2)
            self.alpha_m = self.alpha_f = None

        self.use_adaptive_alpha = config['modules'].get('use_adaptive_alpha', False)
        # dynAlpha 模式：每个话语随机采样 alpha ∈ [alpha_low, alpha_high]
        self.alpha_low  = config['modules'].get('alpha_low',  0.5)
        self.alpha_high = config['modules'].get('alpha_high', 1.0)

        # ── 改动 1：gender-aware dynAlpha（性别分治采样区间） ──
        # 未配置时回退到全局 [alpha_low, alpha_high]，行为与原版一致。
        self.alpha_low_m  = config['modules'].get('alpha_low_m',  self.alpha_low)
        self.alpha_high_m = config['modules'].get('alpha_high_m', self.alpha_high)
        self.alpha_low_f  = config['modules'].get('alpha_low_f',  self.alpha_low)
        self.alpha_high_f = config['modules'].get('alpha_high_f', self.alpha_high)

        # ── SPN 重定向参数（方案A） ──
        self.spn_redirect_quantile = config['modules'].get('spn_redirect_quantile', 0.5)
        self.spn_redirect_alpha    = config['modules'].get('spn_redirect_alpha', 0.0)

        # ── 改动 2：SPN 模块总开关（消融用） ──
        # True  = 启用 SPN 重定向（phone 0/1 走分支 A 二次裁决）
        # False = 关闭，phone 0/1 与标准音素一样走分支 B（SPN-off baseline）
        self.enable_spn_redirect = config['modules'].get('enable_spn_redirect', True)

        # ── 其他参数 ──
        self.dur_weight   = config['modules'].get('dur_weight', 0.0)
        self.data_dir     = config['modules'].get('data_dir', str(DATA_DIR))
        self.batch_size   = config['modules'].get('batch_size', 8)

        bank_dir_name  = config['modules'].get('bank_dir', 'banks_v2')
        self.banks_dir = CKPT_DIR / bank_dir_name
        self.n_pools   = config['modules'].get('n_pools', N_POOLS)

        self._load_models()

    # ── Bank / Model loading ────────────────────────────────────────

    def _load_bank(self, path):
        raw = torch.load(str(path), map_location='cpu')
        bank = {}
        for ph, d in raw.items():
            bank[ph] = {
                'l6':  d['l6'].to(self.device),
                'l24': d['l24'].to(self.device),
            }
        fallback = {
            'l6':  torch.cat([v['l6']  for v in bank.values()], 0),
            'l24': torch.cat([v['l24'] for v in bank.values()], 0),
        }
        return bank, fallback

    def _load_models(self):
        from models.ssl.wrappers import WavLMSSLExtractor
        from models.phone_predictor.predictor import PhonePredictor, DurationPredictor
        from models.vocoder.hifigan import HiFiGAN

        self.wavlm = WavLMSSLExtractor(
            ckpt_path=str(CKPT_DIR / 'WavLM-Large.pt'),
            layer=6, device=self.device)
        self.phone_predictor = PhonePredictor.load(
            str(CKPT_DIR / 'phone_decoder.pt'), device=self.device)
        self.duration_predictor = DurationPredictor.load(
            str(CKPT_DIR / 'duration_decoder.pt'), device=self.device)
        self.vocoder = HiFiGAN.load(
            checkpoint_path=str(CKPT_DIR / 'hifigan.pt'), device=self.device)

        self.pool_banks = {}
        for pid in range(self.n_pools):
            self.pool_banks[pid] = {}
            for gender in ['m', 'f']:
                path = self.banks_dir / f'pool_{pid}_gender-{gender}.pt'
                if path.exists():
                    self.pool_banks[pid][gender] = self._load_bank(path)

        # gender-aware dynAlpha 的区间摘要（仅在未用固定 gender_alpha 时打印）
        if self.use_gender_alpha:
            alpha_desc = f"fixed(m={self.alpha_m}, f={self.alpha_f})"
        else:
            alpha_desc = (f"dyn(m=[{self.alpha_low_m},{self.alpha_high_m}], "
                          f"f=[{self.alpha_low_f},{self.alpha_high_f}])")

        print(
            f"[spnRedirect+genderAlpha] 模型加载完成 "
            f"(mode={self.mode}, alpha={alpha_desc}, top_k={self.top_k}, "
            f"pools={self.n_pools}, spn_enabled={self.enable_spn_redirect}, "
            f"spn_quantile={self.spn_redirect_quantile}, "
            f"spn_alpha={self.spn_redirect_alpha})"
        )

    def _select_bank(self, src_gender):
        pool_id = random.randint(0, self.n_pools - 1)
        key = src_gender if self.mode == 'same' else (
            'f' if src_gender == 'm' else 'm')
        return self.pool_banks[pool_id][key]

    def _get_alpha(self, src_gender):
        """主检索 α。
        优先级：固定 gender_alpha > gender-aware dynAlpha > 全局 dynAlpha。
        """
        if self.use_gender_alpha:
            return self.alpha_m if src_gender == 'm' else self.alpha_f
        # gender-aware dynAlpha：男女各自区间随机采样
        if src_gender == 'm':
            return random.uniform(self.alpha_low_m, self.alpha_high_m)
        return random.uniform(self.alpha_low_f, self.alpha_high_f)

    # ── VPC interface ───────────────────────────────────────────────

    def run_anonymization_pipeline(self, datasets):
        for dataset_name, dataset_path in datasets.items():
            self._process_dataset(dataset_name, dataset_path)

    def _load_spk2gender(self, dataset_path):
        spk2gender = {}
        spk2gender_file = dataset_path / 'spk2gender'
        if spk2gender_file.exists():
            for line in spk2gender_file.read_text().strip().split('\n'):
                parts = line.strip().split()
                if len(parts) == 2:
                    spk2gender[parts[0]] = parts[1]
        utt2gender = {}
        utt2spk_file = dataset_path / 'utt2spk'
        if utt2spk_file.exists():
            for line in utt2spk_file.read_text().strip().split('\n'):
                parts = line.strip().split()
                if len(parts) == 2:
                    utt, spk = parts
                    utt2gender[utt] = spk2gender.get(spk, 'm')
        return utt2gender

    def _process_dataset(self, dataset_name, dataset_path):
        output_dir = dataset_path.parent / f"{dataset_name}{self.anon_suffix}"
        wav_dir = output_dir / self.config['results_dir']
        wav_dir.mkdir(parents=True, exist_ok=True)

        wav_scp = dataset_path / 'wav.scp'
        if not wav_scp.exists():
            return

        utt2gender = self._load_spk2gender(dataset_path)
        entries = []
        for line in wav_scp.read_text().strip().split('\n'):
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                utt_id, wav_ref = parts
                out_path = wav_dir / f"{utt_id}.wav"
                if out_path.exists() and not self.force_compute:
                    continue
                if wav_ref.rstrip().endswith('|'):
                    entries.append((utt_id,
                                    ('pipe', wav_ref.rstrip()[:-1].strip()),
                                    out_path))
                else:
                    entries.append((utt_id, ('path', wav_ref), out_path))

        for i in tqdm(range(0, len(entries), self.batch_size),
                      desc=f"spnRedirect {dataset_name}"):
            batch = entries[i:i + self.batch_size]
            self._anonymize_batch(batch, utt2gender)

    def _anonymize_batch(self, batch, utt2gender):
        wavs, lens, metas = [], [], []

        for utt_id, (ref_type, wav_ref), out_path in batch:
            try:
                if ref_type == 'pipe':
                    proc = subprocess.run(
                        wav_ref, shell=True, capture_output=True)
                    if proc.returncode != 0:
                        continue
                    waveform, sr = torchaudio.load(io.BytesIO(proc.stdout))
                else:
                    wav_path = Path(wav_ref)
                    if not wav_path.is_absolute():
                        wav_path = Path(
                            self.config['data_dir']).parent / wav_path
                    if not wav_path.exists():
                        continue
                    waveform, sr = torchaudio.load(wav_path)

                if sr != 16000:
                    waveform = torchaudio.functional.resample(
                        waveform, sr, 16000)
                wav = waveform.mean(0)
                wavs.append(wav)
                lens.append(len(wav))
                metas.append((utt_id, utt2gender.get(utt_id, 'm'), out_path))
            except Exception as e:
                print(f"[Warning] Failed to load {utt_id}: {e}")
                continue

        if not wavs:
            return

        max_len = max(lens)
        wav_batch = torch.zeros(len(wavs), max_len, device=self.device)
        for j, w in enumerate(wavs):
            wav_batch[j, :len(w)] = w

        with torch.no_grad():
            multi = self.wavlm.forward_multi_layer(wav_batch, layers=[6, 24])
            l6_batch  = multi[6]
            l24_batch = multi[24]

            phones_batch = []
            for j, length in enumerate(lens):
                T = length // 320
                phones_batch.append(
                    self.phone_predictor(l24_batch[j, :T]).cpu().numpy())

        h_anon_list = []
        for j, (utt_id, src_gender, out_path) in enumerate(metas):
            T = lens[j] // 320
            h_retrieved = self._retrieve(
                l6_batch[j, :T], l24_batch[j, :T],
                phones_batch[j], src_gender)
            h_anon_list.append(h_retrieved)

        for j, (utt_id, src_gender, out_path) in enumerate(metas):
            h = h_anon_list[j].unsqueeze(0).to(self.device)
            with torch.no_grad():
                wav_out = self.vocoder(h).squeeze(0)

            torchaudio.save(str(out_path), wav_out.unsqueeze(0).cpu(), 16000)

    # ── Core retrieve ───────────────────────────────────────────────

    def _retrieve(self, src_l6, src_l24, phones, src_gender):
        bank, fallback = self._select_bank(src_gender)

        # 当前话语的主检索 alpha（性别分治）
        alpha = self._get_alpha(src_gender)

        # ── 1. RLE：去重 + 时长 ──
        unique_phones, phone_durations = [], []
        cur, cnt = phones[0], 1
        for i in range(1, len(phones)):
            if phones[i] == cur:
                cnt += 1
            else:
                unique_phones.append(cur)
                phone_durations.append(cnt)
                cur, cnt = phones[i], 1
        unique_phones.append(cur)
        phone_durations.append(cnt)

        phones_t = torch.tensor(
            unique_phones, dtype=torch.long, device=self.device)
        dur_true = torch.tensor(
            phone_durations, dtype=torch.float32, device=self.device)

        if self.dur_weight > 0:
            dur_pred = self.duration_predictor(phones_t).squeeze(0)
            dur_anon = (
                self.dur_weight * dur_pred +
                (1 - self.dur_weight) * dur_true
            ).clamp(min=1).round().long()
        else:
            dur_anon = dur_true.long()

        # ── 2. 时长插值展开 ──
        adj_l6, adj_l24, adj_phones = [], [], []
        idx = 0
        for ph, orig, new in zip(unique_phones, phone_durations, dur_anon):
            end = idx + orig
            new_len = new.item()
            if new_len == 0:
                idx = end
                continue
            if orig == 1:
                adj_l6.append(src_l6[idx:idx + 1].expand(new_len, -1))
                adj_l24.append(src_l24[idx:idx + 1].expand(new_len, -1))
            else:
                t = torch.linspace(0, 1, new_len, device=self.device)
                src_idx_f = t * (orig - 1)
                idx_low  = src_idx_f.long().clamp(max=orig - 2)
                idx_high = idx_low + 1
                w = (src_idx_f - idx_low.float()).unsqueeze(-1)
                adj_l6.append(
                    (1 - w) * src_l6[idx + idx_low] +
                    w * src_l6[idx + idx_high])
                adj_l24.append(
                    (1 - w) * src_l24[idx + idx_low] +
                    w * src_l24[idx + idx_high])
            adj_phones.append(
                torch.full((new_len,), ph,
                           dtype=torch.long, device=self.device))
            idx = end

        l6_adj     = torch.cat(adj_l6,     dim=0)
        l24_adj    = torch.cat(adj_l24,    dim=0)
        phones_adj = torch.cat(adj_phones, dim=0)

        T = l6_adj.shape[0]
        h_anon = torch.zeros(T, 1024, device=self.device)

        # 归一化 query
        q_l6_norm  = l6_adj  / (l6_adj.norm( dim=-1, keepdim=True) + 1e-8)
        q_l24_norm = l24_adj / (l24_adj.norm(dim=-1, keepdim=True) + 1e-8)

        # ── 3. 预计算「标准音素质心矩阵」用于 SPN 重定向 ──
        std_phone_ids     = []
        std_centroids_l24 = []
        for _ph, _entry in bank.items():
            if _ph in SIL_PHONES:
                continue
            _c = _entry['l24'].mean(0)
            std_phone_ids.append(_ph)
            std_centroids_l24.append(_c)

        has_std_phones = len(std_centroids_l24) > 0
        if has_std_phones:
            std_centroids_l24_t = torch.stack(std_centroids_l24, dim=0)
            std_centroids_l24_norm = std_centroids_l24_t / (
                std_centroids_l24_t.norm(dim=-1, keepdim=True) + 1e-8)

        # ── 4. 主检索循环 ──
        for phone_id in torch.unique(phones_adj):
            ph   = int(phone_id.item())
            mask = phones_adj == phone_id
            N_q  = mask.sum().item()

            # ============================================================
            # 分支 A：SIL / SPN 帧 → 二次判断 + 分流处理
            #   仅在 enable_spn_redirect=True 时进入；否则 phone 0/1
            #   falls through 到分支 B（SPN-off baseline）。
            # ============================================================
            if self.enable_spn_redirect and ph in SIL_PHONES and has_std_phones:

                q_l24_spn = q_l24_norm[mask]

                dist_to_std = torch.cdist(q_l24_spn, std_centroids_l24_norm)
                min_dist, nearest_local_idx = dist_to_std.min(dim=-1)

                threshold = torch.quantile(
                    min_dist, self.spn_redirect_quantile).item()

                is_pseudo = min_dist < threshold
                is_true   = ~is_pseudo

                # ---- 伪噪声帧：重定向到最近邻标准音素桶 ----
                if is_pseudo.any():
                    pseudo_q_l6_norm  = q_l6_norm[mask][is_pseudo]
                    pseudo_q_l24_norm = q_l24_norm[mask][is_pseudo]
                    nearest_pids_local = nearest_local_idx[is_pseudo]

                    pseudo_out = torch.zeros(
                        is_pseudo.sum(), 1024, device=self.device)

                    for local_idx in torch.unique(nearest_pids_local):
                        target_ph  = std_phone_ids[local_idx.item()]
                        sub_mask   = nearest_pids_local == local_idx
                        N_sub      = sub_mask.sum().item()

                        entry  = bank.get(target_ph, None)
                        tgt_l24 = entry['l24'] if entry else fallback['l24']
                        tgt_l6  = entry['l6']  if entry else fallback['l6']
                        N_t = tgt_l24.shape[0]

                        if N_t == 0:
                            continue
                        if N_t == 1:
                            pseudo_out[sub_mask] = tgt_l6[0].expand(N_sub, -1)
                            continue

                        tgt_l6_norm_sub  = tgt_l6  / (
                            tgt_l6.norm( dim=-1, keepdim=True) + 1e-8)
                        tgt_l24_norm_sub = tgt_l24 / (
                            tgt_l24.norm(dim=-1, keepdim=True) + 1e-8)

                        d_l6  = torch.cdist(
                            pseudo_q_l6_norm[sub_mask],  tgt_l6_norm_sub)
                        d_l24 = torch.cdist(
                            pseudo_q_l24_norm[sub_mask], tgt_l24_norm_sub)

                        d_mix = (self.spn_redirect_alpha * d_l6 +
                                 (1 - self.spn_redirect_alpha) * d_l24)

                        k = min(self.top_k, N_t)
                        topk_idx = d_mix.topk(k, dim=-1, largest=False).indices
                        rand_col = torch.randint(0, k, (N_sub,), device=self.device)
                        sel_idx  = topk_idx[
                            torch.arange(N_sub, device=self.device), rand_col]
                        pseudo_out[sub_mask] = tgt_l6[sel_idx]

                    tmp = h_anon[mask].clone()
                    tmp[is_pseudo] = pseudo_out
                    h_anon[mask]   = tmp

                # ---- 真噪声帧：走 bank 静音桶（privacy-safe，纯 L24） ----
                if is_true.any():
                    entry  = bank.get(ph, None)
                    tgt_l24 = entry['l24'] if entry else fallback['l24']
                    tgt_l6  = entry['l6']  if entry else fallback['l6']
                    N_t     = tgt_l24.shape[0]
                    N_true  = is_true.sum().item()

                    if N_t == 0:
                        pass
                    elif N_t == 1:
                        tmp = h_anon[mask].clone()
                        tmp[is_true] = tgt_l6[0].expand(N_true, -1)
                        h_anon[mask] = tmp
                    else:
                        tgt_l24_norm_sil = tgt_l24 / (
                            tgt_l24.norm(dim=-1, keepdim=True) + 1e-8)
                        tgt_l6_norm_sil  = tgt_l6  / (
                            tgt_l6.norm( dim=-1, keepdim=True) + 1e-8)

                        d_l24_sil = torch.cdist(
                            q_l24_norm[mask][is_true], tgt_l24_norm_sil)
                        d_l6_sil  = torch.cdist(
                            q_l6_norm[mask][is_true],  tgt_l6_norm_sil)
                        d_mix_sil = 0.0 * d_l6_sil + 1.0 * d_l24_sil

                        k = min(self.top_k, N_t)
                        topk_idx = d_mix_sil.topk(k, dim=-1, largest=False).indices
                        rand_col = torch.randint(
                            0, k, (N_true,), device=self.device)
                        sel_idx  = topk_idx[
                            torch.arange(N_true, device=self.device), rand_col]

                        tmp = h_anon[mask].clone()
                        tmp[is_true] = tgt_l6[sel_idx]
                        h_anon[mask] = tmp

                continue

            # ============================================================
            # 分支 B：标准音素 → 原有检索逻辑（不改动）
            #   SPN-off 时 phone 0/1 也走这里。
            # ============================================================
            entry  = bank.get(ph, None)
            tgt_l24 = entry['l24'] if entry else fallback['l24']
            tgt_l6  = entry['l6']  if entry else fallback['l6']

            N_t = tgt_l24.shape[0]
            if N_t == 0:
                continue
            if N_t == 1:
                h_anon[mask] = tgt_l6[0].expand(N_q, -1)
                continue

            tgt_l6_norm  = tgt_l6  / (tgt_l6.norm( dim=-1, keepdim=True) + 1e-8)
            tgt_l24_norm = tgt_l24 / (tgt_l24.norm(dim=-1, keepdim=True) + 1e-8)

            dist_l6  = torch.cdist(q_l6_norm[mask],  tgt_l6_norm)
            dist_l24 = torch.cdist(q_l24_norm[mask], tgt_l24_norm)

            if self.use_adaptive_alpha and N_t >= 5:
                ph_alpha = PHONE_ALPHA_MAP.get(ph, alpha)
            else:
                ph_alpha = alpha
            dist_mix = ph_alpha * dist_l6 + (1 - ph_alpha) * dist_l24

            k = min(self.top_k, N_t)
            topk_idx = dist_mix.topk(k, dim=-1, largest=False).indices

            rand_col     = torch.randint(0, k, (N_q,), device=self.device)
            selected_idx = topk_idx[
                torch.arange(N_q, device=self.device), rand_col]
            h_anon[mask] = tgt_l6[selected_idx]

        return h_anon

    # ── 单文件 / pipe 接口（调试用） ────────────────────────────────

    def _anonymize_from_pipe(self, pipe_cmd, output_path, src_gender='m'):
        proc = subprocess.run(pipe_cmd, shell=True, capture_output=True)
        if proc.returncode != 0:
            return
        waveform, sr = torchaudio.load(io.BytesIO(proc.stdout))
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        self._anonymize_waveform(
            waveform.mean(0, keepdim=True).to(self.device),
            output_path, src_gender)

    def _anonymize_file(self, input_path, output_path, src_gender='m'):
        waveform, sr = torchaudio.load(input_path)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        self._anonymize_waveform(
            waveform.mean(0, keepdim=True).to(self.device),
            output_path, src_gender)

    @torch.no_grad()
    def _anonymize_waveform(self, waveform, output_path, src_gender='m'):
        multi = self.wavlm.forward_multi_layer(waveform, layers=[6, 24])
        l6  = multi[6].squeeze(0)
        l24 = multi[24].squeeze(0)
        phones = self.phone_predictor(l24).cpu().numpy()

        h_anon = self._retrieve(l6, l24, phones, src_gender)
        wav = self.vocoder(h_anon.unsqueeze(0)).squeeze()

        torchaudio.save(str(output_path), wav.unsqueeze(0).cpu(), 16000)