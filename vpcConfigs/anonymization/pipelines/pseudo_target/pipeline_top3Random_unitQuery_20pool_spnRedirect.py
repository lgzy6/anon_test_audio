"""pipeline_top3Random_unitQuery_20pool_spnRedirect.py

改动说明（相对于 dynAlpha 版本）：
    针对 SIL/SPN（phone id 0/1）实现「方案A：SPN 桶分裂重定向」。

    原始问题：全局 bank 下 SPN 桶积累了大量被 phone predictor 错分的真实音素帧
    （尤其是方言/说话人特有发音帧），导致 SPN 桶的 L24 分布多峰、高方差，
    一次 KMeans 聚类无语义意义，检索输出随机且混入说话人声道信息。

    解决方案（方案A）：在推理时对每个 SPN/SIL 帧做二次判断：
        · 计算该帧 L24 到所有标准音素桶质心的最小余弦距离
        · 距离 < threshold → "伪噪声帧"：重定向到最近邻标准音素桶做 kNN 检索
        · 距离 ≥ threshold → "真噪声帧"：走 bank 静音桶检索（privacy-safe）

    关键参数（yaml 新增）：
        spn_redirect_quantile: 0.5   # 0.25=激进 / 0.5=均衡(默认) / 0.75=保守
        spn_redirect_alpha:    0.0   # 重定向帧的检索 alpha，0.0=纯 L24 最安全

    ⚠ 安全性：真噪声帧走 bank 静音桶 L24 检索，输出是 bank 质心，
      不透传任何源 L6，与已证伪的 silenceNoise（直接用源 L6 加噪）根本不同。
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

        # ── SPN 重定向参数（方案A） ──
        # spn_redirect_quantile: 分位数阈值
        #   0.25 = 激进（大多数 SPN 帧重定向到标准音素桶）
        #   0.50 = 均衡（默认，中位数分割）
        #   0.75 = 保守（只有离标准音素最近的帧才重定向）
        self.spn_redirect_quantile = config['modules'].get('spn_redirect_quantile', 0.5)
        # spn_redirect_alpha: 重定向帧的检索 alpha
        #   0.0 = 纯 L24 检索（最安全，不泄漏 L6 身份信息）
        self.spn_redirect_alpha = config['modules'].get('spn_redirect_alpha', 0.0)

        # ── 其他参数 ──
        self.use_prosody  = config['modules'].get('use_prosody_injection', False)
        self.prosody_beta = config['modules'].get('prosody_beta', 0.3)
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

        if self.use_prosody:
            from .prosody_module import ProsodyInjector
            self.prosody_injector = ProsodyInjector(
                device=self.device, beta=self.prosody_beta)

        print(
            f"[spnRedirect] 模型加载完成 "
            f"(mode={self.mode}, alpha={self.alpha}, top_k={self.top_k}, "
            f"pools={self.n_pools}, "
            f"spn_quantile={self.spn_redirect_quantile}, "
            f"spn_alpha={self.spn_redirect_alpha})"
        )

    def _select_bank(self, src_gender):
        pool_id = random.randint(0, self.n_pools - 1)
        key = src_gender if self.mode == 'same' else (
            'f' if src_gender == 'm' else 'm')
        return self.pool_banks[pool_id][key]

    def _get_alpha(self, src_gender):
        """根据配置返回当前话语使用的 alpha（dynAlpha 或 gender-alpha 或固定）。"""
        if self.use_gender_alpha:
            return self.alpha_m if src_gender == 'm' else self.alpha_f
        # dynAlpha 模式：每话语随机采样
        return random.uniform(self.alpha_low, self.alpha_high)

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

            if self.use_prosody:
                wav_out = self.prosody_injector.process_wav(wav_out, wavs[j])

            torchaudio.save(str(out_path), wav_out.unsqueeze(0).cpu(), 16000)

    # ── Core retrieve ───────────────────────────────────────────────

    def _retrieve(self, src_l6, src_l24, phones, src_gender):
        bank, fallback = self._select_bank(src_gender)

        # 当前话语的全局 alpha（dynAlpha / gender-alpha / fixed）
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
        #    只做一次，在循环外，避免重复计算
        std_phone_ids     = []   # 标准音素 id 列表（排除 SIL_PHONES）
        std_centroids_l24 = []   # 对应桶的 L24 均值质心 [P, 1024]

        for _ph, _entry in bank.items():
            if _ph in SIL_PHONES:
                continue
            # 用桶内所有 L24 质心的均值作为该桶的代表向量
            _c = _entry['l24'].mean(0)   # [1024]
            std_phone_ids.append(_ph)
            std_centroids_l24.append(_c)

        has_std_phones = len(std_centroids_l24) > 0
        if has_std_phones:
            std_centroids_l24_t = torch.stack(
                std_centroids_l24, dim=0)                         # [P, 1024]
            std_centroids_l24_norm = std_centroids_l24_t / (
                std_centroids_l24_t.norm(dim=-1, keepdim=True) + 1e-8)  # [P, 1024]

        # ── 4. 主检索循环 ──
        for phone_id in torch.unique(phones_adj):
            ph   = int(phone_id.item())
            mask = phones_adj == phone_id
            N_q  = mask.sum().item()

            # ============================================================
            # 分支 A：SIL / SPN 帧 → 二次判断 + 分流处理
            # ============================================================
            if ph in SIL_PHONES and has_std_phones:

                q_l24_spn = q_l24_norm[mask]   # [N_q, 1024]，已归一化

                # 每帧到所有标准音素质心的余弦距离，取最小值
                dist_to_std = torch.cdist(
                    q_l24_spn, std_centroids_l24_norm)        # [N_q, P]
                min_dist, nearest_local_idx = dist_to_std.min(dim=-1)
                # min_dist:          [N_q]  每帧到最近标准音素的距离
                # nearest_local_idx: [N_q]  在 std_phone_ids 中的下标

                # threshold：用分位数自适应划分（batch 内）
                threshold = torch.quantile(
                    min_dist, self.spn_redirect_quantile).item()

                is_pseudo = min_dist < threshold   # 伪噪声 → 重定向到标准音素桶
                is_true   = ~is_pseudo             # 真噪声 → 走静音桶

                # ---- 伪噪声帧：重定向到最近邻标准音素桶 ----
                if is_pseudo.any():
                    pseudo_indices = is_pseudo.nonzero(as_tuple=True)[0]
                    pseudo_q_l6_norm  = q_l6_norm[mask][is_pseudo]   # [N_p, 1024]
                    pseudo_q_l24_norm = q_l24_norm[mask][is_pseudo]  # [N_p, 1024]
                    nearest_pids_local = nearest_local_idx[is_pseudo] # [N_p]

                    pseudo_out = torch.zeros(
                        is_pseudo.sum(), 1024, device=self.device)

                    # 按目标音素桶分组，避免对每帧单独检索
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

                        # 重定向帧：使用 spn_redirect_alpha（默认 0.0=纯 L24）
                        d_mix = (self.spn_redirect_alpha * d_l6 +
                                 (1 - self.spn_redirect_alpha) * d_l24)

                        k = min(self.top_k, N_t)
                        topk_idx = d_mix.topk(k, dim=-1, largest=False).indices
                        rand_col = torch.randint(0, k, (N_sub,), device=self.device)
                        sel_idx  = topk_idx[
                            torch.arange(N_sub, device=self.device), rand_col]
                        pseudo_out[sub_mask] = tgt_l6[sel_idx]

                    # 写回伪噪声帧
                    tmp = h_anon[mask].clone()
                    tmp[is_pseudo] = pseudo_out
                    h_anon[mask]   = tmp

                # ---- 真噪声帧：走 bank 静音桶（privacy-safe kNN，不透传源 L6） ----
                if is_true.any():
                    entry  = bank.get(ph, None)
                    tgt_l24 = entry['l24'] if entry else fallback['l24']
                    tgt_l6  = entry['l6']  if entry else fallback['l6']
                    N_t     = tgt_l24.shape[0]
                    N_true  = is_true.sum().item()

                    if N_t == 0:
                        pass   # bank 无静音桶，输出零向量（已初始化为 0）
                    elif N_t == 1:
                        tmp = h_anon[mask].clone()
                        tmp[is_true] = tgt_l6[0].expand(N_true, -1)
                        h_anon[mask] = tmp
                    else:
                        tgt_l24_norm_sil = tgt_l24 / (
                            tgt_l24.norm(dim=-1, keepdim=True) + 1e-8)
                        tgt_l6_norm_sil  = tgt_l6  / (
                            tgt_l6.norm( dim=-1, keepdim=True) + 1e-8)

                        # 真噪声帧：纯 L24 检索，完全不含源 L6 信息
                        d_l24_sil = torch.cdist(
                            q_l24_norm[mask][is_true], tgt_l24_norm_sil)
                        d_l6_sil  = torch.cdist(
                            q_l6_norm[mask][is_true],  tgt_l6_norm_sil)
                        # alpha=0.0 → 纯 L24
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

                # SPN 分支处理完毕，进入下一个 phone
                continue

            # ============================================================
            # 分支 B：标准音素 → 原有检索逻辑（不改动）
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

            # 音素级 adaptive alpha，回退到话语级 alpha
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

        if self.use_prosody:
            wav = self.prosody_injector.process_wav(wav, waveform.squeeze(0))

        torchaudio.save(str(output_path), wav.unsqueeze(0).cpu(), 16000)