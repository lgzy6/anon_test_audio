"""pipeline_top3Random_unitQuery_20pool_spnRedirect_genderAlpha_sadEnergyGate.py

在 spnRedirect_genderAlpha 版本基础上的第三处增量改动（gate 版）：

  改动 3 —— SAD 能量包络旁路（波形域后补偿，专精 SAD）
    在 HiFi-GAN 之后、波形域，对匿名波形重塑「宏观能量趋势」
    （~1.5Hz 慢变包络 = SAD 的 decrescendo 签名），丢弃 10ms 微动态
    （= 说话人能量习惯 = 身份泄漏来源）。
    与 _retrieve 完全解耦：检索照常产出干净 h_anon，旁路只读
    (源波形, vocoder 输出) 在下游操作，不碰检索一行。

    【gate 版改动】句级门控从「硬阈值（P_sad>gate 全量 / 否则零）」升级为
    「软门控 + anti-ang 守门」：_gate_strength 返回连续强度 g∈[0,1]，
    sad_energy 按 g 线性缩放效果。被误判的 NEU/边缘句只吃到很小比例的处理，
    疑似 ANG 的句子被守门压到 0，从根上挡住「无差别压能量毁掉愤怒句」。
    门控强度计算在本 pipeline（策略层），prosody 模块只当纯信号处理器。

    隐私状态：仍在「注入源信息」红线附近，是探针不是主线。
      - 必须 dur_weight=0（源↔匿名时间轴对齐，否则包络错位）；
      - 每次跑后先看 semi-informed EER，掉破 48 立刻关掉；
      - 起步关门控（全句施加）测 EER；活下来再开 sad 门控调参。

    sad 门控（可选）：仅在句级 P(sad) 超阈值时按强度施加，缩小暴露面 +
    减少对 NEU/ANG/HAP 的污染。情感头必须用 CREMA-D/ESD 训练，
    禁止用 IEMOCAP（= SER 评估集）。

  改动 3 向后兼容：不配置 sad_energy 时行为与原版完全一致。

  ── 前两处改动（沿用） ──
  改动 1 —— gender-aware dynAlpha（性别分治 α）
  改动 2 —— enable_spn_redirect（SPN 模块总开关，消融用）
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

# 情感头标签序（按 docs: SAD/NEU/ANG/HAP）
SAD_IDX = 0
ANG_IDX = 2          # anti-ang 守门用


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
        self.enable_spn_redirect = config['modules'].get('enable_spn_redirect', True)

        # ── 其他参数 ──
        self.dur_weight   = config['modules'].get('dur_weight', 0.0)
        self.data_dir     = config['modules'].get('data_dir', str(DATA_DIR))
        self.batch_size   = config['modules'].get('batch_size', 8)

        bank_dir_name  = config['modules'].get('bank_dir', 'banks_v2')
        self.banks_dir = CKPT_DIR / bank_dir_name
        self.n_pools   = config['modules'].get('n_pools', N_POOLS)

        # ── 改动 3：SAD 能量包络旁路（默认关 → 向后兼容） ──
        sad_cfg = config['modules'].get('sad_energy', None)
        if sad_cfg and sad_cfg.get('enable', False):
            import sys as _sys
            from pathlib import Path as _Path
            _here = str(_Path(__file__).resolve().parent)
            if _here not in _sys.path:
                _sys.path.insert(0, _here)
            from prosody_sad_energy_gate import SadEnergyBypass
            self.sad_energy = SadEnergyBypass(
                slope_strength=sad_cfg.get('slope_strength', 0.5),
                compress=sad_cfg.get('compress', 0.4),
                max_drop_db=sad_cfg.get('max_drop_db', 6.0),
                smooth_ms=sad_cfg.get('smooth_ms', 400),
                sad_gate=sad_cfg.get('sad_gate', 0.5),
                device=self.device,
            )
            # use_gate=False → 全句施加（起步测 EER 用，单变量）
            # use_gate=True  → 句级软门控 + anti-ang 守门（EER 活下来后再开）
            self.emotion_gate = sad_cfg.get('use_gate', False)
            self.emotion_head_ckpt = sad_cfg.get(
                'emotion_head_ckpt', 'emotion_head_cremad.pt')
            # ── gate 版：软门控 + anti-ang 守门参数（在 IEMOCAP-dev 上调，禁止 test 上调） ──
            self.sad_gate_lo  = sad_cfg.get('sad_gate_lo',  sad_cfg.get('sad_gate', 0.4))
            self.sad_gate_hi  = sad_cfg.get('sad_gate_hi',  0.7)
            self.ang_guard_lo = sad_cfg.get('ang_guard_lo', 0.2)
            self.ang_guard_hi = sad_cfg.get('ang_guard_hi', 0.5)
        else:
            self.sad_energy = None
            self.emotion_gate = False

        # 时间轴对齐铁律：能量包络迁移要求源↔匿名逐帧对齐 → dur_weight 必须为 0
        if self.sad_energy is not None and self.dur_weight > 0:
            raise ValueError(
                f"SAD 能量旁路要求 dur_weight=0（源↔匿名时间轴对齐），"
                f"当前 dur_weight={self.dur_weight}")

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

        # ── 改动 3：句级 sad 门控用的情感头（仅门控开启时加载） ──
        # 必须用 CREMA-D/ESD 训练，禁止用 IEMOCAP（= SER 评估集，否则偷看测试集）。
        self.emotion_head = None
        if self.emotion_gate:
            from models.emotion.head import EmotionHead
            self.emotion_head = EmotionHead.load(
                str(CKPT_DIR / self.emotion_head_ckpt), device=self.device)

        self.pool_banks = {}
        for pid in range(self.n_pools):
            self.pool_banks[pid] = {}
            for gender in ['m', 'f']:
                path = self.banks_dir / f'pool_{pid}_gender-{gender}.pt'
                if path.exists():
                    self.pool_banks[pid][gender] = self._load_bank(path)

        # gender-aware dynAlpha 的区间摘要
        if self.use_gender_alpha:
            alpha_desc = f"fixed(m={self.alpha_m}, f={self.alpha_f})"
        else:
            alpha_desc = (f"dyn(m=[{self.alpha_low_m},{self.alpha_high_m}], "
                          f"f=[{self.alpha_low_f},{self.alpha_high_f}])")

        sad_desc = "off"
        if self.sad_energy is not None:
            sad_desc = (f"on(beta={self.sad_energy.beta}, "
                        f"smooth={self.sad_energy.smooth_ms}ms, "
                        f"gate={'soft-sad' if self.emotion_gate else 'all-utt'})")

        print(
            f"[spnRedirect+genderAlpha+sadEnergyGate] 模型加载完成 "
            f"(mode={self.mode}, alpha={alpha_desc}, top_k={self.top_k}, "
            f"pools={self.n_pools}, spn_enabled={self.enable_spn_redirect}, "
            f"spn_quantile={self.spn_redirect_quantile}, "
            f"spn_alpha={self.spn_redirect_alpha}, "
            f"dur_weight={self.dur_weight}, sad_energy={sad_desc})"
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
        if src_gender == 'm':
            return random.uniform(self.alpha_low_m, self.alpha_high_m)
        return random.uniform(self.alpha_low_f, self.alpha_high_f)

    # ── 改动 3（gate 版）：句级门控强度（复用已算的源特征，不再跑一遍模型） ──
    @staticmethod
    def _smoothstep(x, lo, hi):
        """平滑阶跃：x<=lo→0, x>=hi→1, 中间三次平滑过渡。"""
        if hi <= lo:
            return 1.0 if x >= hi else 0.0
        t = min(1.0, max(0.0, (x - lo) / (hi - lo)))
        return t * t * (3.0 - 2.0 * t)

    @torch.no_grad()
    def _gate_strength(self, src_l24_TxD):
        """src_l24_TxD: 该句源 L24 特征 [T,1024]。
        返回门控强度 g∈[0,1]：
          软门控（随 P_sad 平滑上升） × anti-ang 守门（P_ang 高则压向 0）。
          g=0 → sad_energy 完全不动该句；g=1 → 全量施加。
        """
        if self.emotion_head is None:
            return 1.0
        feat = src_l24_TxD.mean(0, keepdim=True)                 # [1,1024]
        probs = torch.softmax(self.emotion_head(feat), dim=-1).squeeze(0)
        p_sad = probs[SAD_IDX].item()
        p_ang = probs[ANG_IDX].item()
        g = self._smoothstep(p_sad, self.sad_gate_lo, self.sad_gate_hi)
        g *= (1.0 - self._smoothstep(p_ang, self.ang_guard_lo, self.ang_guard_hi))
        return g

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

        # 第一轮：检索 h_anon + （门控开启时）顺手算句级门控强度 g
        h_anon_list = []
        gate_list = []
        for j, (utt_id, src_gender, out_path) in enumerate(metas):
            T = lens[j] // 320
            h_retrieved = self._retrieve(
                l6_batch[j, :T], l24_batch[j, :T],
                phones_batch[j], src_gender)
            h_anon_list.append(h_retrieved)

            # 复用已算的源 L24，不再跑模型
            if self.sad_energy is not None and self.emotion_gate:
                gate_list.append(self._gate_strength(l24_batch[j, :T]))
            else:
                gate_list.append(1.0)       # 不门控 = 全句全量施加

        # 第二轮：vocoder → SAD 能量旁路（解耦、后处理） → 保存
        for j, (utt_id, src_gender, out_path) in enumerate(metas):
            h = h_anon_list[j].unsqueeze(0).to(self.device)
            with torch.no_grad():
                wav_out = self.vocoder(h).squeeze(0)

            # ── 改动 3：唯一插入点。检索/vocoder 之外，纯下游 ──
            if self.sad_energy is not None:
                src_wav = wavs[j].to(self.device)   # 第 j 条源波形（与 metas 对齐）
                wav_out = self.sad_energy.process(
                    wav_out, src_wav, sr=16000, gate=gate_list[j])

            torchaudio.save(str(out_path), wav_out.unsqueeze(0).cpu(), 16000)

    # ── Core retrieve （完全不改动，检索与旁路解耦） ─────────────────

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

        # ── 改动 3：单文件路径同样的下游旁路 ──
        if self.sad_energy is not None:
            src_wav = waveform.squeeze(0)
            gate = (self._gate_strength(l24)
                    if self.emotion_gate else 1.0)
            wav = self.sad_energy.process(
                wav, src_wav, sr=16000, gate=gate)

        torchaudio.save(str(output_path), wav.unsqueeze(0).cpu(), 16000)