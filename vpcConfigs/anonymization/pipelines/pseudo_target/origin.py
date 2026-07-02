"""VPC 2024 Pipeline — kNN Top-3 随机采样匿名化"""
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
N_POOLS   = 4

# 音素级自适应Alpha映射 (基于phone_lexicon.txt的42个音素)
# 语音学原理: 元音共振峰携带强说话人身份→低alpha(偏L24检索)
#            辅音摩擦噪声说话人信息少→高alpha(偏L6保质量)
PHONE_ALPHA_MAP = {
    # 静音/噪声: alpha=0.5 (几乎不影响识别)
    0: 0.5, 1: 0.5,
    # 清辅音/摩擦: alpha=0.4 (保内容清晰)
    2: 0.4, 4: 0.4, 5: 0.35, 6: 0.4, 11: 0.4, 13: 0.35,
    16: 0.4, 18: 0.4, 20: 0.4, 23: 0.4, 32: 0.4, 35: 0.4, 38: 0.4,
    # 浊辅音: alpha=0.25
    8: 0.25, 15: 0.25, 19: 0.25, 21: 0.25, 24: 0.25, 28: 0.25, 29: 0.25,
    # 半元音/近音: alpha=0.15
    7: 0.15, 12: 0.15, 30: 0.15, 37: 0.15,
    # 元音: alpha=0.05 (最强隐私保护)
    3: 0.05, 9: 0.05, 10: 0.05, 14: 0.05, 17: 0.05,
    22: 0.05, 25: 0.05, 26: 0.05, 27: 0.05, 31: 0.05,
    33: 0.05, 34: 0.05, 36: 0.05, 39: 0.05, 40: 0.05,
}


class PseudoTargetPipelineTop3:
    def __init__(self, config, force_compute=False, devices=None):
        self.config = config
        self.force_compute = force_compute
        self.device = devices[0] if devices else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.anon_suffix = config['modules']['anon_suffix']
        self.mode    = config['modules'].get('mode', 'same')
        self.top_k   = config['modules'].get('top_k', 3)
        self.alpha   = config['modules'].get('alpha', 0.2)
        self.use_adaptive_alpha = config['modules'].get('use_adaptive_alpha', False)

        self.dur_weight = config['modules'].get('dur_weight', 0.0)
        self.data_dir = config['modules'].get('data_dir', str(DATA_DIR))
        self.batch_size = config['modules'].get('batch_size', 8)
        self._load_models()

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

        self.wavlm = WavLMSSLExtractor(ckpt_path=str(CKPT_DIR / 'WavLM-Large.pt'), layer=6, device=self.device)
        self.phone_predictor = PhonePredictor.load(str(CKPT_DIR / 'phone_decoder.pt'), device=self.device)
        self.duration_predictor = DurationPredictor.load(str(CKPT_DIR / 'duration_decoder.pt'), device=self.device)
        self.vocoder = HiFiGAN.load(checkpoint_path=str(CKPT_DIR / 'hifigan.pt'), device=self.device)

        self.pool_banks = {}
        for pid in range(N_POOLS):
            self.pool_banks[pid] = {}
            for gender in ['m', 'f']:
                path = BANKS_DIR / f'pool_{pid}_gender-{gender}.pt'
                if path.exists():
                    self.pool_banks[pid][gender] = self._load_bank(path)

        print(f"[Top3] 模型加载完成 (mode={self.mode}, alpha={self.alpha}, top_k={self.top_k}, pools={N_POOLS}, adaptive_alpha={self.use_adaptive_alpha})")

    def _select_bank(self, src_gender):
        pool_id = random.randint(0, N_POOLS - 1)
        key = src_gender if self.mode == 'same' else ('f' if src_gender == 'm' else 'm')
        return self.pool_banks[pool_id][key]

    # ── VPC interface ──────────────────────────────────────────────

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
                    entries.append((utt_id, ('pipe', wav_ref.rstrip()[:-1].strip()), out_path))
                else:
                    entries.append((utt_id, ('path', wav_ref), out_path))

        for i in tqdm(range(0, len(entries), self.batch_size), desc=f"Top3 {dataset_name}"):
            batch = entries[i:i+self.batch_size]
            self._anonymize_batch(batch, utt2gender)

    def _anonymize_batch(self, batch, utt2gender):
        wavs, lens, metas = [], [], []

        for utt_id, (ref_type, wav_ref), out_path in batch:
            try:
                if ref_type == 'pipe':
                    proc = subprocess.run(wav_ref, shell=True, capture_output=True)
                    if proc.returncode != 0:
                        continue
                    waveform, sr = torchaudio.load(io.BytesIO(proc.stdout))
                else:
                    wav_path = Path(wav_ref)
                    if not wav_path.is_absolute():
                        wav_path = Path(self.config['data_dir']).parent / wav_path
                    if not wav_path.exists():
                        continue
                    waveform, sr = torchaudio.load(wav_path)

                if sr != 16000:
                    waveform = torchaudio.functional.resample(waveform, sr, 16000)
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
            l6_batch = multi[6]
            l24_batch = multi[24]

            phones_batch = []
            for j, length in enumerate(lens):
                T = length // 320
                phones_batch.append(self.phone_predictor(l24_batch[j, :T]).cpu().numpy())

        h_anon_list = []
        for j, (utt_id, src_gender, out_path) in enumerate(metas):
            T = lens[j] // 320
            h_anon = self._retrieve(l6_batch[j, :T], l24_batch[j, :T], phones_batch[j], src_gender)
            h_anon_list.append(h_anon)

        for j, (utt_id, src_gender, out_path) in enumerate(metas):
            h = h_anon_list[j].unsqueeze(0).to(self.device)
            with torch.no_grad():
                wav_out = self.vocoder(h).squeeze(0)
            torchaudio.save(str(out_path), wav_out.unsqueeze(0).cpu(), 16000)

    def _anonymize_from_pipe(self, pipe_cmd, output_path, src_gender='m'):
        proc = subprocess.run(pipe_cmd, shell=True, capture_output=True)
        if proc.returncode != 0:
            return
        waveform, sr = torchaudio.load(io.BytesIO(proc.stdout))
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        self._anonymize_waveform(waveform.mean(0, keepdim=True).to(self.device), output_path, src_gender)

    def _anonymize_file(self, input_path, output_path, src_gender='m'):
        waveform, sr = torchaudio.load(input_path)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        self._anonymize_waveform(waveform.mean(0, keepdim=True).to(self.device), output_path, src_gender)

    # ── Core ───────────────────────────────────────────────────────

    @torch.no_grad()
    def _anonymize_waveform(self, waveform, output_path, src_gender='m'):
        multi = self.wavlm.forward_multi_layer(waveform, layers=[6, 24])
        l6 = multi[6].squeeze(0)
        l24 = multi[24].squeeze(0)
        phones = self.phone_predictor(l24).cpu().numpy()

        h_anon = self._retrieve(l6, l24, phones, src_gender)

        wav = self.vocoder(h_anon.unsqueeze(0)).squeeze()
        torchaudio.save(str(output_path), wav.unsqueeze(0).cpu(), 16000)

    def _retrieve(self, src_l6, src_l24, phones, src_gender):
        bank, fallback = self._select_bank(src_gender)

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

        phones_t = torch.tensor(unique_phones, dtype=torch.long, device=self.device)
        dur_true = torch.tensor(phone_durations, dtype=torch.float32, device=self.device)

        if self.dur_weight > 0:
            dur_pred = self.duration_predictor(phones_t).squeeze(0)
            dur_anon = (self.dur_weight * dur_pred + (1 - self.dur_weight) * dur_true).clamp(min=1).round().long()
        else:
            dur_anon = dur_true.long()

        adj_l6, adj_l24, adj_phones = [], [], []
        idx = 0
        for ph, orig, new in zip(unique_phones, phone_durations, dur_anon):
            end = idx + orig
            new_len = new.item()
            if new_len == 0:
                idx = end
                continue
            if orig == 1:
                adj_l6.append(src_l6[idx:idx+1].expand(new_len, -1))
                adj_l24.append(src_l24[idx:idx+1].expand(new_len, -1))
            else:
                t = torch.linspace(0, 1, new_len, device=self.device)
                src_idx_f = t * (orig - 1)
                idx_low = src_idx_f.long().clamp(max=orig - 2)
                idx_high = idx_low + 1
                w = (src_idx_f - idx_low.float()).unsqueeze(-1)
                adj_l6.append((1 - w) * src_l6[idx + idx_low] + w * src_l6[idx + idx_high])
                adj_l24.append((1 - w) * src_l24[idx + idx_low] + w * src_l24[idx + idx_high])
            adj_phones.append(torch.full((new_len,), ph, dtype=torch.long, device=self.device))
            idx = end

        l6_adj = torch.cat(adj_l6, dim=0)
        l24_adj = torch.cat(adj_l24, dim=0)
        phones_adj = torch.cat(adj_phones, dim=0)

        T = l6_adj.shape[0]
        h_anon = torch.zeros(T, 1024, device=self.device)

        q_l6_norm = l6_adj / (l6_adj.norm(dim=-1, keepdim=True) + 1e-8)
        q_l24_norm = l24_adj / (l24_adj.norm(dim=-1, keepdim=True) + 1e-8)

        for phone_id in torch.unique(phones_adj):
            ph = int(phone_id.item())
            mask = phones_adj == phone_id
            N_q = mask.sum().item()

            entry = bank.get(ph, None)
            tgt_l24 = entry['l24'] if entry else fallback['l24']
            tgt_l6 = entry['l6'] if entry else fallback['l6']

            N_t = tgt_l24.shape[0]
            if N_t == 0:
                continue
            if N_t == 1:
                h_anon[mask] = tgt_l6[0].expand(N_q, -1)
                continue

            tgt_l6_norm = tgt_l6 / (tgt_l6.norm(dim=-1, keepdim=True) + 1e-8)
            tgt_l24_norm = tgt_l24 / (tgt_l24.norm(dim=-1, keepdim=True) + 1e-8)

            dist_l6 = torch.cdist(q_l6_norm[mask], tgt_l6_norm)
            dist_l24 = torch.cdist(q_l24_norm[mask], tgt_l24_norm)

            # 音素级自适应alpha (当候选不足时回退到全局alpha)
            if self.use_adaptive_alpha and N_t >= 5:
                ph_alpha = PHONE_ALPHA_MAP.get(ph, self.alpha)
            else:
                ph_alpha = self.alpha
            dist_mix = ph_alpha * dist_l6 + (1 - ph_alpha) * dist_l24

            k = min(self.top_k, N_t)
            topk_idx = dist_mix.topk(k, dim=-1, largest=False).indices  # [N_q, k]

            rand_col = torch.randint(0, k, (N_q,), device=self.device)
            selected_idx = topk_idx[torch.arange(N_q, device=self.device), rand_col]
            h_anon[mask] = tgt_l6[selected_idx]

        return h_anon