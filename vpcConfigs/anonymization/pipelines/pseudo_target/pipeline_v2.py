"""VPC 2024 Pipeline — Global Phonetic Joint-Score Anonymization"""
import sys
import io
import subprocess
import torch
import torch.nn.functional as F
import torchaudio
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, '/root/autodl-tmp/anon_test')


class PseudoTargetPipelineV2:
    def __init__(self, config, force_compute=False, devices=None):
        self.config = config
        self.force_compute = force_compute
        self.device = devices[0] if devices else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.anon_suffix = config['modules']['anon_suffix']

        self.mode = config['modules'].get('mode', 'mix')
        self.dur_weight = config['modules'].get('dur_weight', 0.3)
        self.top_n = config['modules'].get('top_n', 100)
        self.top_k = config['modules'].get('top_k', 8)
        self.temperature = config['modules'].get('temperature', 0.1)
        self.lambda_id = config['modules'].get('lambda_id', 0.5)

        self._load_models()

    def _load_bank(self, path):
        bank_raw = torch.load(str(path), map_location='cpu')
        bank = {}
        for ph, data in bank_raw.items():
            l6  = data['l6'].to(self.device)
            l12 = data['l12'].to(self.device)
            l24 = data['l24'].to(self.device)
            bank[ph] = {
                'l6': l6,
                'l12': l12,
                'l12_norm': F.normalize(l12, dim=-1),
                'l24': l24,
            }
        fallback_l6  = torch.cat([v['l6']  for v in bank.values()], dim=0)
        fallback_l12 = torch.cat([v['l12'] for v in bank.values()], dim=0)
        fallback_l24 = torch.cat([v['l24'] for v in bank.values()], dim=0)
        fallback = {
            'l6': fallback_l6,
            'l12': fallback_l12,
            'l12_norm': F.normalize(fallback_l12, dim=-1),
            'l24': fallback_l24,
        }
        del bank_raw
        return bank, fallback

    def _load_models(self):
        ckpt_dir = Path('/root/autodl-tmp/anon_test/checkpoints')

        from models.ssl.wrappers import WavLMSSLExtractor
        from models.phone_predictor.predictor import PhonePredictor, DurationPredictor
        from models.vocoder.hifigan import HiFiGAN

        self.wavlm = WavLMSSLExtractor(
            ckpt_path=str(ckpt_dir / 'WavLM-Large.pt'), layer=6, device=self.device
        )
        self.phone_predictor = PhonePredictor.load(
            str(ckpt_dir / 'phone_decoder.pt'), device=self.device
        )
        self.duration_predictor = DurationPredictor.load(
            str(ckpt_dir / 'duration_decoder.pt'), device=self.device
        )
        self.vocoder = HiFiGAN.load(
            checkpoint_path=str(ckpt_dir / 'hifigan.pt'), device=self.device
        )

        data_dir = self.config['modules'].get('data_dir')
        need_gendered = self.mode in ('same', 'cross', 'all')
        need_mix = self.mode in ('mix', 'all')

        self.banks = {}

        if need_gendered:
            bm, fm = self._load_bank(f"{data_dir}/pseudo_bank_v2.gender-m.pt")
            bf, ff = self._load_bank(f"{data_dir}/pseudo_bank_v2.gender-f.pt")
            self.banks['m'] = (bm, fm)
            self.banks['f'] = (bf, ff)
            print(f"  Bank M: {len(bm)} phones, Bank F: {len(bf)} phones")

        if need_mix:
            bmix, fmix = self._load_bank(f"{data_dir}/pseudo_bank_v2.pt")
            self.banks['mix'] = (bmix, fmix)
            print(f"  Bank Mix: {len(bmix)} phones")

        print(f"[V2] 模型加载完成 (mode={self.mode}, λ={self.lambda_id}, T={self.temperature})")

    def _select_bank(self, src_gender):
        if self.mode == 'same':
            return self.banks[src_gender]
        elif self.mode == 'cross':
            key = 'f' if src_gender == 'm' else 'm'
            return self.banks[key]
        else:
            return self.banks['mix']

    # ================================================================
    # Dataset Processing (VPC Interface)
    # ================================================================

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
                    utt2gender[parts[0]] = spk2gender.get(parts[1], 'm')
        return utt2gender

    def _process_dataset(self, dataset_name, dataset_path):
        output_dir = dataset_path.parent / f"{dataset_name}{self.anon_suffix}"
        wav_dir = output_dir / self.config['results_dir']
        wav_dir.mkdir(parents=True, exist_ok=True)

        wav_scp = dataset_path / 'wav.scp'
        if not wav_scp.exists():
            print(f"Warning: {wav_scp} not found")
            return

        utt2gender = self._load_spk2gender(dataset_path)

        entries = []
        for line in wav_scp.read_text().strip().split('\n'):
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                utt_id, wav_ref = parts
                if wav_ref.rstrip().endswith('|'):
                    entries.append((utt_id, ('pipe', wav_ref.rstrip()[:-1].strip())))
                else:
                    entries.append((utt_id, ('path', wav_ref)))

        for utt_id, (ref_type, wav_ref) in tqdm(entries, desc=f"V2 {dataset_name}"):
            output_path = wav_dir / f"{utt_id}.wav"
            if output_path.exists() and not self.force_compute:
                continue
            src_gender = utt2gender.get(utt_id, 'm')
            if ref_type == 'pipe':
                self._anonymize_from_pipe(wav_ref, output_path, src_gender)
            else:
                wav_path = Path(wav_ref)
                if not wav_path.is_absolute():
                    wav_path = Path(self.config['data_dir']).parent / wav_path
                if wav_path.exists():
                    self._anonymize_file(wav_path, output_path, src_gender)

    # ================================================================
    # Audio Loading
    # ================================================================

    def _anonymize_from_pipe(self, pipe_cmd, output_path, src_gender='m'):
        proc = subprocess.run(pipe_cmd, shell=True, capture_output=True)
        if proc.returncode != 0:
            return
        waveform, sr = torchaudio.load(io.BytesIO(proc.stdout))
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        waveform = waveform.mean(dim=0, keepdim=True).to(self.device)
        self._anonymize_waveform(waveform, output_path, src_gender)

    def _anonymize_file(self, input_path, output_path, src_gender='m'):
        waveform, sr = torchaudio.load(input_path)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        waveform = waveform.mean(dim=0, keepdim=True).to(self.device)
        self._anonymize_waveform(waveform, output_path, src_gender)

    # ================================================================
    # Core: Joint-Score Anonymization
    # ================================================================

    @torch.no_grad()
    def _anonymize_waveform(self, waveform, output_path, src_gender='m'):
        multi_feats = self.wavlm.forward_multi_layer(waveform, layers=[6, 12, 24])
        l12 = multi_feats[12].squeeze(0)
        l24 = multi_feats[24].squeeze(0)
        phones = self.phone_predictor(l24).cpu().numpy()

        h_anon = self._joint_score_anonymize(l12, l24, phones, src_gender)

        wav = self.vocoder(h_anon.unsqueeze(0)).squeeze()
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        torchaudio.save(str(output_path), wav.cpu(), 16000)

    def _joint_score_anonymize(self, l12, l24, src_phones, src_gender):
        bank, fallback = self._select_bank(src_gender)

        e_src = F.normalize(l12.mean(dim=0, keepdim=True), dim=-1)

        unique_phones, phone_durations = [], []
        cur, cnt = src_phones[0], 1
        for i in range(1, len(src_phones)):
            if src_phones[i] == cur:
                cnt += 1
            else:
                unique_phones.append(cur)
                phone_durations.append(cnt)
                cur, cnt = src_phones[i], 1
        unique_phones.append(cur)
        phone_durations.append(cnt)

        phones_t = torch.tensor(unique_phones, dtype=torch.long, device=self.device)
        dur_true = torch.tensor(phone_durations, dtype=torch.float32, device=self.device)

        dur_pred = self.duration_predictor(phones_t).squeeze(0)
        dur_anon = (self.dur_weight * dur_pred + (1 - self.dur_weight) * dur_true).clamp(min=1).round().long()

        adj_l24, adj_l12, adj_phones = [], [], []
        idx = 0
        for ph, orig, new in zip(unique_phones, phone_durations, dur_anon):
            end = idx + orig
            new_len = new.item()
            if new_len == 0:
                idx = end
                continue
            if orig == 1:
                adj_l24.append(l24[idx:idx+1].expand(new_len, -1))
                adj_l12.append(l12[idx:idx+1].expand(new_len, -1))
            else:
                t = torch.linspace(0, 1, new_len, device=self.device)
                src_idx_f = t * (orig - 1)
                idx_low = src_idx_f.long().clamp(max=orig - 2)
                idx_high = idx_low + 1
                w = (src_idx_f - idx_low.float()).unsqueeze(-1)
                adj_l24.append((1 - w) * l24[idx + idx_low] + w * l24[idx + idx_high])
                adj_l12.append((1 - w) * l12[idx + idx_low] + w * l12[idx + idx_high])
            adj_phones.append(torch.full((new_len,), ph, dtype=torch.long, device=self.device))
            idx = end

        l24_adj = torch.cat(adj_l24, dim=0)
        l12_adj = torch.cat(adj_l12, dim=0)
        phones_adj = torch.cat(adj_phones, dim=0)

        T_out = l24_adj.shape[0]
        h_anon = torch.zeros(T_out, 1024, device=self.device)

        for phone_id in torch.unique(phones_adj):
            ph = int(phone_id.item())
            mask = (phones_adj == phone_id)
            N_q = mask.sum().item()

            query_l24 = l24_adj[mask]

            if ph in bank:
                tgt_l6      = bank[ph]['l6']
                tgt_l12_norm = bank[ph]['l12_norm']
                tgt_l24     = bank[ph]['l24']
            else:
                tgt_l6      = fallback['l6']
                tgt_l12_norm = fallback['l12_norm']
                tgt_l24     = fallback['l24']

            N_t = tgt_l24.shape[0]
            if N_t == 0:
                continue
            if N_t <= self.top_k:
                h_anon[mask] = tgt_l6.mean(dim=0).expand(N_q, -1)
                continue

            actual_n = min(self.top_n, N_t)

            topn_vals, topn_idx = torch.cdist(query_l24, tgt_l24).topk(actual_n, largest=False)

            cand_l6       = tgt_l6[topn_idx]
            cand_l12_norm = tgt_l12_norm[topn_idx]

            sim_l12 = (cand_l12_norm * e_src.unsqueeze(1)).sum(dim=-1)

            d_l24_normed = topn_vals / (topn_vals.max(dim=-1, keepdim=True).values + 1e-8)
            sim_l12_shifted = (sim_l12 + 1.0) / 2.0

            score = d_l24_normed + self.lambda_id * sim_l12_shifted

            actual_k = min(self.top_k, actual_n)
            topk_scores, topk_local_idx = score.topk(actual_k, largest=False)

            topk_l6 = torch.gather(
                cand_l6, dim=1,
                index=topk_local_idx.unsqueeze(-1).expand(-1, -1, 1024)
            )

            weights = F.softmax(-topk_scores / self.temperature, dim=-1)
            h_anon[mask] = (topk_l6 * weights.unsqueeze(-1)).sum(dim=1)

        return h_anon