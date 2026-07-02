"""VPC 2024 Pipeline for Pseudo-Target Anonymization"""
import sys
import io
import subprocess
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, '/root/autodl-tmp/anon_test')

class PseudoTargetPipeline:
    def __init__(self, config, force_compute=False, devices=None):
        self.config = config
        self.force_compute = force_compute
        self.device = devices[0] if devices else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.anon_suffix = config['modules']['anon_suffix']
        self._load_models()

    def _load_models(self):
        base_dir = Path('/root/autodl-tmp/anon_test')
        ckpt_dir = base_dir / 'checkpoints'

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

        # 加载男女Bank (新格式: {phone_id: {'l6': tensor, 'l12': tensor}})
        bank_male = self.config['modules'].get('bank_male')
        bank_female = self.config['modules'].get('bank_female')
        self.bank_male = torch.load(bank_male, map_location=self.device)
        self.bank_female = torch.load(bank_female, map_location=self.device)

        # 构建全局后备池
        self.fallback_male = {
            'l6': torch.cat([v['l6'] for v in self.bank_male.values()], dim=0),
            'l24': torch.cat([v['l24'] for v in self.bank_male.values()], dim=0)
        }
        self.fallback_female = {
            'l6': torch.cat([v['l6'] for v in self.bank_female.values()], dim=0),
            'l24': torch.cat([v['l24'] for v in self.bank_female.values()], dim=0)
        }

        self.target_gender = self.config['modules'].get('target_gender', 'cross')
        self.k = self.config['modules'].get('k', 4)
        self.dur_weight = self.config['modules'].get('dur_weight', 0.3)

    def run_anonymization_pipeline(self, datasets):
        for dataset_name, dataset_path in datasets.items():
            self._process_dataset(dataset_name, dataset_path)

    def _load_spk2gender(self, dataset_path):
        """从 spk2gender + utt2spk 构建 utt_id -> gender 映射"""
        spk2gender = {}
        spk2gender_file = dataset_path / 'spk2gender'
        if spk2gender_file.exists():
            for line in spk2gender_file.read_text().strip().split('\n'):
                parts = line.strip().split()
                if len(parts) == 2:
                    spk2gender[parts[0]] = parts[1]  # 'm' or 'f'

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
            print(f"Warning: {wav_scp} not found, skipping {dataset_name}")
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

        for utt_id, (ref_type, wav_ref) in tqdm(entries, desc=f"Anonymizing {dataset_name}"):
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
                if not wav_path.exists():
                    print(f"Warning: {wav_path} not found, skipping {utt_id}")
                    continue
                self._anonymize_file(wav_path, output_path, src_gender)

    def _anonymize_from_pipe(self, pipe_cmd, output_path, src_gender='m'):
        """读取 pipe 格式音频（如 flac -c -d -s xxx.flac）并匿名化"""
        proc = subprocess.run(pipe_cmd, shell=True, capture_output=True)
        if proc.returncode != 0:
            print(f"Warning: pipe command failed: {pipe_cmd}")
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

    def _anonymize_waveform(self, waveform, output_path, src_gender='m'):
        with torch.no_grad():
            multi_feats = self.wavlm.forward_multi_layer(waveform, layers=[6, 24])
            l6_feats = multi_feats[6].squeeze(0)
            l24_feats = multi_feats[24].squeeze(0)
            phones = self.phone_predictor(l24_feats).cpu().numpy()

        h_anon = self._knn_anonymize(l6_feats, l24_feats, phones, src_gender)

        with torch.no_grad():
            anon_wav = self.vocoder(h_anon.unsqueeze(0).to(self.device)).squeeze()

        if anon_wav.dim() == 1:
            anon_wav = anon_wav.unsqueeze(0)
        torchaudio.save(str(output_path), anon_wav.cpu(), 16000)

    def _detect_gender(self, l24_feats):
        """简单性别检测：基于特征均值"""
        mean_energy = l24_feats.mean(dim=0).mean().item()
        return 'm' if mean_energy > 0 else 'f'

    def _knn_anonymize(self, src_l6, src_l24, src_phones, src_gender, tau=3.0):
        # 性别路由
        if self.target_gender == 'cross':
            tgt_gender = 'f' if src_gender == 'm' else 'm'
        else:
            tgt_gender = src_gender

        bank = self.bank_male if tgt_gender == 'm' else self.bank_female
        fallback = self.fallback_male if tgt_gender == 'm' else self.fallback_female

        # 预计算源说话人 L6 质心（整句均值，只算一次）
        e_src = src_l6.mean(dim=0)  # [1024]

        unique_phones, phone_durations = [], []
        current_phone, current_count = src_phones[0], 1

        for i in range(1, len(src_phones)):
            if src_phones[i] == current_phone:
                current_count += 1
            else:
                unique_phones.append(current_phone)
                phone_durations.append(current_count)
                current_phone, current_count = src_phones[i], 1
        unique_phones.append(current_phone)
        phone_durations.append(current_count)

        phones_tensor = torch.tensor(unique_phones, dtype=torch.long, device=self.device)
        durations_tensor = torch.tensor(phone_durations, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            pred_durations = self.duration_predictor(phones_tensor).squeeze(-1)
        n_frames = (self.dur_weight * pred_durations + (1 - self.dur_weight) * durations_tensor).clamp(min=1).long()

        adjusted_l24, adjusted_phones = [], []
        feat_idx_start = 0

        for phone_id, orig_dur, new_dur in zip(unique_phones, phone_durations, n_frames):
            feat_idx_end = feat_idx_start + orig_dur - 1
            indices = torch.linspace(feat_idx_start, feat_idx_end, new_dur.item(), dtype=torch.long, device=self.device)
            adjusted_l24.append(src_l24[indices])
            adjusted_phones.append(torch.full((new_dur.item(),), phone_id, dtype=torch.long, device=self.device))
            feat_idx_start = feat_idx_end + 1

        src_l24_adj = torch.cat(adjusted_l24, dim=0)
        src_phones_adj = torch.cat(adjusted_phones, dim=0)

        h_anon = torch.zeros(src_l24_adj.shape[0], 1024, device=self.device)

        for phone_id in torch.unique(src_phones_adj):
            ph_int = int(phone_id.item())
            mask = src_phones_adj == phone_id
            query_l24 = src_l24_adj[mask]

            if ph_int in bank:
                tgt_l6  = bank[ph_int]['l6']
                tgt_l24 = bank[ph_int]['l24']
            else:
                tgt_l6  = fallback['l6']
                tgt_l24 = fallback['l24']

            N_query = query_l24.shape[0]
            N_tgt   = tgt_l24.shape[0]
            sample_k = self.k

            # Stage 1: L24 语义锁定，top-50（桶不足时按比例缩减）
            pool_size = max(sample_k, min(50, max(int(N_tgt * 0.3), sample_k)) if N_tgt < 100 else 50)
            pool_size = min(pool_size, N_tgt)

            if N_tgt < sample_k:
                # 桶太小，直接均值
                h_anon[mask] = tgt_l6.mean(dim=0).expand(N_query, -1)
                continue

            dists = torch.cdist(query_l24, tgt_l24)
            top_idx = dists.topk(pool_size, largest=False).indices  # [N_query, pool_size]

            # Stage 3: 音色去身份化 —— 逐查询帧选距源说话人最远的 k 帧
            result = torch.zeros(N_query, 1024, device=self.device)
            for i in range(N_query):
                cand_l6 = tgt_l6[top_idx[i]]  # [pool_size, 1024]
                sims = torch.cosine_similarity(cand_l6, e_src.unsqueeze(0), dim=-1)  # [pool_size]

                # 取相似度最低的 k 帧（距源说话人最远）
                far_idx = sims.argsort()[:sample_k]
                selected = cand_l6[far_idx]  # [k, 1024]

                # 非对称 Dirichlet：反相似度作为浓度参数
                alpha = (tau * (1 - sims[far_idx])).clamp(min=1e-3)
                weights = torch.distributions.Dirichlet(alpha).sample().unsqueeze(-1)  # [k, 1]
                result[i] = (selected * weights).sum(dim=0)

            h_anon[mask] = result

        return h_anon.cpu()
