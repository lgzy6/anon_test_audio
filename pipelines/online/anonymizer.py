# pipelines/online/anonymizer.py

"""
Speech Anonymizer - 语音匿名化主入口
融合 SAMM + Eta-WavLM + Private kNN-VC
"""

import torch
import torchaudio
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

# 内部模块
from models.ssl.wrappers import WavLMSSLExtractor
from models.eta_wavlm.projector import EtaWavLMProjector
from models.samm.codebook import SAMMCodebook
from models.samm.pattern_matrix import PatternMatrix
from models.samm.masking import ProsodyAwareMasking
from models.phone_predictor.predictor import PhonePredictor, DurationPredictor
from models.knn_vc.retriever import ConstrainedKNNRetriever
from models.knn_vc.duration import DurationAnonymizer, DurationAdjuster


@dataclass
class AnonymizerConfig:
    """匿名化器配置"""

    # 路径配置
    wavlm_path: str = "checkpoints/WavLM-Large.pt"
    subspace_path: str = "data/samm_anon/checkpoints/speaker_subspace.pt"
    codebook_path: str = "data/samm_anon/checkpoints/codebook.pt"
    pattern_matrix_path: str = "data/samm_anon/checkpoints/pattern_matrix.pt"
    phone_predictor_path: str = "checkpoints/phone_decoder.pt"
    duration_predictor_path: Optional[str] = "checkpoints/duration_decoder.pt"
    target_pool_path: str = "data/samm_anon/checkpoints/target_pool/"
    vocoder_path: str = "checkpoints/hifigan.pt"

    # 模型参数
    ssl_layer: int = 15
    device: str = "cuda"

    # Eta-WavLM 参数 (v2.1 新增)
    use_eta_wavlm: bool = True  # 是否启用 Eta-WavLM 投影

    # SAMM 参数
    mask_prob: float = 0.10
    span_mask_prob: float = 0.15

    # 时长匿名化参数
    use_duration_predictor: bool = True
    duration_predictor_weight: float = 0.7
    duration_noise_std: float = 0.1

    # kNN-VC 参数
    use_phone_clusters: bool = True
    num_clusters: int = 8
    knn_k: int = 4
    symbol_constraint: str = 'soft'

    # kNN-VC 优化参数 (v2.1 新增)
    use_top1: bool = True  # 使用 Top-1 而非加权平均
    use_cosine: bool = True  # 使用余弦相似度而非 L2 距离

    # 目标选择
    target_gender: str = 'same'  # 'same', 'cross', 'M', 'F'

    @classmethod
    def from_yaml(cls, path: str) -> 'AnonymizerConfig':
        """从 YAML 文件加载配置"""
        import yaml
        with open(path, 'r') as f:
            config = yaml.safe_load(f)

        paths = config.get('paths', {})
        ssl = config.get('ssl', {})
        samm = config.get('samm', {}).get('masking', {})
        duration = config.get('duration', {})
        knn_vc = config.get('knn_vc', {})
        target_selection = config.get('target_selection', {})
        eta_wavlm = config.get('eta_wavlm', {})

        # 获取 checkpoints_dir 用于构建默认路径
        checkpoints_dir = paths.get('checkpoints_dir', 'checkpoints')

        params = {
            # 路径配置 - 兼容多种配置格式
            'wavlm_path': paths.get('wavlm_checkpoint', paths.get('wavlm', 'checkpoints/WavLM-Large.pt')),
            'subspace_path': paths.get('subspace', f'{checkpoints_dir}/speaker_subspace.pt'),
            'codebook_path': paths.get('codebook', f'{checkpoints_dir}/codebook.pt'),
            'pattern_matrix_path': paths.get('pattern_matrix', f'{checkpoints_dir}/pattern_matrix.pt'),
            'phone_predictor_path': paths.get('phone_predictor', config.get('phone_predictor', {}).get('checkpoint', 'checkpoints/phone_decoder.pt')),
            'duration_predictor_path': paths.get('duration_predictor', 'checkpoints/duration_decoder.pt'),
            'target_pool_path': paths.get('target_pool', f'{checkpoints_dir}/target_pool/'),
            'vocoder_path': paths.get('vocoder', 'checkpoints/hifigan.pt'),

            # SSL 配置
            'ssl_layer': ssl.get('layer', 15),
            'device': config.get('device', 'cuda'),

            # Eta-WavLM 配置
            'use_eta_wavlm': eta_wavlm.get('enabled', True),

            # SAMM 配置
            'mask_prob': samm.get('token_mask_ratio', samm.get('random_mask_prob', 0.10)),
            'span_mask_prob': samm.get('span_mask_ratio', samm.get('span_mask_prob', 0.15)),

            # Duration 配置
            'use_duration_predictor': duration.get('use_predictor', True),
            'duration_predictor_weight': duration.get('predictor_weight', 0.7),
            'duration_noise_std': duration.get('noise_std', 0.1),

            # kNN-VC 配置 - 兼容多种格式
            'use_phone_clusters': knn_vc.get('use_phone_constraint', knn_vc.get('retrieval', {}).get('use_phone_clusters', True)),
            'num_clusters': knn_vc.get('num_clusters', knn_vc.get('retrieval', {}).get('num_clusters', 8)),
            'knn_k': knn_vc.get('k', knn_vc.get('k_neighbors', 4)),
            'symbol_constraint': knn_vc.get('symbol_constraint', knn_vc.get('constraints', {}).get('symbol', 'soft')),
            'use_top1': knn_vc.get('use_top1', True),
            'use_cosine': knn_vc.get('use_cosine', True),

            # Target 配置
            'target_gender': target_selection.get('gender', 'same'),
        }
        return cls(**params)


class SpeechAnonymizer:
    """语音匿名化器"""

    def __init__(self, config: AnonymizerConfig):
        self.config = config
        self.device = config.device

        print("Initializing Speech Anonymizer...")

        # Stage 1: SSL 特征提取
        print("  Loading WavLM...")
        self.ssl_extractor = WavLMSSLExtractor(
            ckpt_path=config.wavlm_path,
            layer=config.ssl_layer,
            device=config.device,
        )

        # Stage 2: Eta-WavLM 投影
        print("  Loading Eta-WavLM projector...")
        self.projector = EtaWavLMProjector(
            checkpoint_path=config.subspace_path,
            device=config.device,
        )

        # Stage 3: SAMM 组件
        print("  Loading SAMM components...")
        self.codebook = SAMMCodebook.load(config.codebook_path, device=config.device)
        self.pattern_matrix = PatternMatrix.load(config.pattern_matrix_path, device=config.device)
        self.masking = ProsodyAwareMasking(
            token_mask_ratio=config.mask_prob,
            span_mask_ratio=config.span_mask_prob,
        )

        # Stage 3': Phone Predictor
        print("  Loading Phone Predictor...")
        self.phone_predictor = PhonePredictor.load(
            config.phone_predictor_path,
            device=config.device
        )

        # Duration 组件
        self.duration_predictor = None
        if config.use_duration_predictor and config.duration_predictor_path:
            print("  Loading Duration Predictor...")
            self.duration_predictor = DurationPredictor.load(
                config.duration_predictor_path,
                device=config.device
            )

        self.duration_anonymizer = DurationAnonymizer(
            predictor=self.duration_predictor,
            predictor_weight=config.duration_predictor_weight,
            noise_std=config.duration_noise_std,
        )

        # Stage 4: kNN-VC
        print("  Loading Target Pool and kNN Retriever...")
        self.retriever = ConstrainedKNNRetriever(
            target_pool_path=config.target_pool_path,
            k=config.knn_k,
            num_clusters=config.num_clusters,
            use_phone_clusters=config.use_phone_clusters,
            use_top1=config.use_top1,  # 新增: Top-1 策略
            use_cosine=config.use_cosine,  # 新增: 余弦相似度
            device=config.device,
        )

        # Stage 5: Vocoder (延迟加载)
        self.vocoder = None
        self.vocoder_path = config.vocoder_path

        print("Initialization complete!")

    def _load_vocoder(self):
        """延迟加载 vocoder"""
        if self.vocoder is None:
            print("  Loading Vocoder...")
            from models.vocoder.hifigan import HiFiGAN
            self.vocoder = HiFiGAN.load(self.vocoder_path, device=self.device)

    def _determine_gender(self, source_gender: Optional[str] = None) -> int:
        """确定目标性别"""
        target = self.config.target_gender
        if target == 'M':
            return 0
        elif target == 'F':
            return 1
        elif target == 'same':
            return 0 if source_gender == 'M' else 1
        elif target == 'cross':
            return 1 if source_gender == 'M' else 0
        else:
            return 0

    @torch.inference_mode()
    def anonymize(
        self,
        waveform: torch.Tensor,
        source_gender: Optional[str] = None,
        return_intermediates: bool = False,
    ) -> Dict[str, Any]:
        """执行匿名化"""
        if waveform.dim() == 2:
            waveform = waveform.squeeze(0)
        waveform = waveform.to(self.device) 
        intermediates = {}

        # Stage 1: SSL 特征提取
        h = self.ssl_extractor(waveform.unsqueeze(0)).squeeze(0)
        if return_intermediates:
            intermediates['h_ssl'] = h.cpu()

        # Stage 2: Eta-WavLM 说话人去除 (v2.1: 可配置)
        if self.config.use_eta_wavlm:
            h_clean = self.projector(h)
        else:
            h_clean = h
        if return_intermediates:
            intermediates['h_clean'] = h_clean.cpu()

        # Stage 3.1: SAMM 符号分配
        z = self.codebook.encode(h_clean)
        if return_intermediates:
            intermediates['symbols'] = z.cpu()

        # Stage 3.1': 音素预测
        phones = self.phone_predictor(h)
        if return_intermediates:
            intermediates['phones'] = phones.cpu()

        # Stage 3.2: 获取音素时长
        phone_ids, true_durations = self.phone_predictor.get_phone_durations(phones)
        phone_segments = self.phone_predictor.get_phone_segments(phones)

        # Stage 3.3: 时长匿名化
        anon_durations = self.duration_anonymizer.anonymize(
            phone_ids.to(self.device),
            true_durations.to(self.device),
        )

        # Stage 3.4: 符号掩码 (v3.0: 已禁用 - 会污染 Query 导致 WER > 100%)
        # z_masked, _, mask_indicator = self.masking(
        #     z, torch.ones_like(z, dtype=torch.float)
        # )

        # Stage 3.5: 模式正则化 (v3.0: 已禁用)
        # z_smooth = self.pattern_matrix.smooth_sequence(z_masked, mask_indicator)

        # Stage 4.1: 约束 kNN 检索 (v3.0: 不使用符号约束!)
        target_gender = self._determine_gender(source_gender)
        # 关键修改: symbols=None，不再使用被污染的符号进行约束
        h_anon = self.retriever.retrieve_batch(h_clean, phones, None, target_gender)

        # Stage 4.2: 时长调整
        h_anon_adjusted = DurationAdjuster.adjust_features(
            h_anon, phone_segments, anon_durations
        )

        # Stage 5: Vocoder 合成
        result = {'features': h_anon_adjusted.cpu()}

        if self.vocoder_path and Path(self.vocoder_path).exists():
            try:
                self._load_vocoder()
                waveform_anon = self.vocoder(h_anon_adjusted)
                result['waveform'] = waveform_anon.cpu()
            except Exception as e:
                print(f"Vocoder synthesis failed: {e}")

        if return_intermediates:
            result['intermediates'] = intermediates

        return result

    def anonymize_file(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        source_gender: Optional[str] = None,
        save_original: bool = True,
    ) -> Dict[str, Any]:
        """匿名化音频文件"""
        waveform, sr = torchaudio.load(input_path)

        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # 保存原始音频
        if save_original and output_path:
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            original_path = output_dir / "original.wav"
            torchaudio.save(str(original_path), waveform, 16000)
            print(f"Saved original audio to {original_path}")

        result = self.anonymize(waveform.squeeze(0), source_gender)

        if output_path and 'waveform' in result:
            torchaudio.save(output_path, result['waveform'].unsqueeze(0), 16000)
            print(f"Saved anonymized audio to {output_path}")

        return result
