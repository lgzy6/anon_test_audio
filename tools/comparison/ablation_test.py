#!/usr/bin/env python3
"""
逐层消融测试 - 诊断语义丢失的具体阶段
Ablation Study: 在每个推理阶段后评估语义保留
"""

import sys
import torch
import torchaudio
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Optional, List
import json

sys.path.insert(0, str(Path(__file__).parent))


@dataclass
class StageResult:
    """每个阶段的结果"""
    stage_name: str
    stage_desc: str
    features_shape: tuple
    wer: Optional[float] = None
    transcription: Optional[str] = None
    audio_path: Optional[str] = None
    success: bool = True
    error: Optional[str] = None


class AblationTester:
    """消融测试器"""

    def __init__(self, config_path: str, device: str = 'cuda'):
        from pipelines.online.anonymizer import AnonymizerConfig

        self.config = AnonymizerConfig.from_yaml(config_path)
        self.device = device
        self.results: List[StageResult] = []

        # 加载模型
        self._load_models()

    def _load_models(self):
        """加载所有模型"""
        from models.ssl.wrappers import WavLMSSLExtractor
        from models.eta_wavlm.projector import EtaWavLMProjector
        from models.samm.codebook import SAMMCodebook
        from models.samm.pattern_matrix import PatternMatrix
        from models.samm.masking import ProsodyAwareMasking
        from models.phone_predictor.predictor import PhonePredictor, DurationPredictor
        from models.knn_vc.retriever import ConstrainedKNNRetriever
        from models.knn_vc.duration import DurationAnonymizer, DurationAdjuster

        print("加载模型...")

        # SSL
        self.ssl_extractor = WavLMSSLExtractor(
            ckpt_path=self.config.wavlm_path,
            layer=self.config.ssl_layer,
            device=self.device,
        )

        # Eta-WavLM
        self.projector = EtaWavLMProjector(
            checkpoint_path=self.config.subspace_path,
            device=self.device,
        )

        # SAMM
        self.codebook = SAMMCodebook.load(self.config.codebook_path, device=self.device)
        self.pattern_matrix = PatternMatrix.load(self.config.pattern_matrix_path, device=self.device)
        self.masking = ProsodyAwareMasking(
            token_mask_ratio=self.config.mask_prob,
            span_mask_ratio=self.config.span_mask_prob,
        )

        # Phone
        self.phone_predictor = PhonePredictor.load(
            self.config.phone_predictor_path,
            device=self.device
        )

        # Duration
        self.duration_predictor = None
        if self.config.use_duration_predictor and self.config.duration_predictor_path:
            self.duration_predictor = DurationPredictor.load(
                self.config.duration_predictor_path,
                device=self.device
            )

        self.duration_anonymizer = DurationAnonymizer(
            predictor=self.duration_predictor,
            predictor_weight=self.config.duration_predictor_weight,
            noise_std=self.config.duration_noise_std,
        )

        # kNN
        self.retriever = ConstrainedKNNRetriever(
            target_pool_path=self.config.target_pool_path,
            k=self.config.knn_k,
            num_clusters=self.config.num_clusters,
            use_phone_clusters=self.config.use_phone_clusters,
            use_top1=self.config.use_top1,
            use_cosine=self.config.use_cosine,
            device=self.device,
        )

        # Vocoder (延迟加载)
        self.vocoder = None
        self.vocoder_path = self.config.vocoder_path

        print("✓ 模型加载完成")

    def _load_vocoder(self):
        """加载 vocoder"""
        if self.vocoder is None:
            print("  加载 Vocoder...")
            from models.vocoder.hifigan import HiFiGAN
            self.vocoder = HiFiGAN.load(self.vocoder_path, device=self.device)

    def _synthesize_audio(self, features: torch.Tensor, output_path: Path) -> bool:
        """合成音频"""
        try:
            self._load_vocoder()
            with torch.no_grad():
                waveform = self.vocoder(features.to(self.device))
            torchaudio.save(str(output_path), waveform.cpu().unsqueeze(0), 16000)
            return True
        except Exception as e:
            print(f"  ✗ 合成失败: {e}")
            return False

    def _transcribe_audio(self, audio_path: Path) -> Dict:
        """转录音频"""
        import whisper

        if not hasattr(self, 'whisper_model'):
            self.whisper_model = whisper.load_model("base")

        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)
        else:
            waveform = waveform.squeeze(0)

        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)

        audio = waveform.numpy().astype(np.float32)
        result = self.whisper_model.transcribe(audio, language="en", verbose=False)

        return {
            'text': result['text'].strip(),
            'language': result.get('language', 'en')
        }

    def _calculate_wer(self, reference: str, hypothesis: str) -> float:
        """计算 WER"""
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()

        if len(ref_words) == 0:
            return 0.0 if len(hyp_words) == 0 else float('inf')

        # Levenshtein distance
        d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]

        for i in range(len(ref_words) + 1):
            d[i][0] = i
        for j in range(len(hyp_words) + 1):
            d[0][j] = j

        for i in range(1, len(ref_words) + 1):
            for j in range(1, len(hyp_words) + 1):
                if ref_words[i-1] == hyp_words[j-1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    d[i][j] = min(d[i-1][j-1] + 1, d[i][j-1] + 1, d[i-1][j] + 1)

        return d[len(ref_words)][len(hyp_words)] / len(ref_words)

    def test_stage(self, stage_name: str, stage_desc: str, features: torch.Tensor,
                   output_dir: Path, reference_text: str) -> StageResult:
        """测试单个阶段"""
        print(f"\n{'='*70}")
        print(f"Stage: {stage_name}")
        print(f"{'='*70}")
        print(f"描述: {stage_desc}")
        print(f"特征形状: {features.shape}")

        result = StageResult(
            stage_name=stage_name,
            stage_desc=stage_desc,
            features_shape=features.shape
        )

        # 合成音频
        audio_path = output_dir / f"{stage_name}_audio.wav"
        print(f"合成音频: {audio_path.name}")

        if not self._synthesize_audio(features, audio_path):
            result.success = False
            result.error = "合成失败"
            return result

        result.audio_path = str(audio_path)

        # 转录
        print(f"转录音频...")
        try:
            transcription = self._transcribe_audio(audio_path)
            result.transcription = transcription['text']
            print(f"  转录文本: \"{result.transcription}\"")

            # 计算 WER
            wer = self._calculate_wer(reference_text, result.transcription)
            result.wer = wer
            print(f"  WER: {wer:.2%}")

        except Exception as e:
            print(f"  ✗ 转录失败: {e}")
            result.success = False
            result.error = f"转录失败: {e}"

        return result

    def run_ablation(self, audio_path: str, output_dir: str = None) -> List[StageResult]:
        """运行完整消融测试"""
        if output_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = f"outputs/ablation_{timestamp}"

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("\n" + "="*70)
        print("逐层消融测试")
        print("="*70)
        print(f"输入音频: {audio_path}")
        print(f"输出目录: {output_dir}")
        print(f"配置: use_eta_wavlm={self.config.use_eta_wavlm}, "
              f"use_top1={self.config.use_top1}, ssl_layer={self.config.ssl_layer}")

        # 加载原始音频
        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        waveform = waveform.to(self.device)

        # 保存原始音频
        original_path = output_path / "stage0_original.wav"
        torchaudio.save(str(original_path), waveform.cpu(), 16000)

        # 获取原始转录
        print("\n获取原始转录...")
        original_trans = self._transcribe_audio(original_path)
        reference_text = original_trans['text']
        print(f"原始文本: \"{reference_text}\"")

        self.results = []

        # ===== Stage 1: WavLM 特征提取 =====
        print("\n" + "="*70)
        print("Stage 1: WavLM 特征提取")
        print("="*70)
        with torch.no_grad():
            h = self.ssl_extractor(waveform).squeeze(0)

        result = self.test_stage(
            "stage1_wavlm",
            f"WavLM Layer {self.config.ssl_layer} 特征提取",
            h, output_path, reference_text
        )
        self.results.append(result)

        # ===== Stage 2: Eta-WavLM 投影 =====
        if self.config.use_eta_wavlm:
            print("\n" + "="*70)
            print("Stage 2: Eta-WavLM 投影")
            print("="*70)
            with torch.no_grad():
                h_clean = self.projector(h)

            result = self.test_stage(
                "stage2_eta_wavlm",
                "Eta-WavLM 说话人去除",
                h_clean, output_path, reference_text
            )
            self.results.append(result)
        else:
            h_clean = h
            print("\n⊘ Stage 2: Eta-WavLM 已禁用")

        # ===== Stage 3: SAMM 符号化 =====
        print("\n" + "="*70)
        print("Stage 3.1: SAMM Codebook 编码")
        print("="*70)
        with torch.no_grad():
            z = self.codebook.encode(h_clean)
        print(f"符号序列长度: {len(z)}, 唯一符号数: {len(torch.unique(z))}")

        # ===== Stage 3': Phone 预测 =====
        print("\n" + "="*70)
        print("Stage 3': Phone 预测")
        print("="*70)
        with torch.no_grad():
            phones = self.phone_predictor(h)

        phone_ids, true_durations = self.phone_predictor.get_phone_durations(phones)
        phone_segments = self.phone_predictor.get_phone_segments(phones)
        print(f"音素数量: {len(phone_ids)}, 平均时长: {true_durations.float().mean():.2f} 帧")

        # ===== Stage 3.3: Duration 匿名化 =====
        print("\n" + "="*70)
        print("Stage 3.3: Duration 匿名化")
        print("="*70)
        with torch.no_grad():
            anon_durations = self.duration_anonymizer.anonymize(
                phone_ids.to(self.device),
                true_durations.to(self.device),
            )
        print(f"原始总时长: {true_durations.sum():.0f} 帧")
        print(f"匿名化总时长: {anon_durations.sum():.0f} 帧")

        # ===== Stage 3.4: SAMM 掩码 =====
        print("\n" + "="*70)
        print("Stage 3.4: SAMM 掩码")
        print("="*70)
        z_masked, _, mask_indicator = self.masking(
            z, torch.ones_like(z, dtype=torch.float)
        )
        mask_ratio = mask_indicator.float().mean().item()
        print(f"掩码比例: {mask_ratio:.2%}")

        # ===== Stage 3.5: Pattern Matrix 平滑 =====
        print("\n" + "="*70)
        print("Stage 3.5: Pattern Matrix 平滑")
        print("="*70)
        z_smooth = self.pattern_matrix.smooth_sequence(z_masked, mask_indicator)
        print(f"平滑完成")

        # ===== Stage 4.1: kNN 检索 =====
        print("\n" + "="*70)
        print("Stage 4.1: kNN 检索")
        print("="*70)
        target_gender = 0  # Male
        with torch.no_grad():
            h_anon = self.retriever.retrieve_batch(
                h_clean, phones, z_smooth, target_gender
            )

        result = self.test_stage(
            "stage4_knn",
            f"kNN 检索 (Top-1={self.config.use_top1}, Cosine={self.config.use_cosine})",
            h_anon, output_path, reference_text
        )
        self.results.append(result)

        # ===== Stage 4.2: Duration 调整 =====
        print("\n" + "="*70)
        print("Stage 4.2: Duration 调整")
        print("="*70)
        from models.knn_vc.duration import DurationAdjuster
        h_anon_adjusted = DurationAdjuster.adjust_features(
            h_anon, phone_segments, anon_durations
        )
        print(f"调整前: {h_anon.shape}, 调整后: {h_anon_adjusted.shape}")

        result = self.test_stage(
            "stage5_final",
            "最终输出 (Duration 调整后)",
            h_anon_adjusted, output_path, reference_text
        )
        self.results.append(result)

        # 生成报告
        self._generate_report(output_path, reference_text)

        return self.results

    def _generate_report(self, output_path: Path, reference_text: str):
        """生成消融测试报告"""
        print("\n" + "="*70)
        print("消融测试报告")
        print("="*70)

        print(f"\n原始文本: \"{reference_text}\"")
        print(f"\n{'Stage':<20} | {'描述':<40} | {'WER':<10} | {'转录文本'}")
        print("-" * 120)

        for result in self.results:
            wer_str = f"{result.wer:.2%}" if result.wer is not None else "N/A"
            trans_str = result.transcription[:50] if result.transcription else "N/A"
            print(f"{result.stage_name:<20} | {result.stage_desc:<40} | {wer_str:<10} | {trans_str}")

        # 找出语义丢失最严重的阶段
        print("\n" + "="*70)
        print("诊断结论")
        print("="*70)

        valid_results = [r for r in self.results if r.wer is not None]
        if valid_results:
            # 计算每个阶段的 WER 增量
            print(f"\nWER 增量分析:")
            prev_wer = 0.0
            max_delta = 0.0
            worst_stage = None

            for result in valid_results:
                delta = result.wer - prev_wer
                print(f"  {result.stage_name}: WER={result.wer:.2%}, Δ={delta:+.2%}")

                if delta > max_delta:
                    max_delta = delta
                    worst_stage = result

                prev_wer = result.wer

            if worst_stage:
                print(f"\n🔴 语义丢失最严重的阶段:")
                print(f"  {worst_stage.stage_name}: {worst_stage.stage_desc}")
                print(f"  WER 增量: +{max_delta:.2%}")

        # 保存 JSON 报告
        report = {
            'reference_text': reference_text,
            'config': {
                'use_eta_wavlm': self.config.use_eta_wavlm,
                'use_top1': self.config.use_top1,
                'use_cosine': self.config.use_cosine,
                'ssl_layer': self.config.ssl_layer,
                'mask_prob': self.config.mask_prob,
                'span_mask_prob': self.config.span_mask_prob,
            },
            'stages': [
                {
                    'stage_name': r.stage_name,
                    'stage_desc': r.stage_desc,
                    'features_shape': list(r.features_shape),
                    'wer': r.wer,
                    'transcription': r.transcription,
                    'audio_path': r.audio_path,
                    'success': r.success,
                    'error': r.error
                }
                for r in self.results
            ]
        }

        report_path = output_path / "ablation_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\n💾 详细报告已保存: {report_path}")


def main():
    """主函数"""
    import argparse
    import random

    parser = argparse.ArgumentParser(description="逐层消融测试")
    parser.add_argument("--audio", "-a", help="测试音频路径")
    parser.add_argument("--config", "-c", default="configs/test_no_eta.yaml",
                       help="配置文件")
    parser.add_argument("--output", "-o", help="输出目录")

    args = parser.parse_args()

    # 选择测试音频
    if args.audio:
        audio_path = args.audio
    else:
        audio_files = list(Path('/root/autodl-tmp/datasets/LibriSpeech/test-clean').glob('**/*.flac'))
        audio_path = str(random.choice(audio_files))

    print(f"测试音频: {audio_path}")
    print(f"配置文件: {args.config}")

    # 运行消融测试
    tester = AblationTester(args.config)
    results = tester.run_ablation(audio_path, args.output)

    print("\n" + "="*70)
    print("消融测试完成!")
    print("="*70)


if __name__ == "__main__":
    main()
