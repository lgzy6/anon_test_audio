#!/usr/bin/env python3
"""
语义保留评估脚本
使用 Whisper ASR 评估匿名化音频的语义保留程度
"""

import sys
import torch
import torchaudio
from pathlib import Path
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent))


def load_whisper_model(model_size="base"):
    """加载 Whisper ASR 模型"""
    try:
        import whisper
        print(f"\n加载 Whisper {model_size} 模型...")
        model = whisper.load_model(model_size)
        print(f"  ✓ Whisper 加载完成")
        return model
    except ImportError:
        print("错误: 未安装 openai-whisper")
        print("请运行: pip install openai-whisper")
        sys.exit(1)


def transcribe_audio(model, audio_path, language="en"):
    """转录音频"""
    import whisper
    import numpy as np

    print(f"\n转录音频: {Path(audio_path).name}")

    # 使用 torchaudio 加载音频，避免依赖 ffmpeg
    waveform, sr = torchaudio.load(audio_path)

    # 转为单声道
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0)
    else:
        waveform = waveform.squeeze(0)

    # 重采样到 16kHz (Whisper 要求)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)

    # 转为 numpy 并归一化到 [-1, 1]
    audio = waveform.numpy()
    audio = audio.astype(np.float32)

    # Whisper 期望的音频格式
    result = model.transcribe(audio, language=language, verbose=False)

    text = result["text"].strip()
    segments = result.get("segments", [])

    print(f"  转录文本: {text}")

    return {
        "text": text,
        "segments": segments,
        "language": result.get("language", language)
    }


def calculate_wer(reference, hypothesis):
    """计算词错误率 (Word Error Rate)"""
    # 简单的 WER 计算
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    # 使用编辑距离
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
                substitution = d[i-1][j-1] + 1
                insertion = d[i][j-1] + 1
                deletion = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    edit_distance = d[len(ref_words)][len(hyp_words)]
    wer = edit_distance / len(ref_words) if len(ref_words) > 0 else 0

    return wer, edit_distance


def calculate_similarity(text1, text2):
    """计算文本相似度"""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    if len(words1) == 0 and len(words2) == 0:
        return 1.0

    intersection = words1 & words2
    union = words1 | words2

    jaccard = len(intersection) / len(union) if len(union) > 0 else 0

    return jaccard


def analyze_audio_features(audio_path):
    """分析音频特征"""
    waveform, sr = torchaudio.load(audio_path)

    duration = waveform.shape[1] / sr
    rms = torch.sqrt(torch.mean(waveform ** 2)).item()

    return {
        "duration": duration,
        "sample_rate": sr,
        "channels": waveform.shape[0],
        "rms_energy": rms
    }


def evaluate_semantic_preservation(original_path, anonymized_path, output_dir=None, whisper_model="base"):
    """评估语义保留"""

    print("\n" + "="*70)
    print("语义保留评估")
    print("="*70)

    # 检查文件
    original_path = Path(original_path)
    anonymized_path = Path(anonymized_path)

    if not original_path.exists():
        print(f"错误: 原始音频不存在: {original_path}")
        return None

    if not anonymized_path.exists():
        print(f"错误: 匿名化音频不存在: {anonymized_path}")
        return None

    # 创建输出目录
    if output_dir is None:
        output_dir = anonymized_path.parent / "semantic_evaluation"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n输入:")
    print(f"  原始音频: {original_path}")
    print(f"  匿名音频: {anonymized_path}")
    print(f"  输出目录: {output_dir}")

    # 加载 ASR 模型
    model = load_whisper_model(whisper_model)

    # 转录原始音频
    print("\n" + "-"*70)
    print("[1/3] 转录原始音频")
    print("-"*70)
    original_result = transcribe_audio(model, original_path)

    # 转录匿名化音频
    print("\n" + "-"*70)
    print("[2/3] 转录匿名化音频")
    print("-"*70)
    anonymized_result = transcribe_audio(model, anonymized_path)

    # 计算指标
    print("\n" + "-"*70)
    print("[3/3] 计算评估指标")
    print("-"*70)

    original_text = original_result["text"]
    anonymized_text = anonymized_result["text"]

    # WER
    wer, edit_dist = calculate_wer(original_text, anonymized_text)

    # 文本相似度
    similarity = calculate_similarity(original_text, anonymized_text)

    # 音频特征
    original_features = analyze_audio_features(original_path)
    anonymized_features = analyze_audio_features(anonymized_path)

    # 汇总结果
    results = {
        "timestamp": datetime.now().isoformat(),
        "files": {
            "original": str(original_path),
            "anonymized": str(anonymized_path)
        },
        "transcription": {
            "original": original_text,
            "anonymized": anonymized_text
        },
        "metrics": {
            "wer": wer,
            "edit_distance": edit_dist,
            "jaccard_similarity": similarity,
            "text_match": original_text.lower() == anonymized_text.lower()
        },
        "audio_features": {
            "original": original_features,
            "anonymized": anonymized_features
        }
    }

    # 打印结果
    print("\n" + "="*70)
    print("评估结果")
    print("="*70)

    print(f"\n📝 转录对比:")
    print(f"  原始文本: \"{original_text}\"")
    print(f"  匿名文本: \"{anonymized_text}\"")

    print(f"\n📊 语义保留指标:")
    print(f"  WER (词错误率):     {wer:.2%}")
    print(f"  编辑距离:           {edit_dist}")
    print(f"  Jaccard 相似度:     {similarity:.2%}")
    print(f"  完全匹配:           {'✓ 是' if results['metrics']['text_match'] else '✗ 否'}")

    print(f"\n🎵 音频特征对比:")
    print(f"  时长变化: {original_features['duration']:.2f}s → {anonymized_features['duration']:.2f}s")
    print(f"  能量变化: {original_features['rms_energy']:.4f} → {anonymized_features['rms_energy']:.4f}")

    # 判断语义保留程度
    print(f"\n🎯 语义保留评级:")
    if wer < 0.1:
        grade = "优秀 (Excellent)"
        emoji = "🟢"
    elif wer < 0.3:
        grade = "良好 (Good)"
        emoji = "🟡"
    elif wer < 0.5:
        grade = "一般 (Fair)"
        emoji = "🟠"
    else:
        grade = "较差 (Poor)"
        emoji = "🔴"

    print(f"  {emoji} {grade} (WER: {wer:.2%})")

    if wer > 0.3:
        print(f"\n⚠️  警告: 语义保留度较低，可能存在以下问题:")
        print(f"    - kNN 检索策略不当 (加权平均导致模糊)")
        print(f"    - 特征空间不匹配")
        print(f"    - 时长调整过度")

    # 保存结果
    result_file = output_dir / "semantic_evaluation.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n💾 详细结果已保存到: {result_file}")

    return results


def batch_evaluate(test_dir, whisper_model="base"):
    """批量评估目录下的所有测试结果"""
    test_dir = Path(test_dir)

    print(f"\n批量评估: {test_dir}")

    # 查找所有包含 original.wav 和 anonymized.wav 的目录
    test_cases = []
    for original_file in test_dir.glob("**/original.wav"):
        anonymized_file = original_file.parent / "anonymized.wav"
        if anonymized_file.exists():
            test_cases.append((original_file, anonymized_file))

    if not test_cases:
        print("未找到测试用例 (需要 original.wav 和 anonymized.wav)")
        return

    print(f"找到 {len(test_cases)} 个测试用例")

    all_results = []
    for i, (original, anonymized) in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"测试用例 {i}/{len(test_cases)}")
        print(f"{'='*70}")

        result = evaluate_semantic_preservation(
            original, anonymized,
            output_dir=original.parent / "semantic_evaluation",
            whisper_model=whisper_model
        )

        if result:
            all_results.append(result)

    # 汇总统计
    if all_results:
        avg_wer = sum(r["metrics"]["wer"] for r in all_results) / len(all_results)
        avg_similarity = sum(r["metrics"]["jaccard_similarity"] for r in all_results) / len(all_results)

        print("\n" + "="*70)
        print("批量评估汇总")
        print("="*70)
        print(f"测试用例数: {len(all_results)}")
        print(f"平均 WER: {avg_wer:.2%}")
        print(f"平均相似度: {avg_similarity:.2%}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="评估匿名化音频的语义保留程度")
    parser.add_argument("--original", "-o", help="原始音频路径")
    parser.add_argument("--anonymized", "-a", help="匿名化音频路径")
    parser.add_argument("--output", help="输出目录")
    parser.add_argument("--batch", "-b", help="批量评估目录")
    parser.add_argument("--model", "-m", default="base",
                       choices=["tiny", "base", "small", "medium", "large"],
                       help="Whisper 模型大小 (默认: base)")

    args = parser.parse_args()

    if args.batch:
        batch_evaluate(args.batch, args.model)
    elif args.original and args.anonymized:
        evaluate_semantic_preservation(
            args.original, args.anonymized,
            args.output, args.model
        )
    else:
        # 自动查找最近的测试结果
        outputs_dir = Path("outputs")
        if outputs_dir.exists():
            # 找最新的测试目录
            test_dirs = sorted(outputs_dir.glob("test_v2.1_*"), key=lambda p: p.stat().st_mtime, reverse=True)

            if test_dirs:
                latest_dir = test_dirs[0]
                original = latest_dir / "original.wav"
                anonymized = latest_dir / "anonymized.wav"

                if original.exists() and anonymized.exists():
                    print(f"自动检测到最新测试结果: {latest_dir.name}")
                    evaluate_semantic_preservation(
                        original, anonymized,
                        output_dir=latest_dir / "semantic_evaluation",
                        whisper_model=args.model
                    )
                else:
                    print("使用方法:")
                    print("  python evaluate_semantic_preservation.py --original <原始音频> --anonymized <匿名音频>")
                    print("  python evaluate_semantic_preservation.py --batch <测试目录>")
            else:
                print("未找到测试结果")
                print("\n使用方法:")
                print("  python evaluate_semantic_preservation.py --original <原始音频> --anonymized <匿名音频>")
        else:
            print("使用方法:")
            print("  python evaluate_semantic_preservation.py --original <原始音频> --anonymized <匿名音频>")
