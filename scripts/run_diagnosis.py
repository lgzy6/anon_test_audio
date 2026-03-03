#!/usr/bin/env python3
"""
运行语义诊断 - 验证匿名化流程各阶段的语义保留情况
"""

import sys
import torch
import torchaudio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.diagnose_semantic_loss import SemanticDiagnostics


def run_diagnosis(audio_path: str, config_path: str = None):
    """运行完整诊断"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    diag = SemanticDiagnostics(device)

    print("=" * 60)
    print("语义损失诊断")
    print("=" * 60)

    # 1. 加载音频
    print(f"\n加载音频: {audio_path}")
    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
    waveform = waveform.mean(0) if waveform.dim() > 1 else waveform.squeeze()
    waveform = waveform.to(device)

    # 2. 加载模型组件
    print("\n加载模型组件...")
    from models.ssl.wrappers import WavLMSSLExtractor
    from models.phone_predictor.predictor import PhonePredictor
    from models.knn_vc.retriever import ConstrainedKNNRetriever

    ssl = WavLMSSLExtractor(
        ckpt_path='checkpoints/WavLM-Large.pt',
        layer=15, device=device
    )

    phone_pred = PhonePredictor.load(
        'checkpoints/phone_decoder.pt', device=device
    )

    retriever = ConstrainedKNNRetriever(
        target_pool_path='data/samm_anon/checkpoints/target_pool',
        k=4, device=device
    )

    # 3. Stage 1: SSL 特征
    print("\n[Stage 1] SSL 特征提取...")
    with torch.no_grad():
        h_ssl = ssl(waveform.unsqueeze(0)).squeeze(0)
    print(f"  h_ssl shape: {h_ssl.shape}")

    # 诊断 SSL 特征连续性
    diag.diagnose_frame_continuity(h_ssl, "h_ssl")

    # 4. Phone 预测
    print("\n[Stage 3'] Phone 预测...")
    with torch.no_grad():
        phones = phone_pred(h_ssl.unsqueeze(0)).squeeze(0)
    print(f"  phones shape: {phones.shape}")
    print(f"  unique phones: {phones.unique().numel()}")

    # 5. kNN 检索
    print("\n[Stage 4] kNN 检索...")
    with torch.no_grad():
        h_anon = retriever.retrieve_batch(
            h_ssl, phones, symbols=None, target_gender=1
        )
    print(f"  h_anon shape: {h_anon.shape}")

    # 诊断检索后特征连续性
    diag.diagnose_frame_continuity(h_anon, "h_anon")

    # 诊断特征分布偏移
    diag.diagnose_feature_distribution(h_ssl, h_anon, "h_ssl", "h_anon")

    # 6. 检索 phone 对齐 (如果 pool 有 phones)
    if retriever.phones is not None:
        print("\n[诊断] Phone 对齐检查...")
        # 简化：统计检索结果的 phone 分布
        pool_phones = retriever.phones.cpu()
        print(f"  Pool phones 唯一值: {pool_phones.unique().numel()}")

    # 7. 总结
    issues = diag.summary()

    return diag.results, issues


if __name__ == "__main__":
    import glob

    # 找测试音频
    test_files = glob.glob('data/**/*.wav', recursive=True)[:1]
    if test_files:
        run_diagnosis(test_files[0])
    else:
        print("未找到测试音频文件")
