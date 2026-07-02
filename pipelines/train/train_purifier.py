#!/usr/bin/env python3
"""
训练 FeaturePurifier:
  - 输入: 预提取的 L24 特征 (HDF5)
  - 目标: 保留 phone 内容, 对抗去除 speaker 身份
  - 输出: encoder 权重 (用于推理时净化检索向量)

训练成功判据:
  ✅ phone_acc 逼近原始 Phone Predictor 水平 (70%-85%)
  ✅ spk_acc 崩塌至随机水平 (1/num_speakers ≈ 0.5%)
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent.parent))

from purifier import FeaturePurifier
from purifier_dataset import build_dataloader


# ==================== 默认配置 ====================
DEFAULTS = dict(
    feature_dir="/root/autodl-tmp/anon_test/checkpoints/trainother500_200spk",
    output_dir="/root/autodl-tmp/anon_test/checkpoints/purifier",
    # 模型
    input_dim=1024,
    hidden_dim=512,
    num_phones=72,
    dropout=0.1,
    # 训练
    epochs=30,
    batch_size=16384,
    lr=3e-4,
    weight_decay=1e-5,
    # GRL alpha 调度: 从 alpha_start 线性增长到 alpha_end
    alpha_start=0.0,
    alpha_end=1.0,
    alpha_warmup_epochs=5,
    # 损失权重
    lambda_phone=1.0,
    lambda_spk=1.0,
    # 系统
    num_workers=4,
    load_into_memory=True,
    log_interval=50,
    save_interval=5,
    device="cuda",
)


def get_alpha(epoch: int, warmup: int, alpha_start: float, alpha_end: float) -> float:
    """GRL alpha 线性预热调度"""
    if warmup <= 0:
        return alpha_end
    progress = min(epoch / warmup, 1.0)
    return alpha_start + (alpha_end - alpha_start) * progress


def train_one_epoch(model, loader, optimizer, alpha, cfg, epoch):
    model.train()
    total_loss, total_phone_loss, total_spk_loss = 0.0, 0.0, 0.0
    phone_correct, spk_correct, total_count = 0, 0, 0

    for step, (feat, phone_label, spk_label) in enumerate(loader):
        feat = feat.to(cfg["device"], non_blocking=True)
        phone_label = phone_label.to(cfg["device"], non_blocking=True)
        spk_label = spk_label.to(cfg["device"], non_blocking=True)

        z_clean, phone_logits, spk_logits = model(feat, alpha=alpha)

        loss_phone = F.cross_entropy(phone_logits, phone_label)
        loss_spk = F.cross_entropy(spk_logits, spk_label)
        loss = cfg["lambda_phone"] * loss_phone + cfg["lambda_spk"] * loss_spk

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        bs = feat.size(0)
        total_loss += loss.item() * bs
        total_phone_loss += loss_phone.item() * bs
        total_spk_loss += loss_spk.item() * bs
        phone_correct += (phone_logits.argmax(1) == phone_label).sum().item()
        spk_correct += (spk_logits.argmax(1) == spk_label).sum().item()
        total_count += bs

        if (step + 1) % cfg["log_interval"] == 0:
            print(
                f"  [E{epoch} step {step+1}] "
                f"loss={loss.item():.4f}  "
                f"phone_loss={loss_phone.item():.4f}  "
                f"spk_loss={loss_spk.item():.4f}  "
                f"alpha={alpha:.3f}"
            )

    metrics = {
        "loss": total_loss / total_count,
        "phone_loss": total_phone_loss / total_count,
        "spk_loss": total_spk_loss / total_count,
        "phone_acc": phone_correct / total_count * 100,
        "spk_acc": spk_correct / total_count * 100,
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description="训练 FeaturePurifier (GRL)")
    for key, val in DEFAULTS.items():
        arg_type = type(val) if val is not None else str
        if isinstance(val, bool):
            parser.add_argument(f"--{key}", action="store_true", default=val)
        else:
            parser.add_argument(f"--{key}", type=arg_type, default=val)
    args = parser.parse_args()
    cfg = vars(args)

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 数据 ----------
    print("\n[1/3] 加载数据...")
    loader, dataset = build_dataloader(
        cfg["feature_dir"],
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        load_into_memory=cfg["load_into_memory"],
    )
    num_speakers = dataset.num_speakers
    print(f"  实际说话人数: {num_speakers}")

    # ---------- 模型 ----------
    print("\n[2/3] 初始化模型...")
    model = FeaturePurifier(
        input_dim=cfg["input_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_phones=cfg["num_phones"],
        num_speakers=num_speakers,
        dropout=cfg["dropout"],
    ).to(cfg["device"])

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  参数量: {param_count:.2f}M")

    optimizer = optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"], eta_min=1e-6
    )

    # 保存配置
    with open(output_dir / "train_config.json", "w") as f:
        json.dump(cfg, f, indent=2, default=str)

    # ---------- 训练 ----------
    print("\n[3/3] 开始训练...")
    print(f"  epochs={cfg['epochs']}, batch_size={cfg['batch_size']}")
    print(f"  alpha: {cfg['alpha_start']} -> {cfg['alpha_end']} (warmup {cfg['alpha_warmup_epochs']} epochs)")
    random_spk_acc = 100.0 / num_speakers
    print(f"  随机说话人准确率基线: {random_spk_acc:.2f}%")
    print()

    best_metric = float("inf")  # 追踪: phone_loss 低 + spk_acc 低
    history = []

    for epoch in range(1, cfg["epochs"] + 1):
        alpha = get_alpha(
            epoch - 1, cfg["alpha_warmup_epochs"], cfg["alpha_start"], cfg["alpha_end"]
        )

        metrics = train_one_epoch(model, loader, optimizer, alpha, cfg, epoch)
        scheduler.step()

        print(
            f"Epoch {epoch}/{cfg['epochs']}  "
            f"loss={metrics['loss']:.4f}  "
            f"phone_loss={metrics['phone_loss']:.4f} ({metrics['phone_acc']:.1f}%)  "
            f"spk_loss={metrics['spk_loss']:.4f} ({metrics['spk_acc']:.1f}%)  "
            f"alpha={alpha:.3f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        metrics["epoch"] = epoch
        metrics["alpha"] = alpha
        history.append(metrics)

        # 保存检查点
        is_good = metrics["phone_acc"] > 50 and metrics["spk_acc"] < random_spk_acc * 5
        composite = metrics["phone_loss"] + metrics["spk_acc"] / 100.0

        if composite < best_metric and epoch >= cfg["alpha_warmup_epochs"]:
            best_metric = composite
            torch.save(
                {
                    "encoder_state_dict": model.encoder.state_dict(),
                    "full_state_dict": model.state_dict(),
                    "config": cfg,
                    "num_speakers": num_speakers,
                    "epoch": epoch,
                    "metrics": metrics,
                },
                output_dir / "best_purifier.pt",
            )
            print(f"  -> 保存最佳模型 (composite={composite:.4f})")

        if epoch % cfg["save_interval"] == 0:
            torch.save(model.state_dict(), output_dir / f"epoch_{epoch}.pt")

    # 保存最终模型 + 仅 encoder
    torch.save(model.state_dict(), output_dir / "final_purifier.pt")
    torch.save(model.encoder.state_dict(), output_dir / "encoder_only.pt")

    with open(output_dir / "train_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # ---------- 总结 ----------
    print("\n" + "=" * 60)
    print("训练完成!")
    print(f"  输出目录: {output_dir}")
    print(f"  最终 phone_acc: {history[-1]['phone_acc']:.1f}%")
    print(f"  最终 spk_acc:   {history[-1]['spk_acc']:.1f}% (随机={random_spk_acc:.2f}%)")

    if history[-1]["spk_acc"] < random_spk_acc * 3:
        print("  ✅ 说话人信息已有效去除")
    else:
        print("  ⚠️  说话人准确率偏高，建议增大 alpha 或训练更多 epoch")
    print("=" * 60)


if __name__ == "__main__":
    main()