"""train_emotion_head.py

训练句级情感门控头。

  - speaker-disjoint 划分：验证集说话人与训练集不重叠（逼近 IEMOCAP 部署=未见说话人）
  - 输入标准化统计量只在训练集上算，写入 head 的 buffer
  - 类别加权 CE（CREMA-D 数量远多于 ESD，防止偏置）
  - 按验证集「宏召回」(= UAR 等价物) 早停
  - 产出 P(sad) 阈值扫描表，帮你在 IEMOCAP-dev 上选 sad_gate

用法:
    python train_emotion_head.py \
        --feats /root/autodl-tmp/anon_test/checkpoints/emotion_feats.pt \
        --out   /root/autodl-tmp/anon_test/checkpoints/emotion_head_cremad.pt

输出 ckpt 与 models/emotion/head.py 的 EmotionHead.load 直接兼容。
"""

import sys
import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, '/root/autodl-tmp/anon_test')
from models.emotion.head import EmotionHead, EMOTION_LABELS  # noqa: E402

CKPT_DIR = Path('/root/autodl-tmp/anon_test/checkpoints')
SAD_IDX = 0


def speaker_disjoint_split(speakers, val_frac=0.15, seed=0):
    uniq = sorted(set(speakers))
    rng = random.Random(seed)
    rng.shuffle(uniq)
    n_val = max(1, int(len(uniq) * val_frac))
    val_spk = set(uniq[:n_val])
    tr_idx = [i for i, s in enumerate(speakers) if s not in val_spk]
    va_idx = [i for i, s in enumerate(speakers) if s in val_spk]
    return tr_idx, va_idx, val_spk


def macro_recall(logits, y, n_classes=4):
    pred = logits.argmax(-1)
    recs = []
    for c in range(n_classes):
        m = (y == c)
        if m.sum() == 0:
            continue
        recs.append((pred[m] == c).float().mean().item())
    return sum(recs) / len(recs), recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feats', default=str(CKPT_DIR / 'emotion_feats.pt'))
    ap.add_argument('--out', default=str(CKPT_DIR / 'emotion_head_cremad.pt'))
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--hidden', type=int, default=256)
    ap.add_argument('--dropout', type=float, default=0.3)
    ap.add_argument('--val_frac', type=float, default=0.15)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)

    blob = torch.load(args.feats, map_location='cpu')
    X = blob['feats'].float()                 # [N,1024]
    y = blob['labels'].long()                 # [N]
    speakers = blob['speakers']

    tr, va, val_spk = speaker_disjoint_split(speakers, args.val_frac, args.seed)
    Xtr, ytr = X[tr].to(device), y[tr].to(device)
    Xva, yva = X[va].to(device), y[va].to(device)
    print(f'train={len(tr)}  val={len(va)}  val_speakers={len(val_spk)}')
    print('train 类别计数:', torch.bincount(ytr.cpu(), minlength=4).tolist(),
          '(顺序 sad,neu,ang,hap)')

    # 输入标准化统计量：只用训练集
    feat_mean = Xtr.mean(0)
    feat_std = Xtr.std(0)

    # 类别加权（逆频率）
    counts = torch.bincount(ytr.cpu(), minlength=4).float()
    weights = (counts.sum() / (4 * counts.clamp_min(1))).to(device)

    model = EmotionHead(in_dim=X.shape[1], hidden=args.hidden,
                        n_classes=4, dropout=args.dropout).to(device)
    model.set_norm_stats(feat_mean, feat_std)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    crit = nn.CrossEntropyLoss(weight=weights)

    best_mr, best_state = -1.0, None
    for ep in range(args.epochs):
        model.train()
        # 简单全批/小批训练（特征量不大，可整批；如需可改 DataLoader）
        perm = torch.randperm(len(Xtr), device=device)
        bs = 512
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = crit(model(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            logits = model(Xva)
            mr, recs = macro_recall(logits, yva)
        if mr > best_mr:
            best_mr = mr
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        if ep % 5 == 0 or ep == args.epochs - 1:
            rec_str = ' '.join(f'{n}={r:.2f}' for n, r in zip(EMOTION_LABELS, recs))
            print(f'ep{ep:02d} loss={loss.item():.3f} val_UAR={mr:.4f}  [{rec_str}]')

    model.load_state_dict(best_state)
    model.to(device).eval()
    print(f'\n最佳 val_UAR(宏召回) = {best_mr:.4f}')

    # ── 验证集混淆矩阵 ──
    with torch.no_grad():
        pred = model(Xva).argmax(-1).cpu()
    cm = torch.zeros(4, 4, dtype=torch.long)
    for t, p in zip(yva.cpu(), pred):
        cm[t, p] += 1
    print('混淆矩阵 (行=真, 列=预测; 顺序 sad,neu,ang,hap):')
    print(cm.tolist())

    # ── P(sad) 阈值扫描：帮你定 sad_gate ──
    # 真 SAD 句被门控触发的比例(recall) vs 非 SAD 句被误触发的比例(误报)
    with torch.no_grad():
        psad = torch.softmax(model(Xva), -1)[:, SAD_IDX].cpu()
    is_sad = (yva.cpu() == SAD_IDX)
    print('\nP(sad) 阈值扫描 (在 val 上;  '
          'sad命中=真sad句触发率, 非sad误触发=会被污染的句子比例):')
    print(f'{"thr":>5} {"sad命中":>8} {"非sad误触发":>12}')
    for thr in [0.3, 0.4, 0.5, 0.6, 0.7]:
        fire = psad > thr
        sad_hit = (fire & is_sad).sum().item() / max(1, is_sad.sum().item())
        non_fp = (fire & ~is_sad).sum().item() / max(1, (~is_sad).sum().item())
        print(f'{thr:>5.1f} {sad_hit:>8.3f} {non_fp:>12.3f}')

    model.save(args.out)
    print(f'\n已保存 → {args.out}')
    print('提示: sad_gate 不要在 IEMOCAP-test 上选；'
          '在 IEMOCAP-dev 上扫 {0.4,0.5,0.6}，报告 test。')


if __name__ == '__main__':
    main()