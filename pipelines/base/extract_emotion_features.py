"""extract_emotion_features.py

为情感门控头提取训练特征：CREMA-D + ESD(英文) → WavLM-Large L24 均值池化 [1024]。

与 pipeline._sad_prob 完全一致的表示：
    multi = wavlm.forward_multi_layer(wav[1,T], layers=[24])
    feat  = multi[24].squeeze(0).mean(0)   # [1024]

输出 emotion_feats.pt:
    {
      'feats'    : FloatTensor [N,1024],
      'labels'   : LongTensor  [N]      # 0=sad 1=neu 2=ang 3=hap
      'speakers' : list[str]   [N]      # 用于 speaker-disjoint 划分
      'datasets' : list[str]   [N]      # 'cremad' / 'esd'
    }

用法:
    python extract_emotion_features.py \
        --cremad_dir /path/to/CREMA-D/AudioWAV \
        --esd_dir    /path/to/ESD \
        --esd_speakers 0011-0020 \
        --out  /root/autodl-tmp/anon_test/checkpoints/emotion_feats.pt

禁用 IEMOCAP（= SER 评估集）。请勿把任何 IEMOCAP 路径传进来。
"""

import sys
import argparse
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm

sys.path.insert(0, '/root/autodl-tmp/anon_test')
from models.ssl.wrappers import WavLMSSLExtractor  # noqa: E402

CKPT_DIR = Path('/root/autodl-tmp/anon_test/checkpoints')

# 标签顺序铁律: SAD=0 NEU=1 ANG=2 HAP=3
LABEL2IDX = {'sad': 0, 'neu': 1, 'ang': 2, 'hap': 3}

# CREMA-D 文件名第 3 段 → 我们的 4 类（DIS/FEA 丢弃）
CREMAD_MAP = {'SAD': 'sad', 'NEU': 'neu', 'ANG': 'ang', 'HAP': 'hap'}

# ESD 情感目录名 → 我们的 4 类（Surprise 丢弃；大小写兼容）
ESD_MAP = {'sad': 'sad', 'neutral': 'neu', 'angry': 'ang', 'happy': 'hap'}


def parse_speaker_range(s):
    """'0011-0020' → {'0011',...,'0020'}；也支持逗号列表 '0011,0013'。"""
    out = set()
    for part in s.split(','):
        part = part.strip()
        if '-' in part:
            a, b = part.split('-')
            w = len(a)
            out |= {str(i).zfill(w) for i in range(int(a), int(b) + 1)}
        elif part:
            out.add(part)
    return out


@torch.no_grad()
def pooled_l24(wavlm, wav_path, device):
    wav, sr = torchaudio.load(str(wav_path))
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    wav = wav.mean(0, keepdim=True).to(device)       # [1, samples] 单声道
    multi = wavlm.forward_multi_layer(wav, layers=[24])
    l24 = multi[24].squeeze(0)                        # [T, 1024]
    return l24.mean(0).cpu()                          # [1024]


def collect_cremad(cremad_dir):
    """返回 [(wav_path, label_str, speaker_id), ...]。"""
    items = []
    for p in sorted(Path(cremad_dir).glob('*.wav')):
        # 文件名: {actor}_{sentence}_{EMO}_{level}.wav, e.g. 1001_DFA_ANG_XX.wav
        parts = p.stem.split('_')
        if len(parts) < 3:
            continue
        emo = CREMAD_MAP.get(parts[2].upper())
        if emo is None:                              # DIS / FEA → 丢弃
            continue
        items.append((p, emo, f'cremad_{parts[0]}'))
    return items


def collect_esd(esd_dir, speakers):
    """ESD 目录结构: ESD/<spk>/<Emotion>/[train|evaluation|test]/*.wav
    （不同发布版本层级略有差异，这里递归扫 wav 并从路径推断 spk/emotion）。"""
    items = []
    esd_dir = Path(esd_dir)
    for p in sorted(esd_dir.rglob('*.wav')):
        parts = [x.lower() for x in p.parts]
        # 找情感目录
        emo = None
        for seg in parts:
            if seg in ESD_MAP:
                emo = ESD_MAP[seg]
                break
        if emo is None:                              # Surprise / 未知 → 丢弃
            continue
        # 找说话人 id（ESD 说话人是 4 位数字目录）
        spk = None
        for raw in p.parts:
            if raw.isdigit() and len(raw) == 4:
                spk = raw
                break
        if spk is None or (speakers and spk not in speakers):
            continue
        items.append((p, emo, f'esd_{spk}'))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cremad_dir',
                    default='/root/autodl-tmp/datasets/AudioWAV',
                    help='CREMA-D AudioWAV 目录')
    ap.add_argument('--esd_dir',
                    default='/root/autodl-tmp/datasets/Emotion Speech Dataset',
                    help='ESD 根目录（路径含空格，已作为默认值写死，避免命令行转义）')
    ap.add_argument('--esd_speakers', default='0011-0020',
                    help='ESD 英文说话人，默认 0011-0020；空字符串=全部')
    ap.add_argument('--wavlm_ckpt', default=str(CKPT_DIR / 'WavLM-Large.pt'))
    ap.add_argument('--out', default=str(CKPT_DIR / 'emotion_feats.pt'))
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    assert args.cremad_dir or args.esd_dir, '至少提供一个数据集目录'

    device = torch.device(args.device)
    wavlm = WavLMSSLExtractor(ckpt_path=args.wavlm_ckpt, layer=24, device=device)

    items = []
    if args.cremad_dir:
        c = collect_cremad(args.cremad_dir)
        print(f'[CREMA-D] {len(c)} clips (已丢弃 DIS/FEA)')
        items += c
    if args.esd_dir:
        spk = parse_speaker_range(args.esd_speakers) if args.esd_speakers else set()
        e = collect_esd(args.esd_dir, spk)
        print(f'[ESD] {len(e)} clips (说话人={sorted(spk) or "全部"}, 已丢弃 Surprise)')
        items += e

    # 类别分布
    from collections import Counter
    print('类别分布:', Counter(lbl for _, lbl, _ in items))

    feats, labels, speakers, datasets = [], [], [], []
    for wav_path, lbl, spk in tqdm(items, desc='extract L24'):
        try:
            f = pooled_l24(wavlm, wav_path, device)
        except Exception as ex:
            print(f'[skip] {wav_path}: {ex}')
            continue
        feats.append(f)
        labels.append(LABEL2IDX[lbl])
        speakers.append(spk)
        datasets.append('cremad' if spk.startswith('cremad') else 'esd')

    torch.save({
        'feats': torch.stack(feats),
        'labels': torch.tensor(labels, dtype=torch.long),
        'speakers': speakers,
        'datasets': datasets,
    }, args.out)
    print(f'已保存 {len(feats)} 条 → {args.out}')


if __name__ == '__main__':
    main()