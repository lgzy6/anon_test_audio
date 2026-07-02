#!/usr/bin/env python3
"""
将情感数据集 (ESD英文 + RAVDESS) 的特征追加到现有 pool 的 h5 文件中。
情感说话人按性别分散追加到 pool_0~pool_3，每个 pool 各追加一部分。
"""
import os
os.environ["OMP_NUM_THREADS"] = "16"

import sys
import json
import torch
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torchaudio
from models.ssl.wrappers import WavLMSSLExtractor
from models.phone_predictor.predictor import PhonePredictor
from data.datasets.esd import ESDDataset
from data.datasets.ravdess import RAVDESSDataset


def scan_audiowav(root: str):
    """AudioWAV: {spk_id}_{...}.wav, spk_id奇数=男偶数=女，只存路径不预加载"""
    utts = []
    for wav in sorted(Path(root).glob("*.wav")):
        spk_id = wav.stem.split("_")[0]
        try:
            spk_num = int(spk_id)
        except ValueError:
            continue
        utts.append({
            "utt_id": f"audiowav_{wav.stem}",
            "speaker_id": f"audiowav_{spk_id}",
            "gender": "m" if spk_num % 2 == 1 else "f",
            "emotion": "unknown",
            "dataset": "audiowav",
            "audio_path": str(wav),
        })
    return utts


def _load_waveform(utt: dict) -> torch.Tensor:
    """从 utt 中获取 waveform，支持预加载 tensor 或路径延迟加载"""
    if "waveform" in utt:
        w = utt["waveform"]
        return w if isinstance(w, torch.Tensor) else torch.from_numpy(w)
    waveform, sr = torchaudio.load(utt["audio_path"])
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(0, keepdim=True)
    return waveform.squeeze(0)

WAVLM_CKPT   = "/root/autodl-tmp/anon_test/checkpoints/WavLM-Large.pt"
PHONE_CKPT   = "/root/autodl-tmp/anon_test/checkpoints/phone_decoder.pt"
CKPT_DIR     = Path("/root/autodl-tmp/anon_test/checkpoints")
ESD_ROOT     = "/root/autodl-tmp/datasets/Emotion Speech Dataset"
RAVDESS_ROOT = "/root/autodl-tmp/datasets/actordata/audio_speech_actors_01-24"
AUDIOWAV_DIR = "/root/autodl-tmp/datasets/AudioWAV"
DEVICE       = "cuda"
LAYERS       = [6, 24]
NUM_POOLS    = 4


def append_to_pool(pool_dir: Path, utterances_by_gender: dict, extractor, phone_predictor):
    """将情感 utterances 追加到指定 pool 目录下的 m/ 和 f/ h5 文件"""
    for gender, utts in utterances_by_gender.items():
        if not utts:
            continue
        out_dir = pool_dir / gender
        if not (out_dir / "metadata.json").exists():
            print(f"  跳过 {out_dir}（无 metadata.json）")
            continue

        with open(out_dir / "metadata.json") as f:
            meta = json.load(f)

        current_frame = meta["total_frames"]
        new_entries = []

        h5_files = {f"l{l}": h5py.File(out_dir / f"layer_{l}.h5", "a") for l in LAYERS}
        h5_phones = h5py.File(out_dir / "phones.h5", "a")

        for utt in tqdm(utts, desc=f"  {pool_dir.name}/{gender}", leave=False):
            try:
                w = _load_waveform(utt)
                if w.dim() > 1:
                    w = w.mean(0)
                w = w.unsqueeze(0).to(DEVICE)

                with torch.inference_mode():
                    multi_feats = extractor.forward_multi_layer(w, layers=LAYERS)
                    phones = phone_predictor(multi_feats[24][0].unsqueeze(0)).squeeze(0).cpu().numpy()

                num_frames = multi_feats[LAYERS[0]][0].shape[0]
                start, end = current_frame, current_frame + num_frames

                for l in LAYERS:
                    ds = h5_files[f"l{l}"]["features"]
                    ds.resize((end, 1024))
                    ds[start:end] = multi_feats[l][0].cpu().numpy()

                h5_phones["phones"].resize((end,))
                h5_phones["phones"][start:end] = phones

                new_entries.append({
                    "utt_id": utt["utt_id"],
                    "speaker_id": utt["speaker_id"],
                    "gender": gender,
                    "emotion": utt.get("emotion", "unknown"),
                    "dataset": utt.get("dataset", "emotion"),
                    "h5_start_idx": start,
                    "h5_end_idx": end,
                })
                current_frame = end

            except Exception as e:
                print(f"    跳过 {utt['utt_id']}: {e}")
                continue

            torch.cuda.empty_cache()

        for h5f in h5_files.values():
            h5f.close()
        h5_phones.close()

        meta["total_frames"] = current_frame
        meta["utterances"].extend(new_entries)
        with open(out_dir / "metadata.json", "w") as f:
            json.dump(meta, f)

        print(f"  {pool_dir.name}/{gender}: +{len(new_entries)} utts, total_frames={current_frame}")


def main():
    print("加载情感数据集...")
    esd = ESDDataset(ESD_ROOT, language="en")
    ravdess = RAVDESSDataset(RAVDESS_ROOT)

    all_by_gender = {"m": [], "f": []}
    for dataset, name in [(esd, "esd"), (ravdess, "ravdess")]:
        for idx in range(len(dataset)):
            item = dataset[idx]
            item["dataset"] = name
            g = item.get("gender", "unknown")
            if g in all_by_gender:
                all_by_gender[g].append(item)

    print("扫描 AudioWAV...")
    for utt in scan_audiowav(AUDIOWAV_DIR):
        g = utt["gender"]
        if g in all_by_gender:
            all_by_gender[g].append(utt)

    print(f"情感数据: male={len(all_by_gender['m'])}, female={len(all_by_gender['f'])}")

    # 均匀分配到 4 个 pool
    splits = {"m": {}, "f": {}}
    for g in ["m", "f"]:
        utts = all_by_gender[g]
        n = len(utts)
        for pid in range(NUM_POOLS):
            s = pid * n // NUM_POOLS
            e = (pid + 1) * n // NUM_POOLS
            splits[g][pid] = utts[s:e]

    print("\n加载模型...")
    extractor = WavLMSSLExtractor(WAVLM_CKPT, layer=6, device=DEVICE)
    phone_predictor = PhonePredictor.load(PHONE_CKPT, device=DEVICE)

    for pid in range(NUM_POOLS):
        print(f"\n=== Pool {pid} ===")
        pool_dir = CKPT_DIR / f"pool_{pid}"
        append_to_pool(
            pool_dir,
            {"m": splits["m"][pid], "f": splits["f"][pid]},
            extractor,
            phone_predictor,
        )

    print("\n完成！情感数据已追加到所有 pool。")
    print("现在可以重新运行 build_multi_bank_v2_sil.py 重建 bank。")


if __name__ == "__main__":
    main()
