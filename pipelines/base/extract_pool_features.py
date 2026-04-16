#!/usr/bin/env python3
"""
为 pool_splits.json 中的说话人提取 WavLM L6/L24 特征 + 音素预测。
m/f 单独提取，mix 通过合并 m/f 生成（不重复提取音频）。
"""
import os
os.environ["OMP_NUM_THREADS"] = "4"

import sys
import json
import argparse
import h5py
import numpy as np
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from models.ssl.wrappers import WavLMSSLExtractor
from models.phone_predictor.predictor import PhonePredictor

CKPT_DIR    = Path("/root/autodl-tmp/anon_test/checkpoints")
AUDIO_DIR   = Path("/root/autodl-tmp/datasets/LibriTTS/train-other-500")
SPLITS_JSON = CKPT_DIR / "pool_splits.json"
LAYERS      = [6, 24]
SR          = 16000
HOP         = 320


def scan_utterances(spk_ids, audio_dir, gender_map):
    utts = []
    for spk_id in spk_ids:
        spk_dir = audio_dir / spk_id
        if not spk_dir.exists():
            continue
        for wav_path in sorted(spk_dir.rglob("*.wav")):
            utts.append({
                "utt_id":     wav_path.stem,
                "speaker_id": spk_id,
                "gender":     gender_map.get(spk_id, "m"),
                "path":       str(wav_path),
            })
    return utts


@torch.no_grad()
def extract_and_save(utts, out_dir, extractor, phone_predictor, batch_size=4):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    h5_files = {l: h5py.File(out_dir / f"layer_{l}.h5", "w") for l in LAYERS}
    ds = {l: h5_files[l].create_dataset("features", shape=(0, 1024),
          maxshape=(None, 1024), dtype="float32", chunks=(1000, 1024)) for l in LAYERS}
    phones_h5 = h5py.File(out_dir / "phones.h5", "w")
    phones_ds = phones_h5.create_dataset("phones", shape=(0,), maxshape=(None,),
                                          dtype="int32", chunks=(10000,))

    metadata = {"utterances": [], "total_frames": 0}
    frame_idx = 0

    for i in tqdm(range(0, len(utts), batch_size), desc=f"Extracting {out_dir.name}"):
        batch = utts[i:i+batch_size]
        wavs, valid = [], []
        for u in batch:
            try:
                wav, sr = torchaudio.load(u["path"])
                if sr != SR:
                    wav = torchaudio.functional.resample(wav, sr, SR)
                wavs.append(wav.mean(0))
                valid.append(u)
            except Exception:
                continue

        if not wavs:
            continue

        max_len = max(w.shape[0] for w in wavs)
        padded = torch.zeros(len(wavs), max_len)
        for j, w in enumerate(wavs):
            padded[j, :w.shape[0]] = w
        padded = padded.to(extractor.device)

        multi = extractor.forward_multi_layer(padded, layers=LAYERS)

        for j, u in enumerate(valid):
            T_actual = min(wavs[j].shape[0] // HOP, multi[24][j].shape[0])
            for l in LAYERS:
                feat = multi[l][j, :T_actual].cpu().numpy()
                cur = ds[l].shape[0]
                ds[l].resize((cur + T_actual, 1024))
                ds[l][cur:cur + T_actual] = feat

            phone_ids = phone_predictor(multi[24][j, :T_actual]).cpu().numpy()
            cur_p = phones_ds.shape[0]
            phones_ds.resize((cur_p + T_actual,))
            phones_ds[cur_p:cur_p + T_actual] = phone_ids

            metadata["utterances"].append({
                "utt_id":       u["utt_id"],
                "speaker_id":   u["speaker_id"],
                "gender":       u["gender"],
                "h5_start_idx": frame_idx,
                "h5_end_idx":   frame_idx + T_actual,
            })
            frame_idx += T_actual

    metadata["total_frames"] = frame_idx
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)
    for h in h5_files.values():
        h.close()
    phones_h5.close()
    print(f"  {out_dir.name}: {len(metadata['utterances'])} utts, {frame_idx} frames")


def merge_mix(pool_dir):
    """合并 m/f 的 h5 生成 mix，不重新提取音频。"""
    mix_dir = pool_dir / "mix"
    mix_dir.mkdir(parents=True, exist_ok=True)

    combined_utts, frame_offset = [], 0
    for gender in ["m", "f"]:
        meta_path = pool_dir / gender / "metadata.json"
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        for u in meta["utterances"]:
            u = dict(u)
            u["h5_start_idx"] += frame_offset
            u["h5_end_idx"] += frame_offset
            combined_utts.append(u)
        frame_offset += meta["total_frames"]

    total_frames = frame_offset

    for fname in ["layer_6.h5", "layer_24.h5"]:
        with h5py.File(mix_dir / fname, "w") as out:
            ds_out = out.create_dataset("features", shape=(total_frames, 1024),
                                        dtype="float32", chunks=(1000, 1024))
            offset = 0
            for gender in ["m", "f"]:
                src = pool_dir / gender / fname
                if not src.exists():
                    continue
                with h5py.File(src) as inp:
                    data = inp["features"][:]
                    ds_out[offset:offset + len(data)] = data
                    offset += len(data)

    with h5py.File(mix_dir / "phones.h5", "w") as out:
        ds_out = out.create_dataset("phones", shape=(total_frames,),
                                    dtype="int32", chunks=(10000,))
        offset = 0
        for gender in ["m", "f"]:
            src = pool_dir / gender / "phones.h5"
            if not src.exists():
                continue
            with h5py.File(src) as inp:
                data = inp["phones"][:]
                ds_out[offset:offset + len(data)] = data
                offset += len(data)

    with open(mix_dir / "metadata.json", "w") as f:
        json.dump({"utterances": combined_utts, "total_frames": total_frames}, f)
    print(f"  mix: merged {len(combined_utts)} utts, {total_frames} frames")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", default=str(SPLITS_JSON))
    parser.add_argument("--audio-dir", default=str(AUDIO_DIR))
    parser.add_argument("--pool", type=int, default=None, help="只处理指定 pool (0-3)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    with open(args.splits) as f:
        splits = json.load(f)

    gender_map = {}
    for pool in splits["pools"]:
        for spk in pool["male"]:
            gender_map[spk] = "m"
        for spk in pool["female"]:
            gender_map[spk] = "f"

    extractor = WavLMSSLExtractor(
        ckpt_path=str(CKPT_DIR / "WavLM-Large.pt"),
        layer=6, device=args.device,
    )
    phone_predictor = PhonePredictor.load(str(CKPT_DIR / "phone_decoder.pt"), device=args.device)

    pools = splits["pools"]
    if args.pool is not None:
        pools = [p for p in pools if p["pool_id"] == args.pool]

    for pool in pools:
        pid = pool["pool_id"]
        print(f"\n=== Pool {pid} ===")
        pool_dir = CKPT_DIR / f"pool_{pid}"
        for gender, spk_ids in [("m", pool["male"]), ("f", pool["female"])]:
            out_dir = pool_dir / gender
            utts = scan_utterances(spk_ids, Path(args.audio_dir), gender_map)
            if not utts:
                print(f"  {gender}: 无音频，跳过")
                continue
            extract_and_save(utts, out_dir, extractor, phone_predictor, args.batch_size)
        merge_mix(pool_dir)


if __name__ == "__main__":
    main()
