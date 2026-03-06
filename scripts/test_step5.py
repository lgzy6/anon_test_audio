#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 5 (Window-based): Target Pool Style Precomputation

Precompute window-level style embeddings for all target pool utterances.
This enables fast α-weighted style-guided retrieval in Step 6.

Inputs:
  - checkpoints/style_extractor_window.pkl (from Step 4)
  - checkpoints/eta_projection.pt
  - checkpoints/speaker_embeddings_pca.npy
  - cache/features/wavlm/{split}/features.h5

Outputs:
  - checkpoints/target_pool_window_styles.npz
    {
      'utt_ids': list of utterance indices
      'speaker_ids': (N_utt,)
      'genders': (N_utt,)
      'window_styles': list of (W_i, 64) arrays per utterance
      'window_starts': list of (W_i,) arrays
      'window_ends': list of (W_i,) arrays
    }
"""

import os
import sys
import json
import pickle
import argparse
from pathlib import Path

import numpy as np
import torch
import h5py
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.phone_predictor.predictor import PhonePredictor

# Import WindowConfig and WindowStyleExtractor from test.py
exec(open(Path(__file__).parent / "test.py").read(), globals())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--win-ms", type=float, default=400.0)
    ap.add_argument("--hop-ms", type=float, default=200.0)
    ap.add_argument("--frame-ms", type=float, default=20.0)
    ap.add_argument("--min-voiced-ratio", type=float, default=0.4)
    ap.add_argument("--max-utts", type=int, default=None, help="Limit utterances for testing")
    args = ap.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    ckpt_dir = base_dir / "checkpoints"
    cache_dir = base_dir / "cache" / "features" / "wavlm"

    cfg_path = Path(args.config) if args.config else (base_dir / "configs" / "base.yaml")
    if cfg_path.exists():
        import yaml
        with open(cfg_path, "r") as f:
            y = yaml.safe_load(f)
        train_split = y["offline"].get("train_split", "train-clean-100")
        split_name = train_split.replace("-", "_")
        cache_dir = Path(y["paths"]["cache_dir"]) / "features" / "wavlm" / split_name
        ckpt_dir = Path(y["paths"]["checkpoints_dir"])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load models
    print("[1/3] Loading models...")
    extractor = WindowStyleExtractor.load(ckpt_dir / "style_extractor_window.pkl")

    eta_ckpt = torch.load(ckpt_dir / "eta_projection.pt", map_location=device)
    A = eta_ckpt["A_star"].to(device)
    b = eta_ckpt["b_star"].to(device)

    emb = np.load(ckpt_dir / "speaker_embeddings_pca.npy")
    utt_indices = np.load(ckpt_dir / "utt_indices.npy")
    utt_to_pca = {int(u): i for i, u in enumerate(utt_indices)}

    with open(cache_dir / "metadata.json", "r") as f:
        meta = json.load(f)
    utterances = meta["utterances"]

    phone_ckpt = ckpt_dir / "phone_decoder.pt"
    phone_predictor = PhonePredictor.load(str(phone_ckpt), device=device) if phone_ckpt.exists() else None

    # Window config
    win_frames = max(2, int(round(args.win_ms / args.frame_ms)))
    hop_frames = max(1, int(round(args.hop_ms / args.frame_ms)))
    wcfg = WindowConfig(
        win_frames=win_frames,
        hop_frames=hop_frames,
        min_voiced_ratio=args.min_voiced_ratio,
        silence_id=0
    )

    # Precompute
    print(f"\n[2/3] Precomputing window styles for {len(utterances)} utterances...")

    utt_ids = []
    speaker_ids = []
    genders = []
    window_styles = []
    window_starts = []
    window_ends = []

    with h5py.File(cache_dir / "features.h5", "r") as f:
        feats = f["features"]
        for utt_idx, utt in enumerate(tqdm(utterances, desc="Processing")):
            if args.max_utts and len(utt_ids) >= args.max_utts:
                break
            if utt_idx not in utt_to_pca:
                continue

            pca_idx = utt_to_pca[utt_idx]
            s0, s1 = utt["h5_start_idx"], utt["h5_end_idx"]
            if s1 <= s0:
                continue

            s_np = feats[s0:s1]
            if s_np.shape[0] == 0:
                continue

            s = torch.from_numpy(s_np).float().to(device)
            d = torch.from_numpy(emb[pca_idx]).float().to(device)
            eta = s - (d @ A + b).unsqueeze(0)

            if phone_predictor:
                with torch.inference_mode():
                    ph = phone_predictor(s).cpu().numpy().astype(np.int16)
            else:
                ph = np.random.randint(0, 41, size=len(s)).astype(np.int16)

            eta_np = eta.cpu().numpy()
            Z, ws, we = extractor.transform_windows(eta_np, ph, wcfg)

            if len(Z) == 0:
                continue

            utt_ids.append(utt_idx)
            speaker_ids.append(utt["speaker_id"])
            genders.append(utt.get("gender", "unknown"))
            window_styles.append(Z)
            window_starts.append(ws)
            window_ends.append(we)

    # Flatten all windows into single arrays
    all_styles = np.vstack(window_styles)
    all_starts = np.concatenate(window_starts)
    all_ends = np.concatenate(window_ends)

    # Build boundaries for each utterance
    boundaries = []
    offset = 0
    for ws in window_styles:
        boundaries.append([offset, offset + len(ws)])
        offset += len(ws)

    speaker_ids = np.array(speaker_ids)
    genders = np.array(genders)
    boundaries = np.array(boundaries)

    # Save
    print(f"\n[3/3] Saving precomputed styles...")
    save_path = ckpt_dir / "target_pool_window_styles.npz"
    np.savez(
        save_path,
        utt_ids=np.array(utt_ids),
        speaker_ids=speaker_ids,
        genders=genders,
        styles=all_styles,
        window_starts=all_starts,
        window_ends=all_ends,
        utt_boundaries=boundaries
    )

    print(f"  Saved: {save_path}")
    print(f"  Utterances: {len(utt_ids)}")
    print(f"  Speakers: {len(np.unique(speaker_ids))}")
    print(f"  Total windows: {len(all_styles)}")
    print(f"  Avg windows/utt: {len(all_styles) / len(utt_ids):.1f}")


if __name__ == "__main__":
    main()
