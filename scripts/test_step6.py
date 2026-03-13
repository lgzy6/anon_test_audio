#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 6 (Window-based): α-weighted Style-Guided Retrieval

Perform local controllable anonymization using window-level style matching.

Core idea:
  - Extract source window-level style trajectory
  - Match each source window to target pool windows using:
    sim = (1-α) * content_sim + α * style_sim
  - Replace source features with matched target features

Inputs:
  - checkpoints/style_extractor_window.pkl
  - checkpoints/target_pool_window_styles.npz (from Step 5)
  - checkpoints/eta_projection.pt
  - Source audio features

Outputs:
  - Anonymized features for vocoder synthesis
"""

import os
import sys
import json
import argparse
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import h5py
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.phone_predictor.predictor import PhonePredictor

# Import from test.py
from test import WindowConfig, WindowStyleExtractor


@dataclass
class RetrievalConfig:
    alpha: float = 0.2  # Style weight (0=content only, 1=style only)
    top_k: int = 1  # Top-k retrieval
    phone_bucket: bool = True  # Use phone-constrained retrieval
    smooth_trajectory: bool = True  # Smooth style trajectory before matching


class WindowStyleRetriever:
    """
    Window-level style-guided retriever with α-weighted similarity.
    """

    def __init__(self, extractor, pool_data, pool_feats, pool_phones, device="cuda"):
        self.extractor = extractor
        self.device = device

        # Target pool
        self.pool_utt_ids = pool_data["utt_ids"]
        self.pool_speaker_ids = pool_data["speaker_ids"]
        self.pool_genders = pool_data["genders"]
        self.pool_window_styles = pool_data["window_styles"]  # List of (W_i, 64)
        self.pool_window_starts = pool_data["window_starts"]
        self.pool_window_ends = pool_data["window_ends"]

        self.pool_feats = pool_feats.to(device)
        self.pool_phones = pool_phones.to(device)

        print(f"[Retriever] Loaded pool:")
        print(f"  Utterances: {len(self.pool_utt_ids)}")
        print(f"  Total windows: {sum(len(z) for z in self.pool_window_styles)}")

    def _build_phone_buckets(self):
        """Build phone-to-window index for fast lookup."""
        phone_to_windows = {}
        for utt_idx, (styles, starts, ends) in enumerate(zip(
            self.pool_window_styles, self.pool_window_starts, self.pool_window_ends
        )):
            for win_idx, (s, e) in enumerate(zip(starts, ends)):
                ph = self.pool_phones[s:e]
                dom_ph = int(ph[ph != 0].mode()[0]) if (ph != 0).any() else 0
                if dom_ph not in phone_to_windows:
                    phone_to_windows[dom_ph] = []
                phone_to_windows[dom_ph].append((utt_idx, win_idx, s, e))
        return phone_to_windows

    def retrieve(self, src_eta, src_phones, wcfg, rcfg):
        """
        Retrieve target features for source utterance.

        Returns:
            tgt_feats: (T, D) matched target features
            info: dict with matching statistics
        """
        # Extract source window styles
        src_styles, src_starts, src_ends = self.extractor.transform_windows(
            src_eta, src_phones, wcfg
        )

        if len(src_styles) == 0:
            return None, {"error": "No valid windows in source"}

        # Build phone buckets
        if rcfg.phone_bucket:
            phone_buckets = self._build_phone_buckets()

        # Match each source window
        T = len(src_eta)
        tgt_feats = torch.zeros(T, self.pool_feats.shape[1], device=self.device)
        match_info = []

        for i, (src_style, s, e) in enumerate(zip(src_styles, src_starts, src_ends)):
            src_ph = src_phones[s:e]
            dom_ph = int(src_ph[src_ph != 0].mode()[0]) if (src_ph != 0).any() else 0

            # Get candidate windows
            if rcfg.phone_bucket and dom_ph in phone_buckets:
                candidates = phone_buckets[dom_ph]
            else:
                # Fallback: all windows
                candidates = []
                for utt_idx, (styles, starts, ends) in enumerate(zip(
                    self.pool_window_styles, self.pool_window_starts, self.pool_window_ends
                )):
                    for win_idx, (ws, we) in enumerate(zip(starts, ends)):
                        candidates.append((utt_idx, win_idx, ws, we))

            if not candidates:
                tgt_feats[s:e] = torch.from_numpy(src_eta[s:e]).to(self.device)
                continue

            # Compute α-weighted similarity
            best_sim = -1
            best_cand = None

            for utt_idx, win_idx, ws, we in candidates:
                tgt_style = self.pool_window_styles[utt_idx][win_idx]

                # Style similarity (cosine)
                style_sim = float(np.dot(src_style, tgt_style))

                # Content similarity (phone overlap)
                tgt_ph = self.pool_phones[ws:we].cpu().numpy()
                content_sim = float((src_ph[:, None] == tgt_ph[None, :]).any(axis=1).mean())

                # Combined similarity
                combined_sim = (1 - rcfg.alpha) * content_sim + rcfg.alpha * style_sim

                if combined_sim > best_sim:
                    best_sim = combined_sim
                    best_cand = (ws, we, style_sim, content_sim)

            if best_cand:
                ws, we, style_sim, content_sim = best_cand
                tgt_feats[s:e] = self.pool_feats[ws:we]
                match_info.append({
                    "src_win": i,
                    "style_sim": style_sim,
                    "content_sim": content_sim,
                    "combined_sim": best_sim
                })

        avg_style_sim = np.mean([m["style_sim"] for m in match_info]) if match_info else 0
        avg_content_sim = np.mean([m["content_sim"] for m in match_info]) if match_info else 0

        return tgt_feats.cpu().numpy(), {
            "num_windows": len(src_styles),
            "avg_style_sim": float(avg_style_sim),
            "avg_content_sim": float(avg_content_sim),
            "matches": match_info
        }


def test_retrieval():
    """Test window-level style-guided retrieval on a sample utterance."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--test-utt-idx", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--win-ms", type=float, default=400.0)
    ap.add_argument("--hop-ms", type=float, default=200.0)
    ap.add_argument("--frame-ms", type=float, default=20.0)
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
    print("[1/4] Loading models...")
    extractor = WindowStyleExtractor.load(ckpt_dir / "style_extractor_window.pkl")

    pool_data = np.load(ckpt_dir / "target_pool_window_styles.npz", allow_pickle=True)

    eta_ckpt = torch.load(ckpt_dir / "eta_projection.pt", map_location=device)
    A = eta_ckpt["A_star"].to(device)
    b = eta_ckpt["b_star"].to(device)

    emb = np.load(ckpt_dir / "speaker_embeddings_pca.npy")
    utt_indices = np.load(ckpt_dir / "utt_indices.npy")
    utt_to_pca = {int(u): i for i, u in enumerate(utt_indices)}

    with open(cache_dir / "metadata.json", "r") as f:
        meta = json.load(f)
    utterances = meta["utterances"]

    phone_predictor = PhonePredictor.load(str(ckpt_dir / "phone_decoder.pt"), device=device)

    # Load pool features
    print("[2/4] Loading pool features...")
    with h5py.File(cache_dir / "features.h5", "r") as f:
        pool_feats = torch.from_numpy(f["features"][:]).float()

    # Compute pool phones
    print("[3/4] Computing pool phones...")
    with h5py.File(cache_dir / "features.h5", "r") as f:
        feats_ds = f["features"]
        pool_phones_list = []
        for utt in tqdm(utterances[:len(pool_data["utt_ids"])], desc="Pool phones"):
            s0, s1 = utt["h5_start_idx"], utt["h5_end_idx"]
            s = torch.from_numpy(feats_ds[s0:s1]).float().to(device)
            with torch.inference_mode():
                ph = phone_predictor(s).cpu()
            pool_phones_list.append(ph)
    pool_phones = torch.cat(pool_phones_list)

    # Initialize retriever
    retriever = WindowStyleRetriever(extractor, pool_data, pool_feats, pool_phones, device)

    # Test on source utterance
    print(f"\n[4/4] Testing retrieval on utterance {args.test_utt_idx}...")
    test_utt = utterances[args.test_utt_idx]
    pca_idx = utt_to_pca[args.test_utt_idx]

    with h5py.File(cache_dir / "features.h5", "r") as f:
        s_np = f["features"][test_utt["h5_start_idx"]:test_utt["h5_end_idx"]]

    s = torch.from_numpy(s_np).float().to(device)
    d = torch.from_numpy(emb[pca_idx]).float().to(device)
    eta = (s - (d @ A + b).unsqueeze(0)).cpu().numpy()

    with torch.inference_mode():
        phones = phone_predictor(s).cpu().numpy()

    win_frames = max(2, int(round(args.win_ms / args.frame_ms)))
    hop_frames = max(1, int(round(args.hop_ms / args.frame_ms)))
    wcfg = WindowConfig(win_frames, hop_frames, 0.4, 0)
    rcfg = RetrievalConfig(alpha=args.alpha)

    tgt_feats, info = retriever.retrieve(eta, phones, wcfg, rcfg)

    print(f"\n{'='*50}")
    print(f"RESULT:")
    print(f"  Windows matched: {info['num_windows']}")
    print(f"  Avg style sim: {info['avg_style_sim']:.3f}")
    print(f"  Avg content sim: {info['avg_content_sim']:.3f}")
    print(f"  α={args.alpha} (style weight)")
    print(f"{'='*50}")


if __name__ == "__main__":
    test_retrieval()
