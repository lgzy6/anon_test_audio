#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 4 (Window, LOCAL controllable style) - Improved

Improvements vs previous version:
1) Prosody channel isolation & amplification:
   - First PCA on high-dim eta-stats (e.g., 4096 dims) -> eta_pca_dim (default 59)
   - Concatenate 5-dim energy-shape features -> total 64 dims
   - Optional energy_weight to amplify prosody signal

2) Control whitening & L2 timing:
   - PCA(whiten=True/False) toggle
   - Optional pre-L2 before PCA (pre_l2) and final L2 after concat (always)

3) Temporal smoothing:
   - Moving Average (MA) or EMA on window style trajectory to reduce jitter

Outputs:
- checkpoints/style_extractor_window.pkl

Validation:
- trajectory smoothness / prosody corr / clustering sanity (optional)
"""

import os
import sys
import json
import pickle
import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import h5py
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.phone_predictor.predictor import PhonePredictor


@dataclass
class WindowConfig:
    win_frames: int
    hop_frames: int
    min_voiced_ratio: float = 0.6
    silence_id: int = 0


def l2norm(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (n + eps)


def moving_average(z: np.ndarray, k: int) -> np.ndarray:
    """Simple centered moving average, k must be odd. If k<=1, return z."""
    if k is None or k <= 1 or len(z) < 3:
        return z
    k = int(k)
    if k % 2 == 0:
        k += 1
    pad = k // 2
    z_pad = np.pad(z, ((pad, pad), (0, 0)), mode="edge")
    out = np.empty_like(z, dtype=np.float32)
    for i in range(len(z)):
        out[i] = z_pad[i:i + k].mean(axis=0)
    return out


def ema_smooth(z: np.ndarray, alpha: float) -> np.ndarray:
    """Exponential moving average smoothing. alpha in (0,1], larger=less smoothing."""
    if alpha is None or alpha <= 0 or alpha >= 1 or len(z) < 2:
        return z
    out = np.empty_like(z, dtype=np.float32)
    out[0] = z[0]
    for i in range(1, len(z)):
        out[i] = alpha * z[i] + (1 - alpha) * out[i - 1]
    return out


class WindowStyleExtractor:
    """
    Window-level style extractor with prosody isolation:

    raw_eta_stats = [mean(η), std(η), mean(|Δη|), std(Δη)]  -> PCA_eta -> z_eta (eta_pca_dim)
    raw_energy    = [e_mean, e_std, e_range, e_slope, e_contrast] -> scale by energy_weight
    z = concat(z_eta, raw_energy)  -> final L2

    Optional:
    - PCA_eta whitening on/off
    - pre_l2 before PCA_eta on/off
    - temporal smoothing after transform_windows
    """

    def __init__(
        self,
        final_dim: int = 64,
        eta_pca_dim: int = 59,
        whiten: bool = True,
        pre_l2: bool = True,
        energy_weight: float = 3.0,
        smooth_ma_k: int = 0,
        smooth_ema_alpha: float = 0.0,
    ):
        assert final_dim == 64, "This implementation targets final_dim=64 for downstream usage."
        assert eta_pca_dim <= final_dim - 7, "Need room for 7 prosody dims."
        self.final_dim = final_dim
        self.eta_pca_dim = eta_pca_dim
        self.whiten = whiten
        self.pre_l2 = pre_l2
        self.energy_weight = energy_weight

        # Smoothing params (applied in transform_windows if enabled)
        self.smooth_ma_k = smooth_ma_k
        self.smooth_ema_alpha = smooth_ema_alpha

        self.pca_eta: PCA | None = None
        self.eta_raw_dim: int | None = None

    def _eta_stats(self, eta_win: np.ndarray) -> np.ndarray:
        """
        High-dim eta stats for content-independent style:
        - mu, sd, mean(|Δ|), std(Δ)
        """
        mu = eta_win.mean(axis=0)
        sd = eta_win.std(axis=0)

        if eta_win.shape[0] >= 2:
            delta = eta_win[1:] - eta_win[:-1]
            delta_abs_mean = np.abs(delta).mean(axis=0)
            delta_std = delta.std(axis=0)
        else:
            delta_abs_mean = np.zeros_like(mu)
            delta_std = np.zeros_like(mu)

        raw = np.concatenate([mu, sd, delta_abs_mean, delta_std], axis=0).astype(np.float32)
        return raw

    def _energy_feats(self, eta_win: np.ndarray, phones_win: np.ndarray = None) -> np.ndarray:
        """
        7-dim prosody features:
        [e_mean, e_std, e_range, e_slope, e_contrast, voiced_ratio, silence_ratio]
        """
        mag = np.linalg.norm(eta_win, axis=1).astype(np.float32)
        T = len(mag)

        e_mean = float(mag.mean())
        e_std = float(mag.std())
        e_range = float(mag.max() - mag.min())

        if T >= 4:
            t = np.arange(T, dtype=np.float32)
            e_slope = float(np.polyfit(t, mag, 1)[0])
            mid = T // 2
            e_contrast = float(mag[mid:].mean() - mag[:mid].mean())
        else:
            e_slope = 0.0
            e_contrast = 0.0

        # Timing features (voicing/silence ratio)
        if phones_win is not None:
            voiced_ratio = float((phones_win != 0).mean())
            silence_ratio = float((phones_win == 0).mean())
        else:
            voiced_ratio = 0.5
            silence_ratio = 0.5

        e = np.array([e_mean, e_std, e_range, e_slope, e_contrast, voiced_ratio, silence_ratio], dtype=np.float32)
        return e

    def fit(self, eta: np.ndarray, phones: np.ndarray, cfg: WindowConfig, max_windows: int | None = None) -> None:
        """
        Fit PCA on eta-stats only (high-dim), then energy is appended later.
        """
        win = cfg.win_frames
        hop = cfg.hop_frames
        n = len(eta)

        eta_stats_list = []
        energy_list = []

        for s in range(0, n - win + 1, hop):
            e = s + win
            ph = phones[s:e]
            voiced = (ph != cfg.silence_id).mean()
            if voiced < cfg.min_voiced_ratio:
                continue

            eta_win = eta[s:e]
            ph_win = phones[s:e]
            eta_stats_list.append(self._eta_stats(eta_win))
            energy_list.append(self._energy_feats(eta_win, ph_win))

            if max_windows is not None and len(eta_stats_list) >= max_windows:
                break

        if not eta_stats_list:
            raise RuntimeError("No valid windows found. Lower min_voiced_ratio or check phones.")

        X_eta = np.stack(eta_stats_list, axis=0)
        X_e = np.stack(energy_list, axis=0)

        if self.pre_l2:
            X_eta = l2norm(X_eta)
            # energy we do NOT l2norm; we scale it explicitly and keep absolute meaning

        self.eta_raw_dim = X_eta.shape[1]

        # PCA on eta stats only
        eta_dim = min(self.eta_pca_dim, X_eta.shape[1], X_eta.shape[0] - 1)
        self.pca_eta = PCA(n_components=eta_dim, whiten=self.whiten, random_state=42)
        Z_eta = self.pca_eta.fit_transform(X_eta).astype(np.float32)

        # Append energy (scaled) to form final 64 dims
        # We keep energy in raw space but scale it to be "heard" by similarity
        Z = np.concatenate([Z_eta, X_e * float(self.energy_weight)], axis=1).astype(np.float32)

        # Final normalize for cosine stability
        Z = l2norm(Z)

        explained = float(self.pca_eta.explained_variance_ratio_.sum())
        print(f"[Step4] Fitted WindowStyleExtractor (prosody-isolated)")
        print(f"  windows used: {len(Z)}")
        print(f"  eta raw dim: {self.eta_raw_dim} -> eta pca dim: {eta_dim} (whiten={self.whiten})")
        print(f"  final dim: {Z.shape[1]} (= eta_pca_dim + 7 prosody)")
        print(f"  eta PCA explained variance: {explained:.2%}")
        print(f"  energy_weight: {self.energy_weight}")
        if self.smooth_ma_k and self.smooth_ma_k > 1:
            print(f"  smoothing: MA(k={self.smooth_ma_k})")
        elif self.smooth_ema_alpha and self.smooth_ema_alpha > 0:
            print(f"  smoothing: EMA(alpha={self.smooth_ema_alpha})")
        else:
            print(f"  smoothing: off")

    def transform_windows(self, eta: np.ndarray, phones: np.ndarray, cfg: WindowConfig):
        """
        Extract window-level style sequence.
        Returns:
          z: (W, 64)
          win_starts: (W,)
          win_ends: (W,)
        """
        assert self.pca_eta is not None, "Call fit() first."

        win = cfg.win_frames
        hop = cfg.hop_frames
        n = len(eta)

        eta_stats_list = []
        energy_list = []
        win_starts = []
        win_ends = []

        for s in range(0, n - win + 1, hop):
            e = s + win
            ph = phones[s:e]
            voiced = (ph != cfg.silence_id).mean()
            if voiced < cfg.min_voiced_ratio:
                continue

            eta_win = eta[s:e]
            ph_win = phones[s:e]
            eta_stats_list.append(self._eta_stats(eta_win))
            energy_list.append(self._energy_feats(eta_win, ph_win))
            win_starts.append(s)
            win_ends.append(e)

        if not eta_stats_list:
            return (np.zeros((0, self.final_dim), dtype=np.float32),
                    np.zeros((0,), dtype=np.int32),
                    np.zeros((0,), dtype=np.int32))

        X_eta = np.stack(eta_stats_list, axis=0).astype(np.float32)
        X_e = np.stack(energy_list, axis=0).astype(np.float32)

        if self.pre_l2:
            X_eta = l2norm(X_eta)

        Z_eta = self.pca_eta.transform(X_eta).astype(np.float32)
        Z = np.concatenate([Z_eta, X_e * float(self.energy_weight)], axis=1).astype(np.float32)
        Z = l2norm(Z)

        # Temporal smoothing on final z
        if self.smooth_ma_k and self.smooth_ma_k > 1:
            Z = moving_average(Z, self.smooth_ma_k)
            Z = l2norm(Z)
        elif self.smooth_ema_alpha and self.smooth_ema_alpha > 0:
            Z = ema_smooth(Z, self.smooth_ema_alpha)
            Z = l2norm(Z)

        return Z, np.array(win_starts, dtype=np.int32), np.array(win_ends, dtype=np.int32)

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path):
        with open(path, "rb") as f:
            return pickle.load(f)


def compute_eta_for_utterances(cache_dir: Path, ckpt_dir: Path, max_frames: int | None = None):
    """
    Compute η = s - (d @ A + b) for utterances in cache_dir.
    Returns: eta_all, phones_all, spk_all
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    eta_ckpt = torch.load(ckpt_dir / "eta_projection.pt", map_location=device)
    A = eta_ckpt["A_star"].to(device)
    b = eta_ckpt["b_star"].to(device)

    emb = np.load(ckpt_dir / "speaker_embeddings_pca.npy")
    utt_indices = np.load(ckpt_dir / "utt_indices.npy")

    with open(cache_dir / "metadata.json", "r") as f:
        meta = json.load(f)
    utterances = meta["utterances"]

    utt_to_pca = {int(u): i for i, u in enumerate(utt_indices)}

    phone_predictor = None
    phone_ckpt = ckpt_dir / "phone_decoder.pt"
    if phone_ckpt.exists():
        phone_predictor = PhonePredictor.load(str(phone_ckpt), device=device)

    all_eta = []
    all_phones = []
    all_spk = []
    total = 0

    with h5py.File(cache_dir / "features.h5", "r") as f:
        feats = f["features"]
        for utt_idx, utt in enumerate(tqdm(utterances, desc="Computing η")):
            if utt_idx not in utt_to_pca:
                continue
            pca_idx = utt_to_pca[utt_idx]
            s0, s1 = utt["h5_start_idx"], utt["h5_end_idx"]
            if s1 <= s0:
                continue

            s_np = feats[s0:s1]
            T = s_np.shape[0]
            if T == 0:
                continue

            s = torch.from_numpy(s_np).float().to(device)
            d = torch.from_numpy(emb[pca_idx]).float().to(device)
            eta = s - (d @ A + b).unsqueeze(0)

            if phone_predictor is not None:
                with torch.inference_mode():
                    ph = phone_predictor(s).cpu().numpy().astype(np.int16)
            else:
                ph = np.random.randint(0, 41, size=T).astype(np.int16)

            all_eta.append(eta.cpu().numpy())
            all_phones.append(ph)
            all_spk.extend([utt["speaker_id"]] * T)

            total += T
            if max_frames is not None and total >= max_frames:
                break

    eta_all = np.concatenate(all_eta, axis=0)[:max_frames].astype(np.float32) if max_frames else np.concatenate(all_eta, axis=0).astype(np.float32)
    phones_all = np.concatenate(all_phones, axis=0)[:max_frames].astype(np.int16) if max_frames else np.concatenate(all_phones, axis=0).astype(np.int16)
    spk_all = np.array(all_spk[:max_frames]) if max_frames else np.array(all_spk)

    return eta_all, phones_all, spk_all


def validate_quality(z: np.ndarray, spk: np.ndarray, phones: np.ndarray, cfg: WindowConfig, win_starts, win_ends, eta: np.ndarray):
    """
    Validation for local controllability:
    1) Trajectory smoothness
    2) Prosody correlation proxy (energy)
    3) Clustering sanity (optional)
    """
    if len(z) < 100:
        print("[Validation] Too few windows, skipping.")
        return

    print("\n[Validation] Window-level style quality")
    print(f"  Windows: {len(z)}, Speakers: {len(np.unique(spk))}")

    # 1) Trajectory smoothness
    if len(z) > 1:
        diffs = np.linalg.norm(z[1:] - z[:-1], axis=1)
        print(f"\n  Trajectory smoothness:")
        print(f"    Mean step: {diffs.mean():.4f}")
        print(f"    Std step:  {diffs.std():.4f}")
        print(f"    P90 step:  {np.quantile(diffs, 0.90):.4f}")
        print(f"    P99 step:  {np.quantile(diffs, 0.99):.4f}")
        print(f"    Max step:  {diffs.max():.4f}")
        if diffs.mean() < 0.5:
            print("    ✓ Smooth enough for local guidance")
        else:
            print("    ⚠ High jitter (consider overlap windows and smoothing)")

    # 2) Prosody correlation (energy dynamics)
    win_energy = []
    for s, e in zip(win_starts, win_ends):
        mag = np.linalg.norm(eta[s:e], axis=1)
        win_energy.append(float(mag.mean()))
    win_energy = np.array(win_energy, dtype=np.float32)

    if len(win_energy) > 10 and len(z) > 1:
        # Extract energy dimension from style vector (last 7 dims are prosody features)
        z_energy_dim = z[:, -7]  # First prosody feature is e_mean

        # Absolute correlation (not affected by smoothing)
        corr_abs = float(np.corrcoef(z_energy_dim, win_energy)[0, 1])

        # Change correlation (adjacent windows)
        style_change = np.linalg.norm(z[1:] - z[:-1], axis=1)
        energy_change = np.abs(win_energy[1:] - win_energy[:-1])

        # Long-range change correlation (every 3 windows to avoid smoothing suppression)
        if len(z) > 6:
            style_change_long = np.linalg.norm(z[3:] - z[:-3], axis=1)
            energy_change_long = np.abs(win_energy[3:] - win_energy[:-3])
            corr_change_long = float(np.corrcoef(style_change_long, energy_change_long)[0, 1]) if energy_change_long.std() > 1e-6 else 0.0
        else:
            corr_change_long = 0.0

        corr_change = float(np.corrcoef(style_change, energy_change)[0, 1]) if energy_change.std() > 1e-6 else 0.0

        print(f"\n  Prosody correlation:")
        print(f"    Energy-dim absolute corr: {corr_abs:.3f}")
        print(f"    Style-Energy change corr (adj): {corr_change:.3f}")
        print(f"    Style-Energy change corr (long): {corr_change_long:.3f}")

        best_corr = max(corr_abs, corr_change, corr_change_long)
        if best_corr > 0.3:
            print("    ✓ Strong prosody signal")
        elif best_corr > 0.15:
            print("    ◐ Moderate prosody signal")
        else:
            print("    ⚠ Weak prosody signal (increase energy_weight)")

    # 3) Clustering sanity (not a gate for continuous prosody)
    print(f"\n  Clustering sanity (optional):")
    print("    K   Silhouette    ARI_spk   ARI_phone")
    print("  ---------------------------------------------")

    # Dominant phone per window
    dom_phone = []
    for s, e in zip(win_starts, win_ends):
        ph = phones[s:e]
        ph2 = ph[ph != cfg.silence_id]
        if len(ph2) == 0:
            dom_phone.append(cfg.silence_id)
        else:
            vals, cnts = np.unique(ph2, return_counts=True)
            dom_phone.append(int(vals[np.argmax(cnts)]))
    dom_phone = np.array(dom_phone)

    # Dominant speaker per window
    dom_spk = []
    for s, e in zip(win_starts, win_ends):
        w = spk[s:e]
        vals, cnts = np.unique(w, return_counts=True)
        dom_spk.append(vals[np.argmax(cnts)])
    dom_spk = np.array(dom_spk)

    uniq_spk = np.unique(dom_spk)
    spk_map = {s: i for i, s in enumerate(uniq_spk)}
    spk_int = np.array([spk_map[s] for s in dom_spk])

    best_sil = -1
    for K in [4, 8, 16]:
        km = KMeans(n_clusters=K, random_state=42, n_init=10)
        labels = km.fit_predict(z)
        sil = silhouette_score(z, labels) if len(np.unique(labels)) > 1 else 0.0
        ari_spk = adjusted_rand_score(spk_int, labels)
        ari_ph = adjusted_rand_score(dom_phone, labels)
        print(f"  {K:>5d}   {sil:>10.4f}   {ari_spk:>8.4f}   {ari_ph:>9.4f}")
        best_sil = max(best_sil, sil)

    print(f"\n  RESULT: Best Silhouette={best_sil:.4f}")
    print("  Note: low silhouette is acceptable for continuous prosody control.")

    # 4) t-SNE visualization
    output_dir = Path(__file__).parent.parent / 'outputs' / 'style_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(z) > 100:
        print(f"\n  Generating t-SNE visualization...")
        vis_idx = np.random.choice(len(z), min(2000, len(z)), replace=False)
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        z_2d = tsne.fit_transform(z[vis_idx])

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Best K clusters
        km = KMeans(n_clusters=4, random_state=42, n_init=10)
        labels = km.fit_predict(z[vis_idx])
        axes[0].scatter(z_2d[:, 0], z_2d[:, 1], c=labels, cmap='tab10', s=10, alpha=0.6)
        axes[0].set_title('Window Style Clusters (K=4)')

        # By speaker
        spk_vis = spk_int[vis_idx]
        axes[1].scatter(z_2d[:, 0], z_2d[:, 1], c=spk_vis, cmap='tab20', s=10, alpha=0.6)
        axes[1].set_title(f'By Speaker ({len(uniq_spk)} speakers)')

        # By phone
        phone_vis = dom_phone[vis_idx]
        axes[2].scatter(z_2d[:, 0], z_2d[:, 1], c=phone_vis, cmap='tab20', s=10, alpha=0.6)
        axes[2].set_title('By Dominant Phone')

        plt.tight_layout()
        save_path = output_dir / 'window_style_clustering.png'
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  Saved: {save_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--max-frames", type=int, default=None, help="None = all frames")

    ap.add_argument("--win-ms", type=float, default=400.0)
    ap.add_argument("--hop-ms", type=float, default=200.0)
    ap.add_argument("--frame-ms", type=float, default=20.0)
    ap.add_argument("--min-voiced-ratio", type=float, default=0.4)

    # extractor params
    ap.add_argument("--eta-pca-dim", type=int, default=57, help="eta-stats PCA dim; final dim = eta_pca_dim + 7")
    ap.add_argument("--no-whiten", dest="whiten", action="store_false", default=True, help="Disable PCA whitening")
    ap.add_argument("--no-pre-l2", dest="pre_l2", action="store_false", default=True, help="Disable pre-L2")
    ap.add_argument("--energy-weight", type=float, default=6.0, help="scale prosody features before concat")

    # smoothing
    ap.add_argument("--smooth-ma-k", type=int, default=3, help="odd window size for moving average; 0/1 to disable")
    ap.add_argument("--smooth-ema-alpha", type=float, default=0.0, help="EMA alpha in (0,1); 0 to disable")

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

    win_frames = max(2, int(round(args.win_ms / args.frame_ms)))
    hop_frames = max(1, int(round(args.hop_ms / args.frame_ms)))
    wcfg = WindowConfig(
        win_frames=win_frames,
        hop_frames=hop_frames,
        min_voiced_ratio=args.min_voiced_ratio,
        silence_id=0
    )

    print("[1/3] Computing η across utterances...")
    eta, phones, spk = compute_eta_for_utterances(cache_dir, ckpt_dir, max_frames=args.max_frames)
    print(f"  Total: {len(eta)} frames, speakers: {len(np.unique(spk))}")

    print("\n[2/3] Fitting WindowStyleExtractor...")
    ext = WindowStyleExtractor(
        final_dim=64,
        eta_pca_dim=args.eta_pca_dim,
        whiten=args.whiten,
        pre_l2=args.pre_l2,
        energy_weight=args.energy_weight,
        smooth_ma_k=args.smooth_ma_k,
        smooth_ema_alpha=args.smooth_ema_alpha,
    )
    ext.fit(eta, phones, wcfg, max_windows=None)
    save_path = ckpt_dir / "style_extractor_window.pkl"
    ext.save(save_path)
    print(f"  Saved: {save_path}")

    print("\n[3/3] Validating window style quality...")
    z, ws, we = ext.transform_windows(eta, phones, wcfg)
    validate_quality(z, spk, phones, wcfg, ws, we, eta)


if __name__ == "__main__":
    main()
