#!/usr/bin/env python3
"""
Bank 构建 v2-centroid + 静音桶(L6聚类)

【与原 build_multi_bank_v2_centroid_sil.py 的唯一差异】
  _build_silence_phone 函数:
    原版: L24 KMeans → L24 距离选帧 → 存原始帧
    本版: L6 KMeans → L6 距离选帧 → L6 二次聚类 → 存质心
  
  原因: 静音帧的 L24 表示几乎无差异(无语言学内容可区分),
        L24 聚类产生的候选本质上是同一片区域的重复采样。
        但静音帧的 L6 有声学差异(元音衰减尾 vs 爆发音闭塞 vs 呼吸),
        L6 聚类能捕获这些差异, 给 vocoder 更多样的静音输入。

  非静音音素: 逻辑完全不变, 与原版逐行一致。
"""
import os
os.environ["OMP_NUM_THREADS"] = "16"

import gc
import ctypes
import json
import argparse
import h5py
import torch
import numpy as np
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans, KMeans
from tqdm import tqdm

CKPT_DIR = Path("/root/autodl-tmp/anon_test/checkpoints")

BUILD_SILENCE_BANK = True

SILENCE_PHONES = (0, 1)
# 静音桶参数 (可通过命令行覆盖)
SILENCE_CLUSTERS = 4         # 一次聚类数 (L6空间, 不需要太多)
SILENCE_FRAMES_PER_CLUSTER = 40  # 每簇选帧数 (多选以支撑二次聚类)
SILENCE_SUB_CLUSTERS = 8     # 二次聚类子簇数


def _cluster_entropy(l24_frames: np.ndarray, spk_ids: np.ndarray,
                     temperature: float = 5.0) -> float:
    """簇的说话人混淆熵 (归一化到 [0,1])。原版逻辑, 未改动。"""
    unique_spks = np.unique(spk_ids)
    if len(unique_spks) < 2:
        return 0.0
    spk_embs = np.stack([l24_frames[spk_ids == s].mean(0) for s in unique_spks])
    spk_embs = spk_embs / (np.linalg.norm(spk_embs, axis=1, keepdims=True) + 1e-8)
    frames_norm = l24_frames / (np.linalg.norm(l24_frames, axis=1, keepdims=True) + 1e-8)
    sims = frames_norm @ spk_embs.T * temperature
    sims -= sims.max(axis=1, keepdims=True)
    exp_s = np.exp(sims)
    probs = exp_s / exp_s.sum(axis=1, keepdims=True)
    ent = -np.sum(probs * np.log(probs + 1e-9), axis=1)
    return float(ent.mean() / np.log(len(unique_spks)))


def _build_centroid_phone(l6_all, l24_all, spk_all, clusters, frames_per_cluster,
                          sub_clusters, min_spk_diversity, entropy_top_p):
    """
    非静音音素建桶 —— 与原版完全一致, 未改动任何一行。
    """
    n, k = len(l24_all), min(clusters, len(l24_all))
    if n < k:
        return (l6_all.mean(0, keepdims=True).astype(np.float32),
                l24_all.mean(0, keepdims=True).astype(np.float32))

    km1 = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=2048)
    km1.fit(l24_all)

    valid_clusters, entropies = [], []
    for c in range(k):
        cm = km1.labels_ == c
        if not np.any(cm):
            continue
        if len(np.unique(spk_all[cm])) < min_spk_diversity:
            continue
        entropies.append(_cluster_entropy(l24_all[cm], spk_all[cm]))
        valid_clusters.append(c)

    if not valid_clusters:
        return None, None

    centroids_l6, centroids_l24 = [], []
    threshold = np.percentile(entropies, (1 - entropy_top_p) * 100)
    for c, ent in zip(valid_clusters, entropies):
        if ent < threshold:
            continue
        cm = km1.labels_ == c
        dists = np.linalg.norm(l24_all[cm] - km1.cluster_centers_[c], axis=1)
        n_sel = min(frames_per_cluster, int(cm.sum()))
        top_local = dists.argsort()[:n_sel]
        cluster_global = np.where(cm)[0][top_local]
        cand_l6 = l6_all[cluster_global]
        cand_l24 = l24_all[cluster_global]

        k2 = min(sub_clusters, len(cand_l24))
        if len(cand_l24) <= k2:
            centroids_l24.append(cand_l24)
            centroids_l6.append(cand_l6)
        else:
            km2 = KMeans(n_clusters=k2, random_state=42, n_init='auto')
            km2.fit(cand_l24)
            centroids_l24.append(km2.cluster_centers_.astype(np.float32))
            centroids_l6.append(np.stack([
                cand_l6[km2.labels_ == sc].mean(0)
                for sc in range(k2) if np.any(km2.labels_ == sc)
            ]))

    if not centroids_l24:
        return None, None
    return np.concatenate(centroids_l6), np.concatenate(centroids_l24)


def _build_silence_phone(l6_all, l24_all, sil_clusters, sil_frames, sil_sub_clusters):
    """
    静音音素建桶 —— 本文件唯一改动点。
    
    原版 (L24):
      L24 KMeans → L24距离选帧 → 存原始帧
      问题: 静音帧L24几乎无差异, 16簇×8帧=128个近似相同的候选
    
    本版 (L6):
      L6 KMeans → L6距离选帧 → L6二次聚类 → 存L6质心
      改进: L6捕获声学微结构差异(衰减/呼吸/闭塞)
            质心保持多说话人平均(身份稀释)
            给vocoder更多样的静音输入
    """
    n = len(l6_all)
    k1 = min(sil_clusters, n)

    if n <= sil_sub_clusters:
        return (l6_all.mean(0, keepdims=True).astype(np.float32),
                l24_all.mean(0, keepdims=True).astype(np.float32))

    # ① 一次聚类: L6 空间 (原版用 L24)
    km1 = MiniBatchKMeans(n_clusters=k1, random_state=42, batch_size=2048)
    km1.fit(l6_all)

    centroids_l6, centroids_l24 = [], []
    for c in range(k1):
        cm = km1.labels_ == c
        if cm.sum() < 2:
            continue

        # ② 选帧: L6 距离 (原版用 L24 距离)
        dists = np.linalg.norm(l6_all[cm] - km1.cluster_centers_[c], axis=1)
        n_sel = min(sil_frames, int(cm.sum()))
        top_local = dists.argsort()[:n_sel]
        cluster_global = np.where(cm)[0][top_local]
        cand_l6 = l6_all[cluster_global]
        cand_l24 = l24_all[cluster_global]

        # ③ 二次聚类: L6 空间 (原版无二次聚类, 直接存原始帧)
        k2 = min(sil_sub_clusters, len(cand_l6))
        if len(cand_l6) <= k2:
            centroids_l6.append(cand_l6)
            centroids_l24.append(cand_l24)
        else:
            km2 = KMeans(n_clusters=k2, random_state=42, n_init='auto')
            km2.fit(cand_l6)    # ← L6 聚类 (质心是L6空间的真正聚类中心)
            centroids_l6.append(km2.cluster_centers_.astype(np.float32))
            centroids_l24.append(np.stack([
                cand_l24[km2.labels_ == sc].mean(0)
                for sc in range(k2) if np.any(km2.labels_ == sc)
            ]))

    if not centroids_l6:
        return None, None
    return np.concatenate(centroids_l6), np.concatenate(centroids_l24)


def _malloc_trim():
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def build_bank(data_dir, pool_id, gender_tag, clusters=8, frames_per_cluster=20,
               sub_clusters=4, min_spk_diversity=3, entropy_top_p=0.5,
               chunk_size=200_000,
               sil_clusters=SILENCE_CLUSTERS, sil_frames=SILENCE_FRAMES_PER_CLUSTER,
               sil_sub_clusters=SILENCE_SUB_CLUSTERS,
               banks_dir=None):
    data_dir = Path(data_dir)
    if banks_dir is None:
        banks_dir = CKPT_DIR / "banks_v2_centroid_silL6_20spk"
    banks_dir.mkdir(parents=True, exist_ok=True)
    out_path = banks_dir / f"pool_{pool_id}_gender-{gender_tag}.pt"

    with open(data_dir / "metadata.json") as f:
        meta = json.load(f)
    total_frames = meta["total_frames"]
    utts = meta["utterances"]

    spk2id = {u["speaker_id"]: i for i, u in enumerate(
        {u["speaker_id"]: u for u in utts}.values())}
    frame_to_spk = np.full(total_frames, -1, dtype=np.int32)
    utt_mask = np.zeros(total_frames, dtype=bool)
    for u in utts:
        s, e = u["h5_start_idx"], u["h5_end_idx"]
        frame_to_spk[s:e] = spk2id[u["speaker_id"]]
        utt_mask[s:e] = True

    with h5py.File(data_dir / "phones.h5") as f:
        all_phones = f["phones"][:]

    is_silence = np.isin(all_phones, SILENCE_PHONES)
    keep_mask = utt_mask & (~is_silence)
    sil_mask = utt_mask & is_silence

    unique_phones = np.unique(all_phones[keep_mask])
    print(f"  有效帧(非静音): {keep_mask.sum()}, 音素数: {len(unique_phones)}")
    if BUILD_SILENCE_BANK:
        print(f"  静音帧: {sil_mask.sum()} (L6聚类建桶)")

    # ---- 分桶 ----
    l6_buckets = {ph: [] for ph in unique_phones}
    l24_buckets = {ph: [] for ph in unique_phones}
    spk_buckets = {ph: [] for ph in unique_phones}
    sil_l6_buckets = {ph: [] for ph in SILENCE_PHONES}
    sil_l24_buckets = {ph: [] for ph in SILENCE_PHONES}

    with h5py.File(data_dir / "layer_6.h5") as h6, \
         h5py.File(data_dir / "layer_24.h5") as h24:
        ds6, ds24 = h6["features"], h24["features"]
        for start in tqdm(range(0, total_frames, chunk_size),
                          desc="分桶", leave=False):
            end = min(start + chunk_size, total_frames)

            m = keep_mask[start:end]
            if np.any(m):
                l6 = ds6[start:end][m]
                l24 = ds24[start:end][m]
                phones_c = all_phones[start:end][m]
                spks_c = frame_to_spk[start:end][m]
                for ph in np.unique(phones_c):
                    pm = phones_c == ph
                    l6_buckets[ph].append(l6[pm])
                    l24_buckets[ph].append(l24[pm])
                    spk_buckets[ph].append(spks_c[pm])

            if BUILD_SILENCE_BANK:
                sm = sil_mask[start:end]
                if np.any(sm):
                    sl6 = ds6[start:end][sm]
                    sl24 = ds24[start:end][sm]
                    sphones_c = all_phones[start:end][sm]
                    for ph in np.unique(sphones_c):
                        pm = sphones_c == ph
                        sil_l6_buckets[ph].append(sl6[pm])
                        sil_l24_buckets[ph].append(sl24[pm])

    bank = {}

    # ---- 非静音音素建桶 (原版质心逻辑, 未改动) ----
    for ph in tqdm(unique_phones, desc="聚类(质心)", leave=False):
        if not l24_buckets[ph]:
            continue
        l6_all = np.concatenate(l6_buckets[ph])
        l24_all = np.concatenate(l24_buckets[ph])
        spk_all = np.concatenate(spk_buckets[ph])
        del l6_buckets[ph], l24_buckets[ph], spk_buckets[ph]

        c_l6, c_l24 = _build_centroid_phone(
            l6_all, l24_all, spk_all, clusters, frames_per_cluster,
            sub_clusters, min_spk_diversity, entropy_top_p)
        if c_l24 is not None:
            bank[int(ph)] = {
                "l6": torch.from_numpy(c_l6).float(),
                "l24": torch.from_numpy(c_l24).float(),
            }
        del l6_all, l24_all, spk_all

    # ---- 静音音素建桶 (L6聚类 — 本文件唯一改动) ----
    if BUILD_SILENCE_BANK:
        for ph in SILENCE_PHONES:
            if not sil_l24_buckets[ph]:
                continue
            sl6_all = np.concatenate(sil_l6_buckets[ph])
            sl24_all = np.concatenate(sil_l24_buckets[ph])
            del sil_l6_buckets[ph], sil_l24_buckets[ph]

            result = _build_silence_phone(
                sl6_all, sl24_all, sil_clusters, sil_frames, sil_sub_clusters)
            if result is not None:
                pool_l6, pool_l24 = result
                bank[int(ph)] = {
                    "l6": torch.from_numpy(pool_l6).float(),
                    "l24": torch.from_numpy(pool_l24).float(),
                }
            del sl6_all, sl24_all

    del l6_buckets, l24_buckets, spk_buckets
    del sil_l6_buckets, sil_l24_buckets

    torch.save(bank, out_path)
    total_f = sum(v["l6"].shape[0] for v in bank.values())
    sil_info = ""
    if BUILD_SILENCE_BANK:
        sil_cnt = sum(bank[p]["l6"].shape[0] for p in SILENCE_PHONES if p in bank)
        sil_info = f", 静音质心:{sil_cnt}"
    print(f"  Bank 完成: {out_path.relative_to(CKPT_DIR)} | "
          f"音素:{len(bank)}, 总质心:{total_f}{sil_info}")

    del bank
    gc.collect()
    _malloc_trim()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=int, default=None)
    parser.add_argument("--clusters", "-c", type=int, default=8)
    parser.add_argument("--frames-per-cluster", "-f", type=int, default=20)
    parser.add_argument("--sub-clusters", "-s", type=int, default=4)
    parser.add_argument("--min-spk-diversity", "-d", type=int, default=3)
    parser.add_argument("--entropy-top-p", "-e", type=float, default=0.7)
    # 静音帧专用参数
    parser.add_argument("--sil-clusters", type=int, default=SILENCE_CLUSTERS,
                        help="静音帧一次聚类数 (L6空间)")
    parser.add_argument("--sil-frames", type=int, default=SILENCE_FRAMES_PER_CLUSTER,
                        help="静音帧每簇选帧数")
    parser.add_argument("--sil-sub-clusters", type=int, default=SILENCE_SUB_CLUSTERS,
                        help="静音帧二次聚类子簇数")
    parser.add_argument("--bank-dir", type=str, default=None)
    args = parser.parse_args()

    if args.bank_dir:
        banks_dir = CKPT_DIR / args.bank_dir
    else:
        e_str = f"{args.entropy_top_p:.1f}".replace(".", "")
        dir_name = (f"banks_c{args.clusters}f{args.frames_per_cluster}"
                    f"s{args.sub_clusters}_e{e_str}_d{args.min_spk_diversity}"
                    f"_centroid_silL6_20spk")
        banks_dir = CKPT_DIR / dir_name

    print(f"Bank 输出目录: {banks_dir}")
    print(f"非静音: 原版质心逻辑 (L24 KMeans → 熵过滤 → L24 二次聚类)")
    print(f"静音:   L6聚类 (clusters={args.sil_clusters}, "
          f"frames={args.sil_frames}, sub_clusters={args.sil_sub_clusters})")

    pool_ids = [args.pool] if args.pool is not None else range(20)
    for pid in pool_ids:
        print(f"\n=== Pool {pid} ===")
        for gender in ["m", "f"]:
            data_dir = CKPT_DIR / f"pool20spk_{pid}" / gender
            if not (data_dir / "metadata.json").exists():
                print(f"  {gender}: 无数据，跳过")
                continue
            print(f"  [{gender}]")
            build_bank(data_dir, pid, gender,
                       clusters=args.clusters,
                       frames_per_cluster=args.frames_per_cluster,
                       sub_clusters=args.sub_clusters,
                       min_spk_diversity=args.min_spk_diversity,
                       entropy_top_p=args.entropy_top_p,
                       sil_clusters=args.sil_clusters,
                       sil_frames=args.sil_frames,
                       sil_sub_clusters=args.sil_sub_clusters,
                       banks_dir=banks_dir)
            gc.collect()
            _malloc_trim()


if __name__ == "__main__":
    main()