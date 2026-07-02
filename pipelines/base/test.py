#!/usr/bin/env python3
"""
Bank 构建 v2-centroid + 静音桶

【用途】
  复现「旧强隐私 bank」的质心构建逻辑, 但补上静音桶, 用于单变量验证:
    对比组1: c8f20s4_e05_d3 质心, 无静音桶  (旧强隐私 bank, 你已有)
    对比组2: c8f20s4_e05_d3 质心, +静音桶   (本脚本产出)
  两者唯一变量 = 静音桶。full-eval 的半知情 EER 差 = 「静音 fallback 假隐私」的量。

【与原 build_multi_bank_v2.py 的差异】
  - 非静音音素: 逻辑完全不变 (二次 KMeans 存 L6 均值 + L24 质心)。
                这是「旧强隐私 bank」要复现的部分, 一个字都没动。
  - 静音音素 (0,1): 新增。用与 build_multi_bank_v3.py 完全相同的静音桶逻辑
                    (一层 KMeans + 小帧池, 无熵过滤)。
                    → 注意: 静音桶存的是真实帧, 非静音存的是合成质心,
                      同一 bank 内两种候选集类型并存 (不同处理路径, 逻辑无冲突)。

【实验开关】
  BUILD_SILENCE_BANK = True  → 产出 +静音桶 版本 (对比组2)
  BUILD_SILENCE_BANK = False → 退化为原版无静音桶 (= 旧强隐私 bank, 可用于复现校验)

【配套推理端】
  pipeline 的 _retrieve 需已删除 `if ph in {0,1} and silence_noise_sigma>0` 分支,
  静音帧走正常检索。_load_bank 里 key 强制 int()。
  (与 B/C 实验用的是同一个 pipeline, 不需要再改)
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

# ==================== 实验开关 ====================
# True  → 质心 + 静音桶 (对比组2)
# False → 质心, 无静音桶 (= 原版旧强隐私 bank)
BUILD_SILENCE_BANK = True

# 静音桶构建参数 —— 必须与 build_multi_bank_v3.py 中的值完全一致,
# 否则与 B/C 实验的静音桶不可比, 验证不干净。
SILENCE_PHONES = (0, 1)
SILENCE_CLUSTERS = 16
SILENCE_FRAMES_PER_CLUSTER = 8
# ==================================================


def _malloc_trim():
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _cluster_entropy(l24_frames: np.ndarray, spk_ids: np.ndarray,
                     temperature: float = 5.0) -> float:
    """簇的说话人混淆熵 (归一化到 [0,1])。用 L24 计算。原版逻辑, 未改动。"""
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
    非静音音素建桶 —— 原 build_multi_bank_v2.py 的质心逻辑, 逐行保留, 未改动。
      ① 一次 KMeans on L24, k=clusters
      ② d 过滤 + 计算每簇熵
      ③ e 过滤 (entropy_top_p)
      ④ 每簇取距质心最近 frames_per_cluster 帧 → 二次 KMeans(sub_clusters)
      ⑤ 保存: L24 质心 + 对应子簇 L6 均值 (合成质心)
    返回 (l6, l24); 无有效簇返回 (None, None)
    """
    n, k = len(l24_all), min(clusters, len(l24_all))
    if n < k:
        # 帧数不足: 存均值 (原版逻辑)
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


def _build_silence_phone(l6_all, l24_all, clusters, frames_per_cluster):
    """
    静音音素建桶 —— 与 build_multi_bank_v3.py 的 _build_silence_phone 完全一致。
    一层 KMeans + 小帧池, 无熵过滤、无二次聚类。存真实帧。
    保持与 B/C 实验静音桶逐行一致, 确保验证不引入新变量。
    """
    n_frames = len(l24_all)
    k = min(clusters, n_frames)
    if n_frames <= k:
        return l6_all, l24_all

    km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=2048)
    km.fit(l24_all)

    pool_l6, pool_l24 = [], []
    for c in range(k):
        cm = km.labels_ == c
        if not np.any(cm):
            continue
        c_l24 = l24_all[cm]
        c_l6 = l6_all[cm]
        d = np.linalg.norm(c_l24 - km.cluster_centers_[c], axis=1)
        n_keep = min(frames_per_cluster, len(c_l24))
        keep = d.argsort()[:n_keep]
        pool_l24.append(c_l24[keep])
        pool_l6.append(c_l6[keep])

    return np.concatenate(pool_l6), np.concatenate(pool_l24)


def build_bank(data_dir, pool_id, gender_tag, clusters=8, frames_per_cluster=20,
               sub_clusters=4, min_spk_diversity=3, entropy_top_p=0.5,
               chunk_size=200_000, banks_dir=None):
    data_dir = Path(data_dir)
    if banks_dir is None:
        banks_dir = CKPT_DIR / "banks_v2_centroid_sil"
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
    keep_mask = utt_mask & (~is_silence)   # 非静音音素帧
    sil_mask = utt_mask & is_silence       # 静音音素帧

    unique_phones = np.unique(all_phones[keep_mask])
    print(f"  有效帧(非静音): {keep_mask.sum()}, 音素数: {len(unique_phones)}")
    if BUILD_SILENCE_BANK:
        print(f"  静音帧: {sil_mask.sum()} (将单独建桶)")

    # ---- 分桶: 非静音 + 静音 ----
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

            # 非静音音素
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

            # 静音音素
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

    # ---- 非静音音素建桶 (原版质心逻辑) ----
    for ph in tqdm(unique_phones, desc="聚类(质心)", leave=False):
        if not l24_buckets[ph]:
            del l6_buckets[ph], l24_buckets[ph], spk_buckets[ph]
            continue
        l6_all = np.concatenate(l6_buckets[ph])
        l24_all = np.concatenate(l24_buckets[ph])
        spk_all = np.concatenate(spk_buckets[ph])
        del l6_buckets[ph], l24_buckets[ph], spk_buckets[ph]

        c_l6, c_l24 = _build_centroid_phone(
            l6_all, l24_all, spk_all, clusters, frames_per_cluster,
            sub_clusters, min_spk_diversity, entropy_top_p)
        del l6_all, l24_all, spk_all

        if c_l24 is not None:
            bank[int(ph)] = {
                "l6": torch.from_numpy(c_l6).float(),
                "l24": torch.from_numpy(c_l24).float(),
            }

    # ---- 静音音素建桶 (与 v3 一致的帧池逻辑) ----
    if BUILD_SILENCE_BANK:
        for ph in SILENCE_PHONES:
            if not sil_l24_buckets[ph]:
                continue
            sl6_all = np.concatenate(sil_l6_buckets[ph])
            sl24_all = np.concatenate(sil_l24_buckets[ph])
            del sil_l6_buckets[ph], sil_l24_buckets[ph]

            pool_l6, pool_l24 = _build_silence_phone(
                sl6_all, sl24_all, SILENCE_CLUSTERS, SILENCE_FRAMES_PER_CLUSTER)
            del sl6_all, sl24_all

            bank[int(ph)] = {
                "l6": torch.from_numpy(pool_l6).float(),
                "l24": torch.from_numpy(pool_l24).float(),
            }

    del l6_buckets, l24_buckets, spk_buckets
    del sil_l6_buckets, sil_l24_buckets

    torch.save(bank, out_path)
    total_f = sum(v["l6"].shape[0] for v in bank.values())
    sil_info = ""
    if BUILD_SILENCE_BANK:
        sil_cnt = sum(bank[p]["l6"].shape[0] for p in SILENCE_PHONES if p in bank)
        sil_info = f", 静音帧:{sil_cnt}"
    print(f"  Bank 完成: {out_path.relative_to(CKPT_DIR)} | "
          f"音素:{len(bank)}, 非静音质心+静音帧总数:{total_f}{sil_info}")

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
    parser.add_argument("--entropy-top-p", "-e", type=float, default=0.5)
    parser.add_argument("--bank-dir", type=str, default=None,
                        help="自定义输出目录名; 默认自动生成")
    args = parser.parse_args()

    if args.bank_dir:
        banks_dir = CKPT_DIR / args.bank_dir
    else:
        e_str = f"{args.entropy_top_p:.1f}".replace(".", "")
        # _sil / _nosil 后缀, 与无静音桶版本区分
        suffix = "_sil" if BUILD_SILENCE_BANK else "_nosil"
        dir_name = (f"banks_c{args.clusters}f{args.frames_per_cluster}"
                    f"s{args.sub_clusters}_e{e_str}_d{args.min_spk_diversity}"
                    f"_centroid{suffix}_20spk")
        banks_dir = CKPT_DIR / dir_name

    print(f"Bank 输出目录: {banks_dir}")
    print(f"静音建桶 (BUILD_SILENCE_BANK): {BUILD_SILENCE_BANK}")
    print(f"参数: c={args.clusters}, f={args.frames_per_cluster}, "
          f"s={args.sub_clusters}, d={args.min_spk_diversity}, e={args.entropy_top_p}")
    if BUILD_SILENCE_BANK:
        print(f"静音桶参数: clusters={SILENCE_CLUSTERS}, "
              f"frames_per_cluster={SILENCE_FRAMES_PER_CLUSTER} "
              f"(须与 build_multi_bank_v3.py 一致)")

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
                       banks_dir=banks_dir)
            gc.collect()
            _malloc_trim()


if __name__ == "__main__":
    main()