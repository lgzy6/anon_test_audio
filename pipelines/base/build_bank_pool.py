#!/usr/bin/env python3
"""
Bank 构建 v3：参数压缩 + 静音帧建桶

【参数压缩】c/f/s/p 四参数 → c + n 两参数
  - c (clusters)        : 音素内变体分辨率。一次 KMeans 把音素声学空间切成几块。
                          结构参数, 固定 8, 不作为调优旋钮。
  - n (n_per_cluster)   : 候选集基数主旋钮。每个通过过滤的一次簇最终贡献 n 帧。
                          这是唯一需要扫描的调优参数 (8/16/32/64)。
  - d (min_spk_diversity): 卫生筛选。删掉单一说话人主导的脏簇。固定 5。
  - e (entropy_top_p)   : 历史参数。设 1.0 使其失效 (不再当隐私主力, 避免伤 WER)。

  旧版 f 粗筛 (距质心最近 f 帧) 已移除 —— 它把候选集预先收窄到簇中心附近,
  丢弃簇边缘的声学过渡帧, 是 WER 偏高的来源之一。现在二次 KMeans 直接在
  完整的一次簇上做, k=n, 每个二次簇取 1 个 medoid (距质心最近的真实帧),
  使 n 帧从整个一次簇均匀采样 (含过渡帧)。

【静音帧建桶】
  旧版 silence_phones 被直接从 keep_mask 剔除 → bank 里根本没有 phone 0/1 桶
  → 推理时静音帧 bank.get(0)=None → 掉进全音素 fallback 检索 (坏默认)。
  v3 给静音音素 (0,1) 单独建桶: 一层 KMeans + 小帧池, 无熵过滤
  (静音没有"说话人混淆度"概念, 声学变异主要是能量+噪声纹理)。

【实验顺序 —— 变量隔离, 一次只动一类】
  第一步: BUILD_SILENCE_BANK = False
          沿用静音剔除状态, 变量 = 帧池化 + 参数清理
          对比对象: medoid 版 → 验证对 WER 的影响
  第二步: BUILD_SILENCE_BANK = True
          加静音桶, 变量 = 静音建桶
          对比对象: 第一步 → 单独验证静音建桶的贡献
  两步的 bank 目录名自带 _nosil / _sil 后缀, 不会互相覆盖。

【配套的推理端改动 (不在本文件)】
  pipeline 的 _retrieve 中:
    - 删掉 `if ph in {0,1} and silence_noise_sigma > 0` 的特殊分支
    - 静音帧走正常检索路径 (此时它有自己的桶了)
    - fallback 保留, 仅作为"某音素完全无数据"的真正兜底
"""
import os
os.environ["OMP_NUM_THREADS"] = "16"

import gc
import json
import argparse
import h5py
import torch
import numpy as np
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

CKPT_DIR = Path("/root/autodl-tmp/anon_test/checkpoints")

# ==================== 实验开关 ====================
# 第一步实验: False (沿用静音剔除, 变量=帧池化+参数清理)
# 第二步实验: True  (加静音桶, 变量=静音建桶)
BUILD_SILENCE_BANK = True

# 静音桶构建参数 (静音帧独立路径, 与 c/n 无关)
SILENCE_PHONES = (0, 1)            # 静音/噪声音素 ID
SILENCE_CLUSTERS = 16              # 静音桶一层 KMeans 簇数
SILENCE_FRAMES_PER_CLUSTER = 8     # 每个静音簇保留的真实帧数
# ==================================================


def _cluster_entropy(l24_frames: np.ndarray, spk_ids: np.ndarray,
                     temperature: float = 5.0) -> float:
    """簇的说话人混淆熵 (归一化到 [0,1])。用 L24 计算。"""
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


def _build_normal_phone(l6_all, l24_all, spk_all, clusters, n_per_cluster,
                        min_spk_diversity, entropy_top_p):
    """
    非静音音素建桶。
      ① 一次 KMeans on L24, k=clusters
      ② d 过滤 (卫生筛选) + 计算每簇熵
      ③ e 过滤 (entropy_top_p=1.0 时不过滤)
      ④ 每个通过的簇: 二次 KMeans(k=n) on 完整一次簇, 每子簇取 1 medoid
    返回: (pool_l6, pool_l24) 真实帧池; 无有效簇时返回 (None, None)
    """
    n_frames, k = len(l24_all), min(clusters, len(l24_all))
    if n_frames < k:
        # 帧数过少: 直接保留全部真实帧
        return l6_all, l24_all

    km1 = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=2048)
    km1.fit(l24_all)

    # ② d 过滤 + 计算熵
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

    pool_l6, pool_l24 = [], []
    # ③ e 过滤 (entropy_top_p=1.0 → threshold 为最小熵 → 全部通过)
    threshold = np.percentile(entropies, (1 - entropy_top_p) * 100)
    for c, ent in zip(valid_clusters, entropies):
        if ent < threshold:
            continue
        cm = km1.labels_ == c
        cluster_l6 = l6_all[cm]
        cluster_l24 = l24_all[cm]
        N = min(n_per_cluster, len(cluster_l24))

        if len(cluster_l24) <= N:
            # 簇本身不够 N 帧: 全保留
            pool_l6.append(cluster_l6)
            pool_l24.append(cluster_l24)
        else:
            # ④ 二次 KMeans k=N on 完整一次簇 → 每子簇取 1 medoid
            #    N 帧从整个一次簇均匀采样 (含簇边缘过渡帧, 不再被 f 粗筛收窄)
            km2 = MiniBatchKMeans(n_clusters=N, random_state=42, batch_size=2048)
            km2.fit(cluster_l24)
            for sc in range(N):
                sc_mask = km2.labels_ == sc
                if not np.any(sc_mask):
                    continue
                sc_l24 = cluster_l24[sc_mask]
                sc_l6 = cluster_l6[sc_mask]
                med = np.linalg.norm(
                    sc_l24 - km2.cluster_centers_[sc], axis=1).argmin()
                pool_l24.append(sc_l24[med:med + 1])
                pool_l6.append(sc_l6[med:med + 1])

    if not pool_l24:
        return None, None
    return np.concatenate(pool_l6), np.concatenate(pool_l24)


def _build_silence_phone(l6_all, l24_all, clusters, frames_per_cluster):
    """
    静音音素建桶。一层 KMeans + 小帧池, 无熵过滤、无二次聚类。
    静音帧无"说话人混淆度"概念, 声学变异主要是能量高低 + 噪声纹理,
    不需要 (也不适用) 非静音路径的 d/e 过滤逻辑。
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


def build_bank(data_dir, pool_id, gender_tag, clusters=8, n_per_cluster=16,
               min_spk_diversity=5, entropy_top_p=1.0, chunk_size=200_000,
               banks_dir=None):
    data_dir = Path(data_dir)
    if banks_dir is None:
        banks_dir = CKPT_DIR / "banks_v3_pool"
    banks_dir.mkdir(parents=True, exist_ok=True)
    out_path = banks_dir / f"pool_{pool_id}_gender-{gender_tag}.pt"

    with open(data_dir / "metadata.json") as f:
        meta = json.load(f)
    total_frames = meta["total_frames"]
    utts = meta["utterances"]

    spk2id = {u["speaker_id"]: i for i, u in enumerate(
        {u["speaker_id"]: u for u in utts}.values())}
    frame_to_spk = np.full(total_frames, -1, dtype=np.int32)
    utt_mask = np.zeros(total_frames, dtype=bool)   # 说话人范围 (不含音素过滤)
    for u in utts:
        s, e = u["h5_start_idx"], u["h5_end_idx"]
        frame_to_spk[s:e] = spk2id[u["speaker_id"]]
        utt_mask[s:e] = True

    with h5py.File(data_dir / "phones.h5") as f:
        all_phones = f["phones"][:]

    is_silence = np.isin(all_phones, SILENCE_PHONES)
    keep_mask = utt_mask & (~is_silence)            # 非静音音素帧
    sil_mask = utt_mask & is_silence                # 静音音素帧

    unique_phones = np.unique(all_phones[keep_mask])
    print(f"  有效帧(非静音): {keep_mask.sum()}, 音素数: {len(unique_phones)}")
    if BUILD_SILENCE_BANK:
        print(f"  静音帧: {sil_mask.sum()} (将单独建桶)")

    # ---- 分桶: 非静音 + 静音 同一遍 chunked 加载 ----
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

            # 静音音素 (仅在开关打开时收集)
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

    # ---- 非静音音素建桶 ----
    for ph in tqdm(unique_phones, desc="聚类(非静音)", leave=False):
        if not l24_buckets[ph]:
            continue
        l6_all = np.concatenate(l6_buckets[ph])
        l24_all = np.concatenate(l24_buckets[ph])
        spk_all = np.concatenate(spk_buckets[ph])
        del l6_buckets[ph], l24_buckets[ph], spk_buckets[ph]

        pool_l6, pool_l24 = _build_normal_phone(
            l6_all, l24_all, spk_all, clusters, n_per_cluster,
            min_spk_diversity, entropy_top_p)
        if pool_l24 is not None:
            bank[int(ph)] = {
                "l6": torch.from_numpy(pool_l6).float(),
                "l24": torch.from_numpy(pool_l24).float(),
            }
        del l6_all, l24_all, spk_all

    # ---- 静音音素建桶 ----
    if BUILD_SILENCE_BANK:
        for ph in SILENCE_PHONES:
            if not sil_l24_buckets[ph]:
                continue
            sl6_all = np.concatenate(sil_l6_buckets[ph])
            sl24_all = np.concatenate(sil_l24_buckets[ph])
            del sil_l6_buckets[ph], sil_l24_buckets[ph]

            pool_l6, pool_l24 = _build_silence_phone(
                sl6_all, sl24_all, SILENCE_CLUSTERS, SILENCE_FRAMES_PER_CLUSTER)
            bank[int(ph)] = {
                "l6": torch.from_numpy(pool_l6).float(),
                "l24": torch.from_numpy(pool_l24).float(),
            }
            del sl6_all, sl24_all

    del l6_buckets, l24_buckets, spk_buckets
    del sil_l6_buckets, sil_l24_buckets
    gc.collect()

    torch.save(bank, out_path)
    total_f = sum(v["l6"].shape[0] for v in bank.values())
    sil_info = ""
    if BUILD_SILENCE_BANK:
        sil_cnt = sum(bank[p]["l6"].shape[0] for p in SILENCE_PHONES if p in bank)
        sil_info = f", 静音帧:{sil_cnt}"
    print(f"  Bank 完成: {out_path.relative_to(CKPT_DIR)} | "
          f"音素:{len(bank)}, 帧池总数:{total_f}{sil_info}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=int, default=None)
    parser.add_argument("--clusters", "-c", type=int, default=8,
                        help="音素内变体分辨率 (结构参数, 建议固定 8)")
    parser.add_argument("--n-per-cluster", "-n", type=int, default=16,
                        help="候选集基数主旋钮 (调优参数, 扫 8/16/32/64)")
    parser.add_argument("--min-spk-diversity", "-d", type=int, default=5,
                        help="卫生筛选: 簇内最少说话人数 (建议固定 5)")
    parser.add_argument("--entropy-top-p", "-e", type=float, default=1.0,
                        help="熵过滤保留比例 (建议 1.0 使其失效)")
    parser.add_argument("--bank-dir", type=str, default=None,
                        help="自定义输出目录名; 默认自动生成")
    args = parser.parse_args()

    if args.bank_dir:
        banks_dir = CKPT_DIR / args.bank_dir
    else:
        e_str = f"{args.entropy_top_p:.1f}".replace(".", "")
        # _sil / _nosil 后缀: 两步实验的 bank 不互相覆盖
        suffix = "_sil" if BUILD_SILENCE_BANK else "_nosil"
        dir_name = (f"banks_c{args.clusters}n{args.n_per_cluster}"
                    f"_e{e_str}_d{args.min_spk_diversity}_pool{suffix}")
        banks_dir = CKPT_DIR / dir_name

    print(f"Bank 输出目录: {banks_dir}")
    print(f"静音建桶 (BUILD_SILENCE_BANK): {BUILD_SILENCE_BANK}")
    print(f"参数: c={args.clusters}, n={args.n_per_cluster}, "
          f"d={args.min_spk_diversity}, e={args.entropy_top_p}")

    pool_ids = [args.pool] if args.pool is not None else range(4)
    for pid in pool_ids:
        print(f"\n=== Pool {pid} ===")
        for gender in ["m", "f"]:
            data_dir = CKPT_DIR / f"pool_{pid}" / gender
            if not (data_dir / "metadata.json").exists():
                print(f"  {gender}: 无数据，跳过")
                continue
            print(f"  [{gender}]")
            build_bank(data_dir, pid, gender,
                       clusters=args.clusters,
                       n_per_cluster=args.n_per_cluster,
                       min_spk_diversity=args.min_spk_diversity,
                       entropy_top_p=args.entropy_top_p,
                       banks_dir=banks_dir)


if __name__ == "__main__":
    main()