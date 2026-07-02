#!/usr/bin/env python3
"""
Bank 构建 v3：参数压缩 + 静音帧建桶 + medoid/帧池双模式

【本脚本能产出 B / C 两个实验所需的 bank】
  C 实验 (帧池 + 静音桶)   : --medoid-mode 关闭 (默认)
                            每个二次子簇取 frames_per_subcluster 帧 → 真实帧池
  B 实验 (medoid + 静音桶) : --medoid-mode 开启
                            每个二次子簇取 1 帧 (距质心最近的真实帧 = medoid)

  B 和 C 共享完全相同的: 分桶 / 一次 KMeans / d,e 过滤 / 二次 KMeans 簇结构 / 静音桶。
  唯一区别 = 二次子簇出 1 帧 还是 出多帧。
  因此 B→C 的指标差 = 纯粹的「medoid vs 帧池 (流形性/基数)」贡献，归因干净。

  对照实验 A (medoid, 无静音桶) 用旧的 build_multi_bank_v2_medoid.py 已有产物，
  或本脚本 --medoid-mode 开启 + BUILD_SILENCE_BANK=False 也可复现。

【参数】
  - c (clusters)            : 音素内变体分辨率。一次 KMeans 簇数。结构参数, 固定 8。
  - n (n_per_cluster)       : 二次 KMeans 簇数 (= 每个一次簇产出的子簇数)。
                              C 模式下: 每子簇出 frames_per_subcluster 帧。
                              B 模式下: 每子簇出 1 帧。
                              B 和 C 必须用相同的 n, 才能保证簇结构一致。
  - p (frames_per_subcluster): 仅 C 模式生效。每个二次子簇保留的帧数。
  - d (min_spk_diversity)   : 卫生筛选。簇内最少说话人数。固定 5。
  - e (entropy_top_p)       : 历史参数。设 1.0 使其失效。

【实验顺序 (变量隔离)】
  Step1  跑 A 和 C 的 pre-eval, 看 A→C 的 WER 总效果
         A→C 有改善 → 补 B, 进 Step2
         A→C 无改善 → 不必补 B, 「真实帧改 WER」假设存疑
  Step2  补 B (本脚本 --medoid-mode + BUILD_SILENCE_BANK=True), 三段 full-eval
         A→B 差 = 静音建桶贡献
         B→C 差 = 帧池化/流形性贡献 (检验核心假设)

【配套推理端改动 (不在本文件)】
  pipeline 的 _retrieve 中删掉 `if ph in {0,1} and silence_noise_sigma>0` 分支,
  静音帧走正常检索 (此时有静音桶)。_load_bank 里 key 强制 int()。
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
# B 实验 / C 实验: True  (加静音桶)
# A 实验复现:        False (无静音桶)
BUILD_SILENCE_BANK = True

# 静音桶构建参数 (静音帧独立路径, 与 c/n 无关)
SILENCE_PHONES = (0, 1)
SILENCE_CLUSTERS = 16
SILENCE_FRAMES_PER_CLUSTER = 8
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


def _subcluster_extract(sc_l6, sc_l24, centroid, medoid_mode, frames_per_subcluster):
    """
    从一个二次子簇中取帧。
      medoid_mode=True : 取 1 帧 (距质心最近 = medoid)        → B 实验
      medoid_mode=False: 取 frames_per_subcluster 帧 (最近的) → C 实验
    返回 (l6_sel, l24_sel)
    """
    d = np.linalg.norm(sc_l24 - centroid, axis=1)
    if medoid_mode:
        idx = d.argmin()
        return sc_l6[idx:idx + 1], sc_l24[idx:idx + 1]
    else:
        n_keep = min(frames_per_subcluster, len(sc_l24))
        keep = d.argsort()[:n_keep]
        return sc_l6[keep], sc_l24[keep]


def _build_normal_phone(l6_all, l24_all, spk_all, clusters, n_per_cluster,
                        min_spk_diversity, entropy_top_p,
                        medoid_mode, frames_per_subcluster):
    """
    非静音音素建桶。B / C 模式共享此函数, 仅 _subcluster_extract 行为不同。
      ① 一次 KMeans on L24, k=clusters
      ② d 过滤 (卫生筛选) + 计算每簇熵
      ③ e 过滤 (entropy_top_p=1.0 时不过滤)
      ④ 每个通过的簇: 二次 KMeans(k=n) on 完整一次簇
         → B: 每子簇取 1 medoid   C: 每子簇取多帧
    """
    n_frames, k = len(l24_all), min(clusters, len(l24_all))
    if n_frames < k:
        return l6_all, l24_all  # 帧数过少, 全保留

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

    pool_l6, pool_l24 = [], []
    threshold = np.percentile(entropies, (1 - entropy_top_p) * 100)
    for c, ent in zip(valid_clusters, entropies):
        if ent < threshold:
            continue
        cm = km1.labels_ == c
        cluster_l6 = l6_all[cm]
        cluster_l24 = l24_all[cm]
        N = min(n_per_cluster, len(cluster_l24))

        if len(cluster_l24) <= N:
            # 簇本身不够 N 帧
            if medoid_mode:
                # B 模式: 仍只能取 1 帧, 取距簇均值最近的
                cen = cluster_l24.mean(0)
                idx = np.linalg.norm(cluster_l24 - cen, axis=1).argmin()
                pool_l6.append(cluster_l6[idx:idx + 1])
                pool_l24.append(cluster_l24[idx:idx + 1])
            else:
                pool_l6.append(cluster_l6)
                pool_l24.append(cluster_l24)
        else:
            # 二次 KMeans k=N on 完整一次簇 (含簇边缘过渡帧)
            km2 = MiniBatchKMeans(n_clusters=N, random_state=42, batch_size=2048)
            km2.fit(cluster_l24)
            for sc in range(N):
                sc_mask = km2.labels_ == sc
                if not np.any(sc_mask):
                    continue
                l6_sel, l24_sel = _subcluster_extract(
                    cluster_l6[sc_mask], cluster_l24[sc_mask],
                    km2.cluster_centers_[sc],
                    medoid_mode, frames_per_subcluster)
                pool_l6.append(l6_sel)
                pool_l24.append(l24_sel)

    if not pool_l24:
        return None, None
    return np.concatenate(pool_l6), np.concatenate(pool_l24)


def _build_silence_phone(l6_all, l24_all, clusters, frames_per_cluster):
    """
    静音音素建桶。一层 KMeans + 小帧池, 无熵过滤、无二次聚类。
    注意: 静音桶在 B 和 C 中用完全相同的构建方式 (都是小帧池),
    这样 B→C 的差里不会混入静音桶的差异。
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
               frames_per_subcluster=8, min_spk_diversity=5, entropy_top_p=1.0,
               medoid_mode=False, chunk_size=200_000, banks_dir=None):
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
    mode_str = "medoid(B)" if medoid_mode else "帧池(C)"
    print(f"  模式: {mode_str} | 有效帧(非静音): {keep_mask.sum()}, "
          f"音素数: {len(unique_phones)}")
    if BUILD_SILENCE_BANK:
        print(f"  静音帧: {sil_mask.sum()} (将单独建桶)")

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

    # ---- 非静音音素建桶 ----
    for ph in tqdm(unique_phones, desc=f"聚类({mode_str})", leave=False):
        if not l24_buckets[ph]:
            continue
        l6_all = np.concatenate(l6_buckets[ph])
        l24_all = np.concatenate(l24_buckets[ph])
        spk_all = np.concatenate(spk_buckets[ph])
        del l6_buckets[ph], l24_buckets[ph], spk_buckets[ph]

        pool_l6, pool_l24 = _build_normal_phone(
            l6_all, l24_all, spk_all, clusters, n_per_cluster,
            min_spk_diversity, entropy_top_p,
            medoid_mode, frames_per_subcluster)
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
          f"音素:{len(bank)}, 候选总数:{total_f}{sil_info}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=int, default=None)
    parser.add_argument("--clusters", "-c", type=int, default=8,
                        help="音素内变体分辨率 (结构参数, 固定 8)")
    parser.add_argument("--n-per-cluster", "-n", type=int, default=16,
                        help="二次 KMeans 簇数 (B 和 C 必须用相同的 n)")
    parser.add_argument("--frames-per-subcluster", "-p", type=int, default=8,
                        help="仅 C 模式生效: 每个二次子簇保留的帧数")
    parser.add_argument("--min-spk-diversity", "-d", type=int, default=3,
                        help="卫生筛选: 簇内最少说话人数 (固定 3)")
    parser.add_argument("--entropy-top-p", "-e", type=float, default=1.0,
                        help="熵过滤保留比例 (建议 1.0 使其失效)")
    parser.add_argument("--medoid-mode", action="store_true",
                        help="开启 = B 实验 (每子簇 1 medoid); "
                             "关闭 = C 实验 (每子簇多帧帧池)")
    parser.add_argument("--bank-dir", type=str, default=None,
                        help="自定义输出目录名; 默认自动生成")
    args = parser.parse_args()

    if args.bank_dir:
        banks_dir = CKPT_DIR / args.bank_dir
    else:
        e_str = f"{args.entropy_top_p:.1f}".replace(".", "")
        sil_suffix = "_sil" if BUILD_SILENCE_BANK else "_nosil"
        if args.medoid_mode:
            # B 实验: banks_c8n16_e10_d5_medoid_sil
            dir_name = (f"banks_c{args.clusters}n{args.n_per_cluster}"
                        f"_e{e_str}_d{args.min_spk_diversity}_medoid{sil_suffix}")
        else:
            # C 实验: banks_c8n16p8_e10_d5_pool_sil
            dir_name = (f"banks_c{args.clusters}n{args.n_per_cluster}"
                        f"p{args.frames_per_subcluster}"
                        f"_e{e_str}_d{args.min_spk_diversity}_pool{sil_suffix}")
        banks_dir = CKPT_DIR / dir_name

    print(f"Bank 输出目录: {banks_dir}")
    print(f"模式: {'B (medoid + 静音桶)' if args.medoid_mode else 'C (帧池 + 静音桶)'}")
    print(f"静音建桶 (BUILD_SILENCE_BANK): {BUILD_SILENCE_BANK}")
    print(f"参数: c={args.clusters}, n={args.n_per_cluster}, "
          f"p={args.frames_per_subcluster}, d={args.min_spk_diversity}, "
          f"e={args.entropy_top_p}, medoid_mode={args.medoid_mode}")

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
                       frames_per_subcluster=args.frames_per_subcluster,
                       min_spk_diversity=args.min_spk_diversity,
                       entropy_top_p=args.entropy_top_p,
                       medoid_mode=args.medoid_mode,
                       banks_dir=banks_dir)


if __name__ == "__main__":
    main()