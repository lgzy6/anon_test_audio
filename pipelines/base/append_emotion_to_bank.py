#!/usr/bin/env python3
"""
从情感数据集 (ESD英文 + RAVDESS + AudioWAV) 提取特征，
对每个 (gender, phone) 桶走完整的 KMeans→多样性过滤→熵过滤→二次聚类→质心，
复制原始 bank 到 {原目录}_emotionAppend，逐音素 concat 情感质心。
"""
import os
os.environ["OMP_NUM_THREADS"] = "16"

import sys
import shutil
import torch
import numpy as np
import torchaudio
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans, KMeans

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.ssl.wrappers import WavLMSSLExtractor
from models.phone_predictor.predictor import PhonePredictor
from data.datasets.esd import ESDDataset
from data.datasets.ravdess import RAVDESSDataset

WAVLM_CKPT    = "/root/autodl-tmp/anon_test/checkpoints/WavLM-Large.pt"
PHONE_CKPT    = "/root/autodl-tmp/anon_test/checkpoints/phone_decoder.pt"
ESD_ROOT      = "/root/autodl-tmp/datasets/Emotion Speech Dataset"
RAVDESS_ROOT  = "/root/autodl-tmp/datasets/actordata/audio_speech_actors_01-24"
AUDIOWAV_DIR  = "/root/autodl-tmp/datasets/AudioWAV"
BANKS_DIR     = Path("/root/autodl-tmp/anon_test/checkpoints/banks_c8f20s4_e05_d3_centroid_sil")
DEVICE        = "cuda"
LAYERS        = [6, 24]

# 情感专用聚类参数
EMO_CLUSTERS        = 6    # 情感数据量少，不需要 8 簇
FRAMES_PER_CLUSTER  = 20
SUB_CLUSTERS        = 4
MIN_SPK_DIVERSITY   = 2    # 情感说话人少，降低多样性门槛
ENTROPY_TOP_P       = 0.8  # 比 LibriSpeech 的 0.5 宽松，保留更多情感簇


def scan_audiowav(root: str):
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
            "audio_path": str(wav),
        })
    return utts


def load_waveform(utt: dict) -> torch.Tensor:
    if "waveform" in utt:
        w = utt["waveform"]
        return w if isinstance(w, torch.Tensor) else torch.from_numpy(w)
    waveform, sr = torchaudio.load(utt["audio_path"])
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(0, keepdim=True)
    return waveform.squeeze(0)


def extract_to_buckets(utts, extractor, phone_predictor):
    """提取特征，按 (gender, phone_id) 分桶，同时记录 speaker_id 用于多样性过滤"""
    buckets = defaultdict(lambda: {"l6": [], "l24": [], "spk": []})
    for utt in tqdm(utts, desc="提取特征"):
        try:
            w = load_waveform(utt).unsqueeze(0).to(DEVICE)
            with torch.inference_mode():
                feats = extractor.forward_multi_layer(w, layers=LAYERS)
                phones = phone_predictor(feats[24][0].unsqueeze(0)).squeeze(0).cpu().numpy()
            l6  = feats[6][0].cpu().numpy()
            l24 = feats[24][0].cpu().numpy()
            gender = utt.get("gender", "unknown")
            spk_id = utt.get("speaker_id", "unknown")
            for ph in np.unique(phones):
                mask = phones == ph
                key = (gender, int(ph))
                buckets[key]["l6"].append(l6[mask])
                buckets[key]["l24"].append(l24[mask])
                buckets[key]["spk"].append(np.full(mask.sum(), spk_id))
        except Exception as e:
            print(f"  跳过 {utt.get('utt_id','?')}: {e}")
        torch.cuda.empty_cache()
    return buckets


def _cluster_entropy(l24_frames, spk_ids, temperature=5.0):
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


def build_emotion_centroids(l6_all, l24_all, spk_all):
    """与 build_multi_bank_v2_sil.py 完全一致的质心逻辑，使用情感专用参数"""
    n = len(l24_all)
    k = min(EMO_CLUSTERS, n)
    if n <= k:
        return l6_all.mean(0, keepdims=True), l24_all.mean(0, keepdims=True)

    km1 = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=2048)
    km1.fit(l24_all)

    valid_clusters, entropies = [], []
    for c in range(k):
        cm = km1.labels_ == c
        if not np.any(cm):
            continue
        if len(np.unique(spk_all[cm])) < MIN_SPK_DIVERSITY:
            continue
        entropies.append(_cluster_entropy(l24_all[cm], spk_all[cm]))
        valid_clusters.append(c)

    if not valid_clusters:
        return None, None

    threshold = np.percentile(entropies, (1 - ENTROPY_TOP_P) * 100)
    centroids_l6, centroids_l24 = [], []
    for c, ent in zip(valid_clusters, entropies):
        if ent < threshold:
            continue
        cm = km1.labels_ == c
        dists = np.linalg.norm(l24_all[cm] - km1.cluster_centers_[c], axis=1)
        n_sel = min(FRAMES_PER_CLUSTER, int(cm.sum()))
        top_idx = np.where(cm)[0][dists.argsort()[:n_sel]]
        cand_l6, cand_l24 = l6_all[top_idx], l24_all[top_idx]

        k2 = min(SUB_CLUSTERS, len(cand_l24))
        if len(cand_l24) <= k2:
            centroids_l24.append(cand_l24)
            centroids_l6.append(cand_l6)
        else:
            km2 = KMeans(n_clusters=k2, random_state=42, n_init="auto")
            km2.fit(cand_l24)
            centroids_l24.append(km2.cluster_centers_.astype(np.float32))
            centroids_l6.append(np.stack([
                cand_l6[km2.labels_ == sc].mean(0)
                for sc in range(k2) if np.any(km2.labels_ == sc)
            ]))

    if not centroids_l24:
        return None, None
    return np.concatenate(centroids_l6), np.concatenate(centroids_l24)


def merge_into_banks(buckets, src_dir: Path):
    dst_dir = src_dir.parent / (src_dir.name + "_emotionAppend")
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir)
    print(f"已复制 bank 到: {dst_dir}")

    # 先对所有桶做质心化
    print("质心化情感特征...")
    emo_centroids = {}  # (gender, ph) -> {l6, l24}
    for (gender, ph), data in tqdm(buckets.items(), desc="质心化"):
        l6_all  = np.concatenate(data["l6"])
        l24_all = np.concatenate(data["l24"])
        spk_all = np.concatenate(data["spk"])
        c_l6, c_l24 = build_emotion_centroids(l6_all, l24_all, spk_all)
        if c_l24 is not None:
            emo_centroids[(gender, ph)] = {
                "l6":  torch.from_numpy(c_l6).float(),
                "l24": torch.from_numpy(c_l24).float(),
            }

    # 按 gender 聚合后写入 bank
    by_gender = defaultdict(dict)
    for (gender, ph), v in emo_centroids.items():
        by_gender[gender][ph] = v

    for bank_file in sorted(dst_dir.glob("*.pt")):
        gender = "m" if "gender-m" in bank_file.name else "f"
        if gender not in by_gender:
            continue
        bank = torch.load(bank_file, map_location="cpu")
        added_phones, added_frames = 0, 0
        for ph, v in by_gender[gender].items():
            orig_n = bank[ph]["l6"].shape[0] if ph in bank else 0
            if ph in bank:
                bank[ph]["l6"]  = torch.cat([bank[ph]["l6"],  v["l6"]],  dim=0)
                bank[ph]["l24"] = torch.cat([bank[ph]["l24"], v["l24"]], dim=0)
            else:
                bank[ph] = v
            emo_n = v["l6"].shape[0]
            total_n = orig_n + emo_n
            emo_pct = emo_n / total_n * 100
            print(f"    phone {ph:2d}: {orig_n}+{emo_n}={total_n} (情感占比 {emo_pct:.0f}%)")
            added_phones += 1
            added_frames += emo_n
        torch.save(bank, bank_file)
        print(f"  {bank_file.name}: +{added_frames} 情感质心帧 across {added_phones} phones")

    return dst_dir


def main():
    print("收集情感数据...")
    esd = ESDDataset(ESD_ROOT, language="en")
    ravdess = RAVDESSDataset(RAVDESS_ROOT)
    audiowav = scan_audiowav(AUDIOWAV_DIR)
    print(f"AudioWAV: {len(audiowav)} utterances")

    all_utts = []
    for ds, name in [(esd, "esd"), (ravdess, "ravdess")]:
        for i in range(len(ds)):
            item = ds[i]
            item.setdefault("dataset", name)
            all_utts.append(item)
    all_utts.extend(audiowav)
    print(f"总计: {len(all_utts)} utterances")

    print("\n加载模型...")
    extractor = WavLMSSLExtractor(WAVLM_CKPT, layer=6, device=DEVICE)
    phone_predictor = PhonePredictor.load(PHONE_CKPT, device=DEVICE)

    print("\n提取特征并按 (gender, phone) 分桶...")
    buckets = extract_to_buckets(all_utts, extractor, phone_predictor)

    print("\n合并情感质心到 bank...")
    dst = merge_into_banks(buckets, BANKS_DIR)
    print(f"\n完成！新 bank 位于: {dst}")


if __name__ == "__main__":
    main()
