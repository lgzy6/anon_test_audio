#!/usr/bin/env python3
"""
Purifier 匿名化合成:
  检索: 提取源音频 WavLM L24 -> purifier 净化 -> 在 purified_l24 bank 中 kNN 检索
  合成: 取检索到的 bank L6 特征 -> HiFi-GAN
"""

import sys
import argparse
import json
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
BASE_DIR = Path(__file__).parent.parent.parent
CKPT_DIR = BASE_DIR / "checkpoints"
DATA_DIR  = CKPT_DIR / "trainother500_200spk"

DEFAULT_AUDIO      = str('/root/autodl-tmp/datasets/LibriSpeech/test-clean/61/70968/61-70968-0000.flac')
DEFAULT_OUTPUT_DIR = str(BASE_DIR / 'outputs' / 'purifier')

from pipelines.train.purifier import FeaturePurifier


def load_purifier(device):
    cfg = json.load(open(CKPT_DIR / "purifier" / "train_config.json"))
    ckpt = torch.load(CKPT_DIR / "purifier" / "best_purifier.pt", map_location=device)
    model = FeaturePurifier(
        input_dim=cfg["input_dim"], hidden_dim=cfg["hidden_dim"],
        num_phones=cfg["num_phones"], num_speakers=ckpt["num_speakers"], dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["full_state_dict"])
    model.eval()
    return model


def load_models(modes, device):
    from models.ssl.wrappers import WavLMSSLExtractor
    from models.phone_predictor.predictor import PhonePredictor
    from models.vocoder.hifigan import HiFiGAN

    models = {
        "wavlm": WavLMSSLExtractor(ckpt_path=str(CKPT_DIR / "WavLM-Large.pt"), layer=6, device=device),
        "phone_predictor": PhonePredictor.load(str(CKPT_DIR / "phone_decoder.pt"), device=device),
        "vocoder": HiFiGAN.load(checkpoint_path=str(CKPT_DIR / "hifigan.pt"), device=device),
        "purifier": load_purifier(device),
    }

    def _load(path):
        bank = torch.load(str(path), map_location=device)
        fallback_l6  = torch.cat([v["l6"]           for v in bank.values()], dim=0)
        fallback_pur = torch.cat([v["purified_l24"]  for v in bank.values()], dim=0)
        return bank, {"l6": fallback_l6, "purified_l24": fallback_pur}

    if any(m in modes for m in ("same", "cross")):
        bm, fm = _load(DATA_DIR / "pseudo_bank_purifier.gender-m.pt")
        bf, ff = _load(DATA_DIR / "pseudo_bank_purifier.gender-f.pt")
        models.update({"bank_m": bm, "fallback_m": fm, "bank_f": bf, "fallback_f": ff})
    if "mix" in modes:
        bx, fx = _load(DATA_DIR / "pseudo_bank_purifier.pt")
        models.update({"bank_mix": bx, "fallback_mix": fx})

    return models


def select_bank(models, src_gender, mode):
    key = {"same": src_gender, "cross": "f" if src_gender == "m" else "m", "mix": "mix"}[mode]
    return models[f"bank_{key}"], models[f"fallback_{key}"]


def adain(content, style_ref):
    """content/style_ref: [N, D]"""
    c_mean = content.mean(-1, keepdim=True)
    c_std  = content.std(-1, keepdim=True).clamp(min=1e-5)
    s_mean = style_ref.mean(-1, keepdim=True)
    s_std  = style_ref.std(-1, keepdim=True).clamp(min=1e-5)
    return s_std * (content - c_mean) / c_std + s_mean


@torch.no_grad()
def extract_features(audio_path, models, device):
    wav, sr = torchaudio.load(audio_path)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    wav = wav.mean(0, keepdim=True).to(device)

    multi = models["wavlm"].forward_multi_layer(wav, layers=[6, 24])
    l24 = multi[24].squeeze(0)
    purified = models["purifier"].encode(l24)
    phones = models["phone_predictor"](multi[24].squeeze(0)).cpu().numpy()

    return {"purified_l24": purified, "l6": multi[6].squeeze(0), "phones": phones}


@torch.no_grad()
def anonymize(source, models, src_gender, mode, top_k, device):
    bank, fallback = select_bank(models, src_gender, mode)
    query = source["purified_l24"].to(device)           # [T, 512]
    phones_t = torch.tensor(source["phones"], dtype=torch.long, device=device).squeeze()

    T = query.shape[0]
    h_anon = torch.zeros(T, 1024, device=device)

    for phone_id in tqdm(torch.unique(phones_t), desc=f"kNN [{mode}]"):
        ph = int(phone_id.item())
        mask = phones_t == phone_id
        N_q = mask.sum().item()

        entry = bank.get(ph, None)
        tgt_pur = entry["purified_l24"].to(device) if entry else fallback["purified_l24"].to(device)
        tgt_l6  = entry["l6"].to(device)           if entry else fallback["l6"].to(device)

        N_t = tgt_pur.shape[0]
        if N_t == 0:
            continue
        if N_t <= top_k:
            h_anon[mask] = tgt_l6.mean(0).expand(N_q, -1)
            continue

        _, idx = torch.cdist(query[mask], tgt_pur).topk(top_k, largest=False)
        aggregated = tgt_l6[idx].mean(1)
        h_anon[mask] = adain(aggregated, source["l6"].to(device)[mask])

    return h_anon.cpu()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audio",      default=DEFAULT_AUDIO)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--src-gender", default="m", choices=["m", "f"])
    p.add_argument("--mode",       default="all",  choices=["same", "cross", "mix", "all"])
    p.add_argument("--top-k",      type=int, default=4)
    p.add_argument("--device",     default="cuda")
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    modes = ["same", "cross", "mix"] if args.mode == "all" else [args.mode]
    out_dir = Path(args.output_dir)

    models = load_models(modes, device)
    source = extract_features(args.audio, models, device)

    for mode in modes:
        h_anon = anonymize(source, models, args.src_gender, mode, args.top_k, device)
        wav_out = models["vocoder"](h_anon.unsqueeze(0).to(device)).squeeze()
        out_path = out_dir / f"anon_{args.src_gender}_{mode}_top{args.top_k}.wav"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(out_path), wav_out.unsqueeze(0).cpu(), 16000)
        print(f"[+] {out_path}")


if __name__ == "__main__":
    main()
