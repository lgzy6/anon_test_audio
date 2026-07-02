#!/usr/bin/env python3
"""prosody_sad_energy_gate.py

SAD 专用能量包络旁路 —— 软门控版 / gate版（decrescendo + 动态压缩）

【相对 v2 的唯一改动】
v2: process(sad_prob) 用硬阈值 —— sad_prob<sad_gate 直接不施加，否则全量施加。
    问题：被误判的 NEU/边缘句要么吃满破坏性处理、要么完全不处理，二值化太粗。
v3: process(gate) 接收一个连续门控强度 gate∈[0,1]，把效果【按 gate 线性缩放】。
    gate=0 → 完全恒等（不动波形）；gate=1 → 全量效果；gate=0.3 → 30% 效果。
    门控强度 g 的计算（软门控 + anti-ang 守门）放在 pipeline 侧（策略层），
    本模块只当纯信号处理器，对 g 做缩放，职责清晰、EER 攸关逻辑不进来。

  缩放实现：gain_log（log 域增益，0=恒等）整体 ×gate 后再 exp。
    因为两个分量都构造成「gain_log=0 ⇔ 不改波形」，所以 ×gate 平滑地
    在「恒等」和「全量」之间插值，数学上干净，gain_clip 仍兜底。

【隐私】与 v2 完全一致：唯一源依赖仍是 decrescendo 的标量负斜率（带宽极低）。
  软门控不引入任何新的源信息流，只是把已有效果按置信度缩小。仍须守 EER。

用法（HiFi-GAN 之后、波形域）：
    pt = SadEnergyBypassV3(slope_strength=0.5, compress=0.4)
    wav_out = pt.process(wav_anon, wav_source, sr=16000, gate=g)   # g∈[0,1]
"""
import numpy as np
import torch


class SadEnergyBypassV3:
    def __init__(
        self,
        slope_strength=0.5,   # decrescendo 强度（迁源负斜率的比例，0=不迁）
        compress=0.4,         # 动态压缩强度（0=不压, 1=完全压平到均值）
        max_drop_db=6.0,      # decrescendo 句首→句尾最大允许下行（dB），防过度
        frame_ms=10,
        smooth_ms=400,        # 压缩前的包络平滑窗（去微动态，只压宏观）
        sad_gate=0.5,         # 【已弃用】保留仅为向后兼容旧 config；门控现由 pipeline 算
        gain_clip=(0.4, 1.5),
        device="cpu",
    ):
        self.slope_strength = slope_strength
        self.compress = compress
        self.max_drop_db = max_drop_db
        self.frame_ms = frame_ms
        self.smooth_ms = smooth_ms
        self.sad_gate = sad_gate          # legacy, 不再用于硬判决
        self.gain_clip = gain_clip
        self.device = device
        self.beta = slope_strength        # 兼容旧 config 字段名

    # ----------------------------------------------------------------
    def process(self, wav_anon, wav_source, sr=16000, gate=1.0):
        """gate: 门控强度 ∈[0,1]。0=不动；1=全量效果。
        向后兼容：若上游仍按 v2 传 sad_prob（0/1 二值或概率），数值同样落在
        [0,1]，会被当作连续强度处理 —— 不会报错，只是退化成「按概率缩放」。
        """
        if gate <= 1e-3 or (self.slope_strength <= 0 and self.compress <= 0):
            return wav_anon

        device = wav_anon.device
        anon = wav_anon.detach().cpu().numpy().astype(np.float64)
        src = wav_source.detach().cpu().numpy().astype(np.float64)

        gain = self._sad_gain(anon, src, sr, gate)
        out = anon * gain

        peak = np.abs(out).max()
        if peak > 0.99:
            out = out * 0.95 / peak
        return torch.from_numpy(out.astype(np.float32)).to(device)

    # ----------------------------------------------------------------
    def _frame_log_rms(self, wav, sr):
        flen = int(sr * self.frame_ms / 1000)
        hop = flen
        n = max(1, (len(wav) - flen) // hop + 1)
        log_rms = np.empty(n)
        for i in range(n):
            s = i * hop
            frame = wav[s:s + flen]
            log_rms[i] = np.log(np.sqrt(np.mean(frame ** 2) + 1e-10))
        return log_rms, hop

    def _smooth(self, x, win_frames):
        if win_frames < 3 or len(x) < win_frames:
            return x.copy()
        w = np.hanning(win_frames); w /= w.sum()
        pad = win_frames // 2
        xp = np.pad(x, pad, mode="reflect")
        return np.convolve(xp, w, mode="same")[pad:pad + len(x)]

    def _sad_gain(self, anon, src, sr, gate=1.0):
        log_anon, hop = self._frame_log_rms(anon, sr)
        log_src, _ = self._frame_log_rms(src, sr)
        win = max(3, int(self.smooth_ms / self.frame_ms))
        n = len(log_anon)

        gain_log = np.zeros(n)

        # ── 分量 1：decrescendo（只迁源的负斜率，单方向） ──
        if self.slope_strength > 0 and len(log_src) >= 3:
            t = np.linspace(0, 1, len(log_src))
            slope = np.polyfit(t, log_src, 1)[0]
            if slope < 0:
                max_drop_log = self.max_drop_db / 8.686
                drop = min(-slope, max_drop_log) * self.slope_strength
                ramp = np.linspace(drop / 2, -drop / 2, n)
                gain_log += ramp

        # ── 分量 2：动态压缩（只用匿名自身，不碰源） ──
        if self.compress > 0:
            macro_anon = self._smooth(log_anon, win)
            gain_log += -self.compress * (macro_anon - macro_anon.mean())

        # ── v3：按门控强度连续缩放（gate=0 → gain_log=0 → 恒等） ──
        gain_log *= float(np.clip(gate, 0.0, 1.0))

        gain_frames = np.clip(np.exp(gain_log), *self.gain_clip)

        centers = np.arange(n) * hop + hop // 2
        idx = np.arange(len(anon))
        return np.interp(idx, centers, gain_frames,
                         left=gain_frames[0], right=gain_frames[-1])


# 向后兼容别名：pipeline 里 from prosody_sad_energy import SadEnergyBypass
SadEnergyBypass = SadEnergyBypassV3