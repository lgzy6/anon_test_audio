#!/usr/bin/env python3
"""
SAD 专用能量包络旁路 v2 —— 方向写死版（decrescendo + 动态压缩）

【为什么重写 v1】
v1 (_macro_gain) 迁「源的去均值宏观包络形状」，即如实搬运源的能量动态。
但「放大/搬运动态」声学上像 anger（高能量强动态），不像 sad。
实测：ACC_sad ↓、ACC_ang ↑ 大幅——方向搞反了。

【v2 设计原则】只施加两个 SAD 单方向分量，都不会把句子推向 ang：
  1. decrescendo：对源 log-能量拟合线性斜率，只取「负斜率」(收尾渐弱)，
     正斜率/起伏一律丢弃。施加一个整体下行 gain。
     —— 只迁「渐弱」这一个 sad 签名，不搬源的强动态段。
  2. dynamic compression：把匿名波形自身能量动态向「自身均值」压缩
     （sad 的能量是平的）。这一项【只用匿名波形自己】，不引入源信息，
     隐私上几乎免费；唯一碰源的是 decrescendo 的负斜率标量。

【隐私】比 v1 更安全：唯一的源依赖是一个标量负斜率（带宽极低）。
仍建议守 EER；但若 v2 仍救不动 ACC_sad，则 SAD 波形域旁路正式证伪。

用法（HiFi-GAN 之后、波形域）：
    pt = SadEnergyBypassV2(slope_strength=0.5, compress=0.4)
    wav_out = pt.process(wav_anon, wav_source, sr=16000, sad_prob=p_sad)
"""
import numpy as np
import torch


class SadEnergyBypassV2:
    def __init__(
        self,
        slope_strength=0.5,   # decrescendo 强度（迁源负斜率的比例，0=不迁）
        compress=0.4,         # 动态压缩强度（0=不压, 1=完全压平到均值）
        max_drop_db=6.0,      # decrescendo 句首→句尾最大允许下行（dB），防过度
        frame_ms=10,
        smooth_ms=400,        # 压缩前的包络平滑窗（去微动态，只压宏观）
        sad_gate=0.5,
        gain_clip=(0.4, 1.5),
        device="cpu",
    ):
        self.slope_strength = slope_strength
        self.compress = compress
        self.max_drop_db = max_drop_db
        self.frame_ms = frame_ms
        self.smooth_ms = smooth_ms
        self.sad_gate = sad_gate
        self.gain_clip = gain_clip
        self.device = device
        # 兼容旧 config 字段名（beta 早期映射到 slope_strength）
        self.beta = slope_strength

    # ----------------------------------------------------------------
    def process(self, wav_anon, wav_source, sr=16000, sad_prob=1.0):
        if sad_prob < self.sad_gate or (self.slope_strength <= 0 and self.compress <= 0):
            return wav_anon

        device = wav_anon.device
        anon = wav_anon.detach().cpu().numpy().astype(np.float64)
        src = wav_source.detach().cpu().numpy().astype(np.float64)

        gain = self._sad_gain(anon, src, sr)
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

    def _sad_gain(self, anon, src, sr):
        log_anon, hop = self._frame_log_rms(anon, sr)
        log_src, _ = self._frame_log_rms(src, sr)
        win = max(3, int(self.smooth_ms / self.frame_ms))
        n = len(log_anon)

        # 工作在 log 域，最终 exp 回幅度域。两个分量叠加。
        gain_log = np.zeros(n)

        # ── 分量 1：decrescendo（只迁源的负斜率，单方向） ──
        if self.slope_strength > 0 and len(log_src) >= 3:
            t = np.linspace(0, 1, len(log_src))
            # 源 log-能量线性拟合：slope = 每「全句」的 log 变化量
            slope = np.polyfit(t, log_src, 1)[0]
            if slope < 0:                       # 只在源「收尾渐弱」时施加
                # 限幅：句首→句尾下行不超过 max_drop_db
                max_drop_log = self.max_drop_db / 8.686   # dB → 自然 log
                drop = min(-slope, max_drop_log) * self.slope_strength
                # 句首 +drop/2，句尾 -drop/2，去均值 → 不改整体响度，只给下行
                ramp = np.linspace(drop / 2, -drop / 2, n)
                gain_log += ramp

        # ── 分量 2：动态压缩（只用匿名自身，不碰源） ──
        if self.compress > 0:
            macro_anon = self._smooth(log_anon, win)
            # 把宏观包络往自身均值收：compress=1 → 完全压平
            gain_log += -self.compress * (macro_anon - macro_anon.mean())

        gain_frames = np.clip(np.exp(gain_log), *self.gain_clip)

        centers = np.arange(n) * hop + hop // 2
        idx = np.arange(len(anon))
        return np.interp(idx, centers, gain_frames,
                         left=gain_frames[0], right=gain_frames[-1])


# 向后兼容别名：pipeline 里 from prosody_sad_energy import SadEnergyBypass
SadEnergyBypass = SadEnergyBypassV2