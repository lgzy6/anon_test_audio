#!/usr/bin/env python3
"""
波形域韵律后处理模块 — F0 轮廓形状迁移 + 能量包络迁移

设计原则：
  - 在 HiFi-GAN 输出之后操作，L6 域的身份已被质心消除
  - 只迁移「相对形状」，不迁移绝对值
    · F0: z-score 归一化后迁移轮廓形状，保留匿名语音的音高范围
    · 能量: 迁移相对包络（帧能量/均值），保留匿名语音的绝对音量
  - beta 参数控制迁移强度 (0=不迁移, 1=完全迁移)

用法（在 pipeline 中）：
  from prosody_module import ProsodyTransfer
  pt = ProsodyTransfer(beta_f0=0.6, beta_energy=0.4)
  wav_out = pt.process(wav_anon, wav_source, sr=16000)

依赖：
  pip install praat-parselmouth --break-system-packages
"""
import numpy as np
import torch

try:
    import parselmouth
    from parselmouth.praat import call
    HAS_PRAAT = True
except ImportError:
    HAS_PRAAT = False
    print("[ProsodyTransfer] praat-parselmouth 未安装，F0 迁移不可用")


class ProsodyTransfer:
    """
    波形域韵律迁移：从源波形提取 F0 轮廓形状和能量包络，
    迁移到匿名波形上。

    参数:
      beta_f0:     F0 轮廓迁移强度 (0-1)，默认 0.6
      beta_energy: 能量包络迁移强度 (0-1)，默认 0.4
      f0_floor:    F0 提取下限 Hz，默认 75
      f0_ceil:     F0 提取上限 Hz，默认 500
      frame_ms:    能量计算帧长 ms，默认 10
    """

    def __init__(self, beta_f0=0.6, beta_energy=0.4,
                 f0_floor=75, f0_ceil=500, frame_ms=10, device="cpu"):
        self.beta_f0 = beta_f0
        self.beta_energy = beta_energy
        self.f0_floor = f0_floor
        self.f0_ceil = f0_ceil
        self.frame_ms = frame_ms
        self.device = device

    # ================================================================
    #  公开接口
    # ================================================================

    def process(self, wav_anon, wav_source, sr=16000):
        """
        将源波形的韵律特征迁移到匿名波形上。

        Args:
          wav_anon:   匿名波形 (torch.Tensor, 1D, 已在 CPU 或 GPU)
          wav_source: 源波形 (torch.Tensor, 1D)
          sr:         采样率

        Returns:
          torch.Tensor: 韵律迁移后的匿名波形
        """
        device = wav_anon.device
        anon_np = wav_anon.detach().cpu().numpy().astype(np.float64)
        src_np  = wav_source.detach().cpu().numpy().astype(np.float64)

        # Step 1: F0 轮廓形状迁移 (PSOLA)
        if self.beta_f0 > 0 and HAS_PRAAT:
            anon_np = self._transfer_f0_shape(anon_np, src_np, sr)

        # Step 2: 能量包络迁移
        if self.beta_energy > 0:
            anon_np = self._transfer_energy(anon_np, src_np, sr)

        return torch.from_numpy(anon_np.astype(np.float32)).to(device)

    # ================================================================
    #  兼容旧接口 (pipeline 中调用 process_wav)
    # ================================================================

    def process_wav(self, wav_anon, wav_source, sr=16000):
        return self.process(wav_anon, wav_source, sr)

    # ================================================================
    #  F0 轮廓形状迁移
    # ================================================================

    def _transfer_f0_shape(self, anon_np, src_np, sr):
        """
        提取源的 F0 轮廓形状 (z-score 归一化)，
        用 PSOLA 将其映射到匿名波形的音高范围上。

        迁移公式:
          f0_src_z = (f0_src - mean_src) / std_src     # 源的相对轮廓
          f0_target = mean_anon + f0_src_z * std_anon * beta
                    + f0_anon * (1 - beta)               # 混合
        """
        try:
            snd_anon = parselmouth.Sound(anon_np, sampling_frequency=sr)
            snd_src  = parselmouth.Sound(src_np, sampling_frequency=sr)

            # 提取 F0
            pitch_anon = snd_anon.to_pitch_ac(
                time_step=self.frame_ms / 1000,
                pitch_floor=self.f0_floor,
                pitch_ceiling=self.f0_ceil,
            )
            pitch_src = snd_src.to_pitch_ac(
                time_step=self.frame_ms / 1000,
                pitch_floor=self.f0_floor,
                pitch_ceiling=self.f0_ceil,
            )

            f0_anon = pitch_anon.selected_array["frequency"]
            f0_src  = pitch_src.selected_array["frequency"]

            # 只在两边都有浊音的帧上计算统计量
            voiced_anon = f0_anon > 0
            voiced_src  = f0_src > 0

            if voiced_anon.sum() < 5 or voiced_src.sum() < 5:
                return anon_np  # 浊音帧太少，跳过

            mean_anon = f0_anon[voiced_anon].mean()
            std_anon  = f0_anon[voiced_anon].std()
            mean_src  = f0_src[voiced_src].mean()
            std_src   = f0_src[voiced_src].std()

            if std_src < 1e-3 or std_anon < 1e-3:
                return anon_np  # F0 几乎恒定，跳过

            # 构建目标 F0 序列
            # 对齐长度: 取较短的
            min_len = min(len(f0_anon), len(f0_src))
            f0_anon_aligned = f0_anon[:min_len].copy()
            f0_src_aligned  = f0_src[:min_len].copy()

            # 对每个帧计算目标 F0
            f0_target = f0_anon_aligned.copy()
            for i in range(min_len):
                if f0_src_aligned[i] > 0 and f0_anon_aligned[i] > 0:
                    # 源的 z-score
                    z = (f0_src_aligned[i] - mean_src) / std_src
                    # 映射到匿名空间
                    f0_mapped = mean_anon + z * std_anon
                    # 混合
                    f0_target[i] = (1 - self.beta_f0) * f0_anon_aligned[i] + \
                                   self.beta_f0 * f0_mapped
                    # 安全裁剪
                    f0_target[i] = np.clip(f0_target[i], self.f0_floor, self.f0_ceil)
                # 如果源无声匿名有声 → 保持匿名 F0
                # 如果源有声匿名无声 → 保持无声 (不强制加 voicing)

            # 用 Praat PSOLA 修改匿名波形的 F0
            manipulation = call(snd_anon, "To Manipulation",
                                self.frame_ms / 1000, self.f0_floor, self.f0_ceil)
            pitch_tier = call(manipulation, "Extract pitch tier")

            # 清除原有 pitch points
            n_points = call(pitch_tier, "Get number of points")
            for _ in range(n_points):
                call(pitch_tier, "Remove point", 1)

            # 写入新的 F0 轨迹
            times = pitch_anon.xs()[:min_len]
            for i, t in enumerate(times):
                if f0_target[i] > 0:
                    call(pitch_tier, "Add point", float(t), float(f0_target[i]))

            call([manipulation, pitch_tier], "Replace pitch tier")
            result = call(manipulation, "Get resynthesis (overlap-add)")
            return result.values[0]

        except Exception as e:
            print(f"[ProsodyTransfer] F0 迁移失败: {e}")
            return anon_np

    # ================================================================
    #  能量包络迁移
    # ================================================================

    def _transfer_energy(self, anon_np, src_np, sr):
        """
        帧级能量包络迁移。
        提取源的相对能量模式 (每帧能量 / 全局均值)，
        将其应用到匿名波形上。

        迁移公式:
          ratio_src = frame_energy_src / mean_energy_src  # 相对包络
          gain = (1 - beta) * 1.0 + beta * ratio_src      # 混合增益
          wav_out = wav_anon * gain_interpolated
        """
        frame_len = int(sr * self.frame_ms / 1000)  # 10ms = 160 samples at 16kHz
        hop = frame_len  # 不重叠

        # 计算帧能量 (RMS)
        def frame_rms(wav, flen, hop):
            n_frames = max(1, (len(wav) - flen) // hop + 1)
            rms = np.zeros(n_frames)
            for i in range(n_frames):
                start = i * hop
                end = min(start + flen, len(wav))
                frame = wav[start:end]
                rms[i] = np.sqrt(np.mean(frame ** 2) + 1e-10)
            return rms

        rms_anon = frame_rms(anon_np, frame_len, hop)
        rms_src  = frame_rms(src_np, frame_len, hop)

        # 对齐长度
        min_frames = min(len(rms_anon), len(rms_src))
        rms_src = rms_src[:min_frames]
        rms_anon = rms_anon[:min_frames]

        # 源的相对能量模式
        mean_src = rms_src.mean()
        if mean_src < 1e-8:
            return anon_np
        ratio_src = rms_src / mean_src  # 相对包络, 均值=1

        # 混合增益: beta=0 → gain=1 (不变), beta=1 → gain=ratio_src
        gain_frames = (1 - self.beta_energy) * np.ones_like(ratio_src) + \
                      self.beta_energy * ratio_src

        # 防止极端增益
        gain_frames = np.clip(gain_frames, 0.1, 5.0)

        # 插值到样本级
        frame_centers = np.arange(min_frames) * hop + hop // 2
        sample_indices = np.arange(len(anon_np))
        gain_samples = np.interp(sample_indices, frame_centers, gain_frames,
                                 left=gain_frames[0], right=gain_frames[-1])

        result = anon_np * gain_samples
        # 防止 clipping
        peak = np.abs(result).max()
        if peak > 0.99:
            result = result * 0.95 / peak

        return result


# ================================================================
#  向后兼容：pipeline 中 import ProsodyInjector
# ================================================================

class ProsodyInjector:
    """
    兼容层：pipeline_top3Random_unitQuery.py 中
    from .prosody_module import ProsodyInjector
    """
    
    def __init__(self, device="cpu", beta=0.5,
                 beta_f0=None, beta_energy=None):
        self.transfer = ProsodyTransfer(
            beta_f0=beta_f0 if beta_f0 is not None else 0.0,      # 默认关闭 F0
            beta_energy=beta_energy if beta_energy is not None else beta,
            device=device,
        )

    def process_wav(self, wav_anon, wav_source, sr=16000):
        return self.transfer.process(wav_anon, wav_source, sr)