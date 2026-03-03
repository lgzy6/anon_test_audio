# models/phone_predictor/predictor.py
"""
Phone Predictor - 统一接口，兼容多种 checkpoint 格式
支持:
1. pknnvc 的 ConvDecoder 格式
2. 自定义 LSTM 格式
3. 简单 MLP 格式
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Union, Tuple
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# pknnvc ConvDecoder (原始实现)
# =============================================================================

class ConvDecoder(nn.Module):
    """
    pknnvc 的 ConvDecoder
    用于 phone 预测和 duration 预测
    """
    def __init__(
        self,
        encoder_embed_dim: int = 256,
        hidden_dim: int = 256,
        kernel_size: int = 3,
        dropout: float = 0.5,
        output_dim: int = 1,
        emb_dim: int = -1,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.output_dim = output_dim
        
        if emb_dim > 0:
            self.emb = nn.Embedding(emb_dim, encoder_embed_dim, padding_idx=0)
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                encoder_embed_dim,
                hidden_dim,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
            ),
            nn.ReLU(),
        )
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(p=dropout)
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                hidden_dim,
                hidden_dim,
                kernel_size=kernel_size,
                padding=1,
            ),
            nn.ReLU(),
        )
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.dropout2 = nn.Dropout(p=dropout)
        
        self.proj = nn.Linear(hidden_dim, output_dim)
    
    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] 特征 或 [B, T] phone indices (如果有embedding)
        Returns:
            logits: [B, T, output_dim] 或 [B, T] (如果 output_dim=1)
        """
        if self.emb_dim > 0:
            x = self.emb(x)
        
        x = self.conv1(x.transpose(-2, -1)).transpose(-2, -1)
        x = self.dropout1(self.ln1(x))
        x = self.conv2(x.transpose(-2, -1)).transpose(-2, -1)
        x = self.dropout2(self.ln2(x))
        x = self.proj(x)
        
        if self.output_dim == 1:
            x = x.squeeze(dim=-1)
        elif self.output_dim > 1:
            x = x.log_softmax(dim=-1)
        
        return x


def load_conv_decoder(ckpt_path: str, device: str = 'cuda') -> ConvDecoder:
    """加载 pknnvc ConvDecoder checkpoint"""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    emb_dim = -1 if "emb.weight" not in ckpt else ckpt["emb.weight"].shape[0]
    
    model = ConvDecoder(
        encoder_embed_dim=ckpt["conv1.0.weight"].shape[1],
        hidden_dim=ckpt["conv1.0.weight"].shape[0],
        kernel_size=ckpt["conv1.0.weight"].shape[2],
        dropout=0,  # inference 时不需要 dropout
        output_dim=ckpt["proj.weight"].shape[0],
        emb_dim=emb_dim,
    ).to(device)
    
    model.load_state_dict(ckpt)
    model.eval()
    
    logger.info(f"Loaded ConvDecoder: input={ckpt['conv1.0.weight'].shape[1]}, "
                f"output={ckpt['proj.weight'].shape[0]}, emb={emb_dim}")
    
    return model


# =============================================================================
# 统一 Phone Predictor 接口
# =============================================================================

class PhonePredictor(nn.Module):
    """
    统一的 Phone Predictor 接口
    
    自动检测并加载不同格式的 checkpoint:
    1. pknnvc ConvDecoder
    2. LSTM-based
    3. Simple MLP
    """
    
    # CMU Phoneset
    PHONE_LIST = [
        'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH',
        'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K',
        'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH',
        'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH', 'SIL', 'SPN'
    ]
    
    def __init__(self, model: nn.Module, model_type: str = 'conv_decoder'):
        super().__init__()
        self.model = model
        self.model_type = model_type
        self.num_phones = len(self.PHONE_LIST)
    
    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        预测 phone
        
        Args:
            x: [B, T, D] 或 [T, D] WavLM 特征
        
        Returns:
            phones: [B, T] 或 [T] phone indices
        """
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze = True
        
        logits = self.model(x)
        
        if logits.dim() == 3:
            phones = logits.argmax(dim=-1)
        else:
            phones = logits.long()
        
        if squeeze:
            phones = phones.squeeze(0)
        
        return phones
    
    def predict_with_probs(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回 phone 预测和概率"""
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze = True
        
        logits = self.model(x)
        probs = torch.softmax(logits, dim=-1) if logits.dim() == 3 else None
        phones = logits.argmax(dim=-1) if logits.dim() == 3 else logits.long()
        
        if squeeze:
            phones = phones.squeeze(0)
            if probs is not None:
                probs = probs.squeeze(0)
        
        return phones, probs
    
    def phone_to_name(self, phone_idx: int) -> str:
        """将 phone index 转换为名称"""
        if 0 <= phone_idx < len(self.PHONE_LIST):
            return self.PHONE_LIST[phone_idx]
        return 'UNK'

    def get_phone_durations(self, phones: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从帧级音素序列提取音素ID和时长

        将连续相同的音素合并，计算每个音素的持续帧数

        Args:
            phones: [T] 帧级音素预测序列

        Returns:
            phone_ids: [N] 去重后的音素ID序列
            durations: [N] 每个音素的持续帧数
        """
        if phones.dim() != 1:
            raise ValueError(f"Expected 1D tensor, got shape {phones.shape}")

        phone_ids = []
        durations = []

        if len(phones) == 0:
            return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)

        current_phone = phones[0].item()
        current_duration = 1

        for i in range(1, len(phones)):
            if phones[i].item() == current_phone:
                current_duration += 1
            else:
                phone_ids.append(current_phone)
                durations.append(current_duration)
                current_phone = phones[i].item()
                current_duration = 1

        # 添加最后一个音素
        phone_ids.append(current_phone)
        durations.append(current_duration)

        return torch.tensor(phone_ids, dtype=torch.long), torch.tensor(durations, dtype=torch.long)

    def get_phone_segments(self, phones: torch.Tensor) -> list:
        """
        从帧级音素序列提取音素段信息

        Args:
            phones: [T] 帧级音素预测序列

        Returns:
            segments: List of (start, end, phone_id)
                start: 段起始帧索引
                end: 段结束帧索引（不包含）
                phone_id: 音素ID
        """
        if phones.dim() != 1:
            raise ValueError(f"Expected 1D tensor, got shape {phones.shape}")

        segments = []

        if len(phones) == 0:
            return segments

        current_phone = phones[0].item()
        start_frame = 0

        for i in range(1, len(phones)):
            if phones[i].item() != current_phone:
                segments.append((start_frame, i, current_phone))
                current_phone = phones[i].item()
                start_frame = i

        # 添加最后一个段
        segments.append((start_frame, len(phones), current_phone))

        return segments
    
    @classmethod
    def load(cls, checkpoint_path: str, device: str = 'cuda') -> 'PhonePredictor':
        """
        自动检测并加载 checkpoint
        """
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        
        # 检测 checkpoint 类型
        if isinstance(ckpt, dict):
            keys = set(ckpt.keys())
            
            # pknnvc ConvDecoder 格式
            if 'conv1.0.weight' in keys and 'ln1.weight' in keys:
                logger.info("Detected pknnvc ConvDecoder format")
                model = load_conv_decoder(checkpoint_path, device)
                return cls(model, model_type='conv_decoder')
            
            # 包装格式 (model state_dict)
            if 'model' in keys or 'state_dict' in keys:
                state_dict = ckpt.get('model', ckpt.get('state_dict', ckpt))
                return cls._load_from_state_dict(state_dict, device)
            
            # 直接是 state_dict
            return cls._load_from_state_dict(ckpt, device)
        
        raise ValueError(f"Unknown checkpoint format: {type(ckpt)}")
    
    @classmethod
    def _load_from_state_dict(cls, state_dict: dict, device: str) -> 'PhonePredictor':
        """从 state_dict 推断模型结构并加载"""
        keys = set(state_dict.keys())
        
        # 检测是否是 ConvDecoder
        if 'conv1.0.weight' in keys:
            emb_dim = -1 if "emb.weight" not in state_dict else state_dict["emb.weight"].shape[0]
            model = ConvDecoder(
                encoder_embed_dim=state_dict["conv1.0.weight"].shape[1],
                hidden_dim=state_dict["conv1.0.weight"].shape[0],
                kernel_size=state_dict["conv1.0.weight"].shape[2],
                dropout=0,
                output_dim=state_dict["proj.weight"].shape[0],
                emb_dim=emb_dim,
            )
            model.load_state_dict(state_dict)
            model.to(device).eval()
            return cls(model, model_type='conv_decoder')
        
        # 检测是否是 LSTM 格式
        if any('lstm' in k.lower() for k in keys):
            logger.info("Detected LSTM format")
            model = cls._build_lstm_model(state_dict)
            model.load_state_dict(state_dict)
            model.to(device).eval()
            return cls(model, model_type='lstm')
        
        # 简单 MLP 格式
        logger.info("Detected MLP format")
        model = cls._build_mlp_model(state_dict)
        model.load_state_dict(state_dict)
        model.to(device).eval()
        return cls(model, model_type='mlp')
    
    @staticmethod
    def _build_lstm_model(state_dict: dict) -> nn.Module:
        """构建 LSTM 模型"""
        # 从 state_dict 推断维度
        input_dim = state_dict.get('input_proj.weight', 
                                   state_dict.get('lstm.weight_ih_l0')).shape[1]
        hidden_dim = state_dict.get('lstm.weight_hh_l0').shape[1]
        output_dim = state_dict.get('proj.weight', 
                                    state_dict.get('output_proj.weight')).shape[0]
        
        return LSTMPhonePredictor(input_dim, hidden_dim, output_dim)
    
    @staticmethod
    def _build_mlp_model(state_dict: dict) -> nn.Module:
        """构建 MLP 模型"""
        # 找到第一个和最后一个 linear 层
        linear_keys = [k for k in state_dict.keys() if 'weight' in k and state_dict[k].dim() == 2]
        
        input_dim = state_dict[linear_keys[0]].shape[1]
        output_dim = state_dict[linear_keys[-1]].shape[0]
        hidden_dim = state_dict[linear_keys[0]].shape[0]
        
        return SimplePhonePredictor(input_dim, hidden_dim, output_dim)


# =============================================================================
# 备用模型实现
# =============================================================================

class LSTMPhonePredictor(nn.Module):
    """LSTM-based Phone Predictor"""
    
    def __init__(self, input_dim: int = 1024, hidden_dim: int = 512, 
                 num_phones: int = 41, num_layers: int = 2):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        self.ln = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, num_phones)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h, _ = self.lstm(h)
        h = self.ln(h)
        return self.proj(h)


class SimplePhonePredictor(nn.Module):
    """Simple MLP Phone Predictor"""
    
    def __init__(self, input_dim: int = 1024, hidden_dim: int = 512, 
                 num_phones: int = 41):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_phones),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


# =============================================================================
# Duration Predictor (复用 ConvDecoder)
# =============================================================================

class DurationPredictor:
    """
    Duration Predictor 包装器
    使用 pknnvc 的 ConvDecoder
    """

    def __init__(self, model: ConvDecoder, device: str = 'cuda'):
        self.model = model
        self.device = device

    def __call__(self, phone_ids: torch.Tensor) -> torch.Tensor:
        """使 DurationPredictor 可调用，委托给 predict 方法"""
        return self.predict(phone_ids)

    @torch.inference_mode()
    def predict(self, phone_ids: torch.Tensor) -> torch.Tensor:
        """
        预测时长

        Args:
            phone_ids: [B, T] 或 [T] phone indices
        Returns:
            durations: [B, T] 或 [T] 预测帧数
        """
        squeeze = False
        if phone_ids.dim() == 1:
            phone_ids = phone_ids.unsqueeze(0)
            squeeze = True

        # 确保输入在正确的设备上
        phone_ids = phone_ids.to(self.device)

        durations = self.model(phone_ids)

        if squeeze:
            durations = durations.squeeze(0)

        return torch.relu(durations)
    
    @classmethod
    def load(cls, checkpoint_path: str, device: str = 'cuda') -> 'DurationPredictor':
        model = load_conv_decoder(checkpoint_path, device)
        return cls(model, device=device)


# =============================================================================
# 便捷函数
# =============================================================================

def create_phone_predictor(config: dict) -> PhonePredictor:
    """从配置创建 Phone Predictor"""
    ckpt_path = config.get('phone_predictor', {}).get('checkpoint')
    device = config.get('device', 'cuda')
    
    if ckpt_path and Path(ckpt_path).exists():
        return PhonePredictor.load(ckpt_path, device)
    
    raise FileNotFoundError(f"Phone predictor checkpoint not found: {ckpt_path}")


def create_duration_predictor(config: dict) -> Optional[DurationPredictor]:
    """从配置创建 Duration Predictor"""
    ckpt_path = config.get('duration_predictor', {}).get('checkpoint')
    device = config.get('device', 'cuda')
    
    if ckpt_path and Path(ckpt_path).exists():
        return DurationPredictor.load(ckpt_path, device)
    
    return None