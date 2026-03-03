# data/io/kaldi.py

"""Kaldi 格式读写"""

from pathlib import Path
from typing import Dict, List, Tuple


def read_scp(path: str) -> Dict[str, str]:
    """读取 scp 文件 (wav.scp, feats.scp 等)"""
    result = {}
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                result[parts[0]] = parts[1]
    return result


def read_map(path: str) -> Dict[str, str]:
    """读取映射文件 (utt2spk, spk2gender 等)"""
    result = {}
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                result[parts[0]] = parts[1]
    return result


def read_trials(path: str) -> List[Tuple[str, str, str]]:
    """读取 trials 文件"""
    trials = []
    with open(path, 'r') as f:
        for line in f: 
            parts = line.strip().split()
            if len(parts) == 3:
                trials.append((parts[0], parts[1], parts[2]))
    return trials