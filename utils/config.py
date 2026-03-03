# utils/config.py

"""配置加载工具"""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """加载 YAML 配置文件"""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def merge_configs(base:  Dict, override: Dict) -> Dict:
    """合并两个配置字典 (override 覆盖 base)"""
    import copy
    result = copy.deepcopy(base)
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    
    return result