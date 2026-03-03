# pipelines/offline/runner_dual_gpu.py
"""Offline 完整流程 Runner (双 GPU 版本)

专为双 4090 + 240GB 内存环境优化
使用方法:
    python scripts/run_offline.py --config configs/dual_4090.yaml
"""

import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .feature_extraction import run_feature_extraction
from .subspace_learning import run_subspace_learning
from .feature_cleaning import run_feature_cleaning
from .codebook_training import run_codebook_training
from .pattern_learning import run_pattern_learning
from .pool_building import run_pool_building

# 双 GPU 版本
try:
    from .pool_building_dual_gpu import run_pool_building_dual_gpu
    HAS_DUAL_GPU_POOL_BUILDING = True
except ImportError:
    HAS_DUAL_GPU_POOL_BUILDING = False

# Robust 版本
try:
    from .pool_building_robust import run_pool_building_robust
    HAS_ROBUST_POOL_BUILDING = True
except ImportError:
    HAS_ROBUST_POOL_BUILDING = False

# Prototype 版本 (研究导向优化)
try:
    from .pool_building_prototype import run_pool_building_prototype
    HAS_PROTOTYPE_POOL_BUILDING = True
except ImportError:
    HAS_PROTOTYPE_POOL_BUILDING = False

# v3.0 版本 (Pattern 分区)
try:
    from .pool_building_v3 import run_pool_building_v3
    HAS_V3_POOL_BUILDING = True
except ImportError:
    HAS_V3_POOL_BUILDING = False

# 低内存版本 (120GB 内存优化)
try:
    from .pool_building_low_mem import run_pool_building_low_mem
    HAS_LOW_MEM_POOL_BUILDING = True
except ImportError:
    HAS_LOW_MEM_POOL_BUILDING = False

# 采样版本 (从大数据集中采样构建)
try:
    from .pool_building_sampled import run_pool_building_sampled
    HAS_SAMPLED_POOL_BUILDING = True
except ImportError:
    HAS_SAMPLED_POOL_BUILDING = False

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class OfflineRunner:
    """
    Offline 流程统一运行器 (支持多种模式)

    支持三种 pool_building 模式:
    1. dual_gpu: 双 GPU 并行 (推荐用于双 4090 + 大内存)
    2. robust: 断点续传 (内存受限时使用)
    3. standard: 标准版本
    """

    def __init__(self, config: Dict):
        self.config = config
        self.config['_config_path'] = config.get('_config_path', 'N/A')
        self._ensure_directories()
        self._validate_config()
        self._setup_pool_building()

    def _setup_pool_building(self):
        """根据配置选择 pool_building 版本"""
        pool_cfg = self.config.get('offline', {}).get('pool_building', {})

        use_v3 = pool_cfg.get('use_v3_version', False)
        use_prototype = pool_cfg.get('use_prototype_version', False)
        use_dual_gpu = pool_cfg.get('use_dual_gpu_version', False)
        use_robust = pool_cfg.get('use_robust_version', False)
        use_low_mem = pool_cfg.get('use_low_mem_version', False)
        use_sampled = pool_cfg.get('use_sampled_version', False)

        if use_sampled and HAS_SAMPLED_POOL_BUILDING:
            self._pool_building_func = run_pool_building_sampled
            logger.info("=" * 60)
            logger.info("Using SAMPLED pool building (stratified sampling)")
            logger.info("  Strategy: Sample ~1500M from 5500M frames")
            logger.info("  Memory: ~50-60GB peak")
            logger.info("=" * 60)
        elif use_low_mem and HAS_LOW_MEM_POOL_BUILDING:
            self._pool_building_func = run_pool_building_low_mem
            logger.info("=" * 60)
            logger.info("Using LOW MEMORY pool building (optimized for 120GB RAM)")
            logger.info("  Features: Streaming processing, sampled sub-indices")
            logger.info("  Memory: ~60GB peak (vs 200GB+ for standard)")
            logger.info("=" * 60)
        elif use_v3 and HAS_V3_POOL_BUILDING:
            self._pool_building_func = run_pool_building_v3
            logger.info("=" * 60)
            logger.info("Using V3 pool building (Pattern-partitioned)")
            logger.info("  Features: Pattern clustering, TargetSelector support")
            logger.info("  No symbol constraint (fixes WER > 100%)")
            logger.info("=" * 60)
        elif use_prototype and HAS_PROTOTYPE_POOL_BUILDING:
            self._pool_building_func = run_pool_building_prototype
            logger.info("=" * 60)
            logger.info("Using PROTOTYPE pool building (research-oriented optimization)")
            logger.info("  Compression: 55M -> ~100K (550x)")
            logger.info("  Build time: ~10 minutes")
            logger.info("=" * 60)
        elif use_dual_gpu and HAS_DUAL_GPU_POOL_BUILDING:
            self._pool_building_func = run_pool_building_dual_gpu
            logger.info("=" * 60)
            logger.info("Using DUAL GPU pool building (optimized for dual 4090)")
            logger.info("=" * 60)
        elif use_robust and HAS_ROBUST_POOL_BUILDING:
            self._pool_building_func = run_pool_building_robust
            logger.info("Using ROBUST pool building (with checkpointing)")
        else:
            self._pool_building_func = run_pool_building
            if use_prototype and not HAS_PROTOTYPE_POOL_BUILDING:
                logger.warning("Prototype pool building requested but not available")
            if use_dual_gpu and not HAS_DUAL_GPU_POOL_BUILDING:
                logger.warning("Dual GPU pool building requested but not available")
            if use_robust and not HAS_ROBUST_POOL_BUILDING:
                logger.warning("Robust pool building requested but not available")
            logger.info("Using STANDARD pool building")

    @property
    def STEPS(self):
        """动态生成步骤列表"""
        return [
            ('feature_extraction', run_feature_extraction, 'Step 1: WavLM 特征提取'),
            ('subspace_learning', run_subspace_learning, 'Step 2: 说话人子空间学习'),
            ('feature_cleaning', run_feature_cleaning, 'Step 3: 去说话人特征生成'),
            ('codebook_training', run_codebook_training, 'Step 4: Codebook 训练'),
            ('pattern_learning', run_pattern_learning, 'Step 5: Pattern Matrix 学习'),
            ('pool_building', self._pool_building_func, 'Step 6: Target Pool 构建'),
        ]

    def _ensure_directories(self):
        """确保必要的目录存在"""
        dirs = [
            self.config['paths']['cache_dir'],
            self.config['paths']['checkpoints_dir'],
            Path(self.config['paths']['cache_dir']) / 'features' / 'wavlm',
            Path(self.config['paths']['cache_dir']) / 'features' / 'cleaned',
            self.config['paths'].get('log_dir', 'logs'),
        ]
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)

    def _validate_config(self):
        """验证配置完整性"""
        required_keys = [
            'paths.cache_dir',
            'paths.checkpoints_dir',
            'paths.wavlm_checkpoint',
            'ssl.layer',
            'ssl.hidden_dim',
        ]

        missing = []
        for key in required_keys:
            parts = key.split('.')
            val = self.config
            for p in parts:
                val = val.get(p, {}) if isinstance(val, dict) else None
            if val is None or val == {}:
                missing.append(key)

        if missing:
            logger.warning(f"Missing config keys: {missing}")

    def run(
        self,
        start_step: int = 1,
        end_step: int = 6,
        steps: Optional[List[int]] = None,
        skip_on_error: bool = False,
    ) -> Dict:
        """运行 Offline 流程"""
        if steps is not None:
            step_indices = [s - 1 for s in steps if 1 <= s <= 6]
        else:
            step_indices = list(range(start_step - 1, end_step))

        # 打印环境信息
        self._print_environment_info()

        logger.info("=" * 70)
        logger.info(" " * 15 + "SAMM-Anon Offline Pipeline (Dual GPU)")
        logger.info("=" * 70)
        logger.info(f"Steps to run: {[i + 1 for i in step_indices]}")
        logger.info(f"Device: {self.config.get('device', 'cuda')}")
        logger.info(f"GPUs: {self.config.get('gpu_ids', [0, 1])}")
        logger.info(f"Config: {self.config.get('_config_path', 'N/A')}")

        total_time = 0
        results = {}
        failed_steps: List[Tuple[int, str]] = []

        for idx in step_indices:
            step_name, step_func, step_desc = self.STEPS[idx]
            step_num = idx + 1

            if not self.check_dependencies(step_num):
                logger.error(f"Step {step_num} 依赖检查失败，跳过")
                failed_steps.append((step_num, step_name))
                if not skip_on_error:
                    break
                continue

            logger.info("\n" + "=" * 70)
            logger.info(f"[{step_num}/6] {step_desc}")
            logger.info("=" * 70)

            start_time = time.time()

            try:
                result = step_func(self.config)
                results[step_name] = result

                elapsed = time.time() - start_time
                total_time += elapsed

                logger.info(f"\n✓ {step_desc} 完成")
                logger.info(f"  耗时: {elapsed:.1f}s ({elapsed/60:.1f}min)")

            except KeyboardInterrupt:
                logger.warning(f"\n⚠️ 用户中断")
                logger.info(f"已完成步骤: {list(results.keys())}")
                sys.exit(0)

            except Exception as e:
                logger.error(f"\n❌ {step_desc} 失败!")
                logger.error(f"错误: {e}")

                import traceback
                traceback.print_exc()

                failed_steps.append((step_num, step_name))

                if not skip_on_error:
                    logger.info(f"\n已完成步骤: {list(results.keys())}")
                    raise

        self._print_summary(results, failed_steps, total_time, step_indices)

        return results

    def _print_environment_info(self):
        """打印环境信息"""
        import torch
        import psutil

        logger.info("\n" + "=" * 70)
        logger.info(" " * 25 + "环境信息")
        logger.info("=" * 70)

        # 内存信息
        mem = psutil.virtual_memory()
        logger.info(f"系统内存: {mem.total / 1e9:.1f} GB (可用: {mem.available / 1e9:.1f} GB)")

        # GPU 信息
        if torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
            logger.info(f"GPU 数量: {n_gpus}")
            for i in range(n_gpus):
                props = torch.cuda.get_device_properties(i)
                mem_total = props.total_memory / 1e9
                mem_free = (props.total_memory - torch.cuda.memory_allocated(i)) / 1e9
                logger.info(f"  GPU {i}: {props.name} ({mem_total:.1f} GB, 可用: {mem_free:.1f} GB)")
        else:
            logger.warning("CUDA 不可用")

        logger.info("=" * 70)

    def _print_summary(self, results: Dict, failed_steps: List,
                       total_time: float, step_indices: List[int]):
        """打印运行总结"""
        logger.info("\n" + "=" * 70)
        if failed_steps:
            logger.warning("⚠️ Pipeline 部分完成")
            logger.warning(f"失败步骤: {failed_steps}")
        else:
            logger.info("✓ Offline Pipeline 全部完成!")
        logger.info("=" * 70)
        logger.info(f"总耗时: {total_time:.1f}s ({total_time/60:.1f}min / {total_time/3600:.2f}h)")
        logger.info(f"完成步骤: {len(results)}/{len(step_indices)}")
        logger.info("=" * 70)

    def check_dependencies(self, step: int) -> bool:
        """检查某步骤的依赖是否满足"""
        cache_dir = Path(self.config['paths']['cache_dir'])
        checkpoint_dir = Path(self.config['paths']['checkpoints_dir'])

        dependencies = {
            1: [],
            2: [cache_dir / 'features' / 'wavlm' / 'features.h5'],
            3: [
                cache_dir / 'features' / 'wavlm' / 'features.h5',
                checkpoint_dir / 'speaker_subspace.pt',
            ],
            4: [cache_dir / 'features' / 'cleaned' / 'features.h5'],
            5: [
                cache_dir / 'features' / 'cleaned' / 'features.h5',
                checkpoint_dir / 'codebook.pt',
            ],
            6: [
                cache_dir / 'features' / 'cleaned' / 'features.h5',
                cache_dir / 'features' / 'cleaned' / 'metadata.json',
                checkpoint_dir / 'codebook.pt',
            ],
        }

        missing = []
        for dep in dependencies.get(step, []):
            if not dep.exists():
                missing.append(str(dep))

        if missing:
            logger.error(f"Step {step} 缺少依赖文件:")
            for m in missing:
                logger.error(f"  - {m}")
            return False

        return True

    def get_status(self) -> Dict[int, bool]:
        """获取各步骤的完成状态"""
        cache_dir = Path(self.config['paths']['cache_dir'])
        checkpoint_dir = Path(self.config['paths']['checkpoints_dir'])

        status = {
            1: (cache_dir / 'features' / 'wavlm' / 'features.h5').exists(),
            2: (checkpoint_dir / 'speaker_subspace.pt').exists(),
            3: (cache_dir / 'features' / 'cleaned' / 'features.h5').exists(),
            4: (checkpoint_dir / 'codebook.pt').exists() or
               (checkpoint_dir / 'codebook_streaming.pt').exists(),
            5: (checkpoint_dir / 'pattern_matrix.pt').exists(),
            6: (checkpoint_dir / 'target_pool' / 'faiss.index').exists() or
               (checkpoint_dir / 'target_pool' / 'faiss_trained.index').exists(),
        }

        return status

    def print_status(self):
        """打印各步骤状态"""
        status = self.get_status()

        print("\n" + "=" * 70)
        print(" " * 18 + "Offline Pipeline 状态 (Dual GPU)")
        print("=" * 70)

        for idx, (step_name, _, step_desc) in enumerate(self.STEPS):
            step_num = idx + 1
            done = status.get(step_num, False)
            mark = "✓" if done else "○"
            print(f"  {mark} Step {step_num}: {step_desc}")

        print("=" * 70)

        completed = sum(status.values())
        print(f"\n进度: {completed}/6 步完成")

        if completed < 6:
            for i in range(1, 7):
                if not status.get(i, False):
                    print(f"建议运行: python scripts/run_offline.py --config configs/dual_4090.yaml --step {i}")
                    break
        else:
            print("✓ 所有步骤已完成！可以进行 Online 推理")
        print()


def run_offline_pipeline_dual_gpu(config: Dict, **kwargs) -> Dict:
    """便捷入口函数 (双 GPU 版本)"""
    runner = DualGPUOfflineRunner(config)
    return runner.run(**kwargs)
