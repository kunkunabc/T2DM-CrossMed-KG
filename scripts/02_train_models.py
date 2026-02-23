# python 02_train_models.py
import logging
import pprint
import torch
import os
import time
import math
import numpy as np
from pathlib import Path
from typing import Dict, Optional

from ruamel import yaml
from pykeen.pipeline import pipeline
from pykeen.models import Model
from pykeen.training.callbacks import TrainingCallback
from pykeen.evaluation import RankBasedEvaluator
from pykeen.triples import TriplesFactory

# === 平台 & 根目录 ===
AUTODL_PLATFORM = os.path.exists("/root/autodl-tmp/DM_Project")
PERSISTENT_ROOT = Path("/root/autodl-tmp/DM_Project") if AUTODL_PLATFORM else Path(__file__).parent.parent.resolve()

# === 日志配置 ===
LOG_PATH = PERSISTENT_ROOT / "training.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)


# === 动态早停回调 ===
class EarlyStoppingCallback(TrainingCallback):
    def __init__(self, patience: int = 5, delta: float = 0.001, metric: str = "Hits@10"):
        super().__init__()
        self.patience = patience
        self.delta = delta
        self.metric = metric
        self.best_value = -np.inf
        self.counter = 0
        self.early_stop = False

    def on_epoch_end(self, epoch: int, **kwargs):
        current_value = kwargs.get(self.metric, None)
        if current_value is None:
            logger.warning(f"无法获取指标 {self.metric}，早停未启用")
            return

        if (current_value - self.best_value) > self.delta:
            self.best_value = current_value
            self.counter = 0
            logger.info(f"✅ 指标提升至 {self.best_value:.4f}，重置早停计数器")
        else:
            self.counter += 1
            logger.info(f"⏳ 指标未提升，早停计数器: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                logger.warning(f"🚨 早停触发，停止训练！最佳 {self.metric} = {self.best_value:.4f}")

    def should_stop(self) -> bool:
        return self.early_stop


# === 优化后的检查点回调 ===
class PlatformCheckpointCallback(TrainingCallback):
    def __init__(self, model_name: str, results_dir: Path, metric: str = "Hits@10"):
        super().__init__()
        self.ckpt_dir = results_dir / model_name / "checkpoints"
        # 强制创建目录并设置权限
        self.ckpt_dir.mkdir(parents=True, exist_ok=True, mode=0o755)
        logger.info(f"检查点目录: {self.ckpt_dir.absolute()} (权限: {oct(os.stat(self.ckpt_dir).st_mode & 0o777)})")
        self.best_metric = -np.inf
        self.metric = metric

    def on_epoch_end(self, epoch: int, **kwargs):
        current_value = kwargs.get(self.metric, None)
        if current_value is None:
            logger.warning(f"未收到指标 {self.metric}，跳过检查点保存")
            return

        logger.info(f"[{self.metric}] 当前值: {current_value:.4f}, 最佳值: {self.best_metric:.4f}")
        if current_value > self.best_metric + 1e-6:  # 避免浮点误差
            self.best_metric = current_value
            model = kwargs["model"]
            path = self.ckpt_dir / f"best_{self.metric.replace('@', '_')}.pt"
            try:
                # 保存完整模型（包括结构和参数）
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "metric": self.metric,
                    "value": current_value
                }, path)
                logger.info(f"✅ 保存最佳模型检查点: {path}")
                if AUTODL_PLATFORM:
                    os.system(f"autodl sync {self.ckpt_dir} > /dev/null 2>&1")
            except Exception as e:
                logger.error(f"保存失败: {e}")


# === 加载配置 ===
def load_config() -> tuple[Dict, Path]:
    cfg_path = PERSISTENT_ROOT / "configs" / "default.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"配置文件未找到: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.YAML(typ="safe", pure=True).load(f)
    cfg["training"]["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"✔ 配置加载成功:\n{pprint.pformat(cfg)}")
    return cfg, PERSISTENT_ROOT


# === 单模型训练流程 ===
def train_model(model_name: str, cfg: Dict, project_root: Path) -> Dict:
    logger.info(f"\n=== 开始训练：{model_name} ===")
    splits_dir = project_root / "splits"
    results_dir = project_root / "results"
    model_dir = results_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True, mode=0o755)
    logger.info(f"模型目录: {model_dir.absolute()}")

    # 动态选择损失函数
    loss_config = {
        "ComplEx": ("marginranking", {"margin": 6.0}),
        "DistMult": ("bcewithlogits", {}),
        "RotatE": ("marginranking", {"margin": 6.0}),
    }
    loss_name, loss_kwargs = loss_config.get(model_name, ("bcewithlogits", {}))

    # 初始化回调
    early_stop_callback = EarlyStoppingCallback(
        patience=cfg["early_stopping"]["patience"],
        delta=cfg["early_stopping"]["delta"],
        metric=cfg["early_stopping"]["monitor_metric"],
    )
    checkpoint_callback = PlatformCheckpointCallback(
        model_name, results_dir, metric=cfg["early_stopping"]["monitor_metric"],
    )

    # === 新增：临时启用 PyKEEN 详细日志 ===
    # 保存当前日志级别
    original_level = logging.getLogger("pykeen").level

    # 设置 PyKEEN 日志级别为 INFO 以显示详细参数
    logging.getLogger("pykeen").setLevel(logging.INFO)
    logging.getLogger("pykeen.pipeline").setLevel(logging.INFO)
    # === 结束新增部分 ===

    # 构造训练参数
    pipeline_args = {
        "training": str(splits_dir / "train.tsv"),
        "validation": str(splits_dir / "valid.tsv"),
        "testing": str(splits_dir / "test.tsv"),
        "model": model_name,
        "model_kwargs": {"embedding_dim": cfg["training"]["embedding_dim"]},
        "dataset_kwargs": {
            "create_inverse_triples": False  # 添加反向三元组
        },
        "loss": loss_name,
        "loss_kwargs": loss_kwargs,
        "optimizer": "Adam",
        "optimizer_kwargs": {
            "lr": cfg["training"]["learning_rate"],
            "weight_decay": cfg["training"]["regularization_coef"],
        },
        "negative_sampler": "bernoulli",  # 伪类型负采样器 其他负采样器 "bernoulli"
        "negative_sampler_kwargs": {"num_negs_per_pos": cfg["training"]["neg_per_pos"]},
        "training_kwargs": {
            "num_epochs": cfg["training"]["num_epochs"],
            "batch_size": cfg["training"]["batch_size"],
            "callbacks": [early_stop_callback, checkpoint_callback],
        },
        "evaluator_kwargs": {
            "filtered": True,
            "batch_size": cfg["evaluation"]["batch_size"],
            "slice_size": cfg["evaluation"]["slice_size"],
        },
        "random_seed": 42,
        "device": cfg["training"]["device"],
        "use_tqdm": True,
    }

    try:
        start = time.time()
        result = pipeline(**pipeline_args)

        # 显式保存最终模型和评估结果
        result.save_to_directory(model_dir)
        logger.info(f"最终模型保存到: {model_dir.absolute()}")

        return {"status": "completed"}
    except Exception as e:
        logger.error(f"训练失败: {e}")
        return {"error": str(e)}
    finally:
        # === 新增：恢复原始日志级别 ===
        logging.getLogger("pykeen").setLevel(original_level)
        logging.getLogger("pykeen.pipeline").setLevel(original_level)
        # === 结束新增部分 ===


# === 主程序 ===
if __name__ == "__main__":
    cfg, project_root = load_config()
    all_metrics = {}
    for name in ["RotatE"]:  # "DistMult","ComplEx","RotatE"
        all_metrics[name] = train_model(name, cfg, project_root)
    logger.info("=== 所有模型训练完成 ===")