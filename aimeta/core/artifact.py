"""ModelCard：模型产物 = 估计器 + 可复现的预处理链 + 计量指纹。

设计动机：模型要能跨仪器迁移，前提是**知道自己是被怎么造出来的**。
裸 ``.pkl`` 无法回答「这条模型用的是哪条预处理链、哪台仪器、哪个波长轴」。
"""
from __future__ import annotations

import hashlib
import json
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

CARD_VERSION = 1


@dataclass
class ModelCard:
    """一个可交付的水质软测量模型。"""

    model_key: str                        # 注册表中的模型名，如 "pls"
    label: str                            # 水质参数，如 "TN"
    estimator: Any                        # 拟合好的估计器
    preproc_chain: List[Dict[str, Any]]   # 预处理链（YAML 可复现）
    wavelengths: List[float]              # 波长轴（部署端必须逐点匹配）
    instrument_id: str = "unknown"        # 训练该模型所用仪器
    data_version: str = "unknown"         # 训练数据版本 / 哈希
    fom: Dict[str, float] = field(default_factory=dict)   # 品质因数
    params: Dict[str, Any] = field(default_factory=dict)  # 模型超参
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))
    version: int = CARD_VERSION

    # ---- 指纹 ----
    def fingerprint(self) -> str:
        """预处理链 + 波长轴 + 模型名的哈希。

        部署端加载后必须校验指纹一致，否则说明「训练时用的链」与「部署时跑的链」不同。
        """
        payload = json.dumps(
            {
                "model_key": self.model_key,
                "preproc_chain": self.preproc_chain,
                "wavelengths": [round(float(w), 6) for w in self.wavelengths],
                "version": self.version,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    # ---- 序列化 ----
    def describe(self) -> Dict[str, Any]:
        """不含估计器的描述字典（可写 JSON / 打印 / 入库）。"""
        return {
            "model_key": self.model_key,
            "label": self.label,
            "preproc_chain": self.preproc_chain,
            "wavelengths": [float(w) for w in self.wavelengths],
            "instrument_id": self.instrument_id,
            "data_version": self.data_version,
            "fom": self.fom,
            "params": self.params,
            "created_at": self.created_at,
            "version": self.version,
            "fingerprint": self.fingerprint(),
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        return path

    @staticmethod
    def load(path: str | Path) -> "ModelCard":
        with open(path, "rb") as f:
            return pickle.load(f)

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        rmse = self.fom.get("rmse_cv", float("nan"))
        return (
            f"<ModelCard {self.label}/{self.model_key} "
            f"@{self.instrument_id} fp={self.fingerprint()} rmse_cv={rmse:.4g}>"
        )


def data_fingerprint(X: np.ndarray, y: Optional[np.ndarray] = None) -> str:
    """训练数据版本指纹，写进 ModelCard.data_version。"""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(np.asarray(X, dtype=np.float64)).tobytes())
    if y is not None:
        h.update(np.ascontiguousarray(np.asarray(y, dtype=np.float64)).tobytes())
    return h.hexdigest()[:16]
