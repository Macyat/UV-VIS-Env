"""配置加载：把「改阈值要动 Python」变成「改 YAML」。

三份配置对应三类本来硬编码在代码里的东西：

    configs/params.yaml              老 preprocessing.get_configs() 的六个 elif
    configs/preprocessing/*.yaml     散落五处的 Wiener→SG→小波→SNV
    configs/instruments/*.yaml       每台工控机的档案与迁移参数
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ..metrics.water_standards import WaterParam

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "configs"


class ConfigError(Exception):
    pass


def _find(root: Optional[Path], name: str) -> Path:
    root = Path(root) if root is not None else DEFAULT_ROOT
    p = root / name
    if not p.exists():
        raise ConfigError(f"config not found: {p}")
    return p


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_params(root: Optional[Path] = None, name: str = "params.yaml"
                ) -> Dict[str, WaterParam]:
    """加载全部水质参数的计量定义。"""
    d = load_yaml(_find(root, name))
    params = d.get("params", d)
    return {k: WaterParam.from_dict(k, v) for k, v in params.items()}


def load_chain(root: Optional[Path] = None,
               name: str = "preprocessing/dayu_v1.yaml") -> List[Dict[str, Any]]:
    """加载预处理链配置。"""
    d = load_yaml(_find(root, name))
    chain = d.get("chain")
    if not chain:
        raise ConfigError(f"no 'chain' in {name}")
    return chain


def load_instrument(instrument_id: str, root: Optional[Path] = None) -> Dict[str, Any]:
    """加载某台仪器的档案（波长轴、光程、迁移配置）。"""
    d = load_yaml(_find(root, f"instruments/{instrument_id}.yaml"))
    d.setdefault("instrument_id", instrument_id)
    return d
