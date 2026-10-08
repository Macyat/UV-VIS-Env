"""curve_resolution：多变量曲线分辨（MCR）。

从混合光谱中分离出各组分的浓度剖面与光谱。

- initializers  最纯变量检测：SIMPLISMA / OPA
- mcr_als       MCR-ALS：初值估计 + 约束（非负/闭合/单峰）+ 解的不确定性评估
"""
from .initializers import simplisma, orthogonal_projection
from .mcr_als import MCRALS
from .afs import afs_lawton_sylvestre

__all__ = ["simplisma", "orthogonal_projection", "MCRALS", "afs_lawton_sylvestre"]
