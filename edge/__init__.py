"""edge：工控机侧部署运行时。

支持线性模型（PLS / Ridge / Lasso / OLS / PCR，本质 y = X@coef + b）
与可编译成矩阵的预处理算子（savgol / snv / msc / 缩放）。

部署端依赖与 aimeta 一致（numpy / scipy / PyWavelets）：
- 纯 numpy 可编译算子构成的链（如 dayu_edge.yaml）可零额外依赖运行；
- wavelet 等需外部库的算子会委托给注册算子执行，需目标机装好对应依赖
  （aimeta 已将 scipy / PyWavelets 列为主依赖，``pip install aimeta`` 即可）。

本包独立于 aimeta 放在仓库根目录，便于单独拷贝到工控机。
"""
from .runtime import (
    LinearEdgeModel, compile_chain, sg_matrix, CompileError,
    load_edge_model, export_card, verify_against_card,
)

__all__ = [
    "LinearEdgeModel", "compile_chain", "sg_matrix", "CompileError",
    "load_edge_model", "export_card", "verify_against_card",
]
