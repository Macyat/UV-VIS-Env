"""edge：工控机侧极简运行时。

现场工控机可能没有完整 sklearn/scipy 环境，因此本包**只依赖 numpy**，
支持线性模型（PLS / Ridge / Lasso / OLS / PCR，本质 y = X@coef + b）
与可编译成矩阵的预处理算子。

Wiener 自适应滤波与小波去噪无法写成矩阵，部署链中请用 savgol / deriv_gram 替代。

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
