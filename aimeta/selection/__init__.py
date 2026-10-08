"""selection：变量 / 波长选择。

- base       Selector 协议 + 置换检验
- cars       竞争性自适应重加权采样（CARS）
"""
from .base import SelectorBase, permutation_test
from .cars import CARS
from .uve import UVE
from .design import diagonal_design

__all__ = ["SelectorBase", "permutation_test", "CARS", "UVE", "diagonal_design"]
