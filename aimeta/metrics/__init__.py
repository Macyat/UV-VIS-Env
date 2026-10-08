"""metrics：计量验收。

- figures_of_merit   灵敏度 / 分析灵敏度 / 选择性 / **多元 LOD、LOQ** / 预测不确定度
- water_standards    类别判定、跨类别误判矩阵、daily R²、报警准确率

LOD / LOQ 由回归向量与空白噪声算出；多元校正情形下不使用单变量的 3σ 公式。
"""
from .figures_of_merit import (
    regression_vector, sensitivity, analytical_sensitivity,
    selectivity, figures_of_merit,
)
from .water_standards import (
    WaterParam, classify, misclassification_matrix,
    daily_r2, alarm_accuracy, acceptance_report,
)
from .preprocessing_eval import evaluate_preprocessing

__all__ = [
    "regression_vector", "sensitivity", "analytical_sensitivity",
    "selectivity", "figures_of_merit",
    "WaterParam", "classify", "misclassification_matrix",
    "daily_r2", "alarm_accuracy", "acceptance_report",
    "evaluate_preprocessing",
]
