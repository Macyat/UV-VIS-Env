"""stats：统计工具箱。

- hypothesis   假设检验与一致性分析：Cochran / Bartlett / Levene / Grubbs / D'Agostino
- intervals    置信区间与容忍区间（b-content / b-expectation）
"""
from .hypothesis import (
    cochran_test, bartlett_test, levene_test, grubbs_test,
    normality_test, consistency_report,
)
from .intervals import (
    confidence_interval_mean, tolerance_interval, prediction_interval,
)

__all__ = [
    "cochran_test", "bartlett_test", "levene_test", "grubbs_test",
    "normality_test", "consistency_report",
    "confidence_interval_mean", "tolerance_interval", "prediction_interval",
]
