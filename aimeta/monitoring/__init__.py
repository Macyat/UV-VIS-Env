"""monitoring：在线监控。

用于判断在线预测是否仍处于受控状态，及时发现探头污染、季节漂移等异常。

- control_charts  Shewhart / CUSUM / EWMA + ARL + 自相关修正
- mspc            PCA 的 T² / SPE + 贡献图（把异常定位到波长）
"""
from .control_charts import (
    shewhart_limits, cusum, ewma, arl0, effective_sample_size,
)
from .mspc import MSPC

__all__ = ["shewhart_limits", "cusum", "ewma", "arl0",
           "effective_sample_size", "MSPC"]
