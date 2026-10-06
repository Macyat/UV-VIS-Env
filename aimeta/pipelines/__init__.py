"""pipelines：把「脚本」变成「可复现的流程」。

- train   SpectrumSet + 预处理链 + 模型 → ModelCard（含 FOM）
- infer   ModelCard + 光谱 → 带上下限保护的预测

老的 ``run.py`` 靠 subprocess 拼命令行 fork ``Train.py``，
参数靠十几个命令行开关传递；这里改成配置驱动的函数调用。
"""
from .train import train_model, sweep
from .infer import predict

__all__ = ["train_model", "sweep", "predict"]
