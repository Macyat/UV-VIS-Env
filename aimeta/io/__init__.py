"""io：数据接入与配置加载。

- config  YAML 配置 → 参数定义 / 预处理链 / 仪器档案
- arff    工控机侧 ARFF 光谱 → SpectrumSet
"""
from .config import load_yaml, load_params, load_chain, load_instrument, ConfigError
from .arff import read_arff

__all__ = ["load_yaml", "load_params", "load_chain", "load_instrument",
           "ConfigError", "read_arff"]
