"""注册表：可插拔能力的地基。"""
import numpy as np
import pytest

from aimeta.core.registry import Registry


def test_register_and_build():
    r = Registry("demo")

    @r.register("a", family="x")
    class A:
        def __init__(self, n=1):
            self.n = n

    assert "a" in r
    assert r.build("a", n=3).n == 3
    assert r.keys() == ["a"]
    assert r.meta("a")["family"] == "x"


def test_family_filter():
    r = Registry("demo2")

    @r.register("p", family="latent")
    class P: pass

    @r.register("q", family="linear")
    class Q: pass

    assert r.keys("latent") == ["p"]
    assert r.families() == ["latent", "linear"]


def test_duplicate_key_rejected():
    r = Registry("demo3")

    @r.register("dup")
    class A: pass

    with pytest.raises(KeyError):
        @r.register("dup")
        class B: pass


def test_global_registries_are_populated():
    """导入子包后，全局注册表必须非空 —— 防「忘了 import 导致注册没发生」。"""
    import aimeta.models  # noqa: F401
    import aimeta.preprocessing  # noqa: F401
    import aimeta.transfer  # noqa: F401
    from aimeta.core.registry import MODELS, PREPROC, TRANSFER

    assert "pls" in MODELS and "ridge" in MODELS
    assert "snv" in PREPROC and "savgol" in PREPROC
    assert "pds" in TRANSFER and "ds" in TRANSFER and "sbc" in TRANSFER
