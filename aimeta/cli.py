"""统一命令行入口：取代「run.py fork Train.py」的脚本式调用。

    python -m aimeta.cli list-models
    python -m aimeta.cli list-ops
    python -m aimeta.cli smoke              # 端到端自检（合成数据）
    python -m aimeta.cli transfer --slave ai17 --master ai14 --method auto
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# 导入即注册：模型 / 预处理 / 迁移 / 变量选择的实现都在子包里注册
import aimeta.models            # noqa: F401
import aimeta.preprocessing     # noqa: F401
import aimeta.selection         # noqa: F401
import aimeta.transfer          # noqa: F401

from .core.registry import MODELS, PREPROC, TRANSFER
from .io.config import load_instrument

# edge 是独立放在仓库根目录的极简运行时（便于单独拷到工控机），不在 aimeta 包内
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _cmd_list_models(args) -> int:
    fam = args.family
    for k, obj, meta in MODELS.items():
        if fam and meta.get("family") != fam:
            continue
        print(f"  {k:14s} family={meta.get('family','-'):8s}")
    print(f"\nfamilies: {', '.join(MODELS.families())}")
    return 0


def _cmd_list_ops(args) -> int:
    for k, obj, meta in PREPROC.items():
        comp = "edge-ok" if meta.get("compilable") else "train-only"
        print(f"  {k:14s} {meta.get('family','-'):10s} {comp}")
    return 0


def _cmd_smoke(args) -> int:
    """端到端自检：合成数据 → 训练 → 导出 edge → 一致性校验 → 监控。"""
    rng = np.random.default_rng(0)
    n, p = 120, 101
    wl = np.linspace(200, 700, p)
    S = np.vstack([
        np.exp(-((wl - 260) ** 2) / (2 * 20 ** 2)),
        np.exp(-((wl - 340) ** 2) / (2 * 30 ** 2)),
        np.exp(-((wl - 520) ** 2) / (2 * 40 ** 2)),
    ])
    C = rng.random((n, 3)) * np.array([1.0, 0.5, 0.3])
    X = C @ S + rng.normal(0, 1e-3, (n, p))
    y = C[:, 0] * 2.0 + 0.1

    from .core.spectra import SpectrumSet
    from .io.config import load_params
    from .pipelines.train import train_model
    from .pipelines.infer import predict
    from .monitoring import MSPC
    from edge.runtime import export_card, load_edge_model, verify_against_card

    chain = [{"op": "savgol", "window": 11, "polyorder": 3}, {"op": "snv"}]
    spectra = SpectrumSet(X=X, wavelengths=wl, y=y)
    params = load_params()
    card = train_model(spectra, "AN", "pls", chain=chain,
                       model_params={"n_components": 5},
                       param_def=params["AN"], instrument_id="ai14")
    print(f"[train] {card!r}")

    pred = predict(card, X)
    print(f"[infer] rmse = {np.sqrt(np.mean((y - pred)**2)):.5f}")

    d = export_card(card, "_smoke_model", X_reference=X)
    em = load_edge_model(d)
    ok, diff = verify_against_card(em, card, X)
    print(f"[edge ] parity={'OK' if ok else 'FAIL'} max_diff={diff:.2e}")

    mspc = MSPC(n_components=3).fit(X)
    bad = X.copy(); bad[0] += 0.05
    r = mspc.monitor(bad)
    print(f"[mspc ] 异常样本 SPE={r['SPE'][0]:.4f} vs 限={r['spe_limit']:.4f} "
          f"-> {'报警' if r['spe_alarm'][0] else '正常'}")
    return 0 if ok else 1


def _cmd_transfer(args) -> int:
    """按仪器配置推荐迁移方法（真实拟合需要标准化样本，此处只做诊断与配置展示）。"""
    cfg = load_instrument(args.slave)
    print(f"slave = {args.slave}  master = {cfg.get('transfer', {}).get('master')}")
    print(f"configured method = {cfg.get('transfer', {}).get('method')}")
    print(f"fallback = {cfg.get('transfer', {}).get('fallback')}")
    if args.method and args.method != "auto":
        print(f"override -> {args.method}")
    print("\n提示：需要 (X_slave, X_master) 标准化样本才能拟合，"
          "可用同一套标液在两台机上各测一遍得到。")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="aimeta", description="UV-Vis 水质软测量算法库")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("list-models"); p1.add_argument("--family", default=None)
    p1.set_defaults(func=_cmd_list_models)

    sub.add_parser("list-ops").set_defaults(func=_cmd_list_ops)
    sub.add_parser("smoke").set_defaults(func=_cmd_smoke)

    p2 = sub.add_parser("transfer")
    p2.add_argument("--slave", required=True)
    p2.add_argument("--master", default=None)
    p2.add_argument("--method", default="auto")
    p2.set_defaults(func=_cmd_transfer)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
