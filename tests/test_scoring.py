"""打分表（scoring table）排名与 CSV 往返测试。"""
from aimeta.pipelines.scoring import (
    rank_models, write_scoring_csv, read_scoring_csv, SCORING_COLUMNS,
)


def _row(model_type, **kw):
    base = dict(
        model_type=model_type,
        alarm_acc=0.9, alarm_err=0.1, mape=0.1, r2_score=0.9,
        pearson_r_score=0.9, daily_r2_score=0.9, daily_pearson_r_score=0.9,
        rmse=0.1,
        **{"rate of reaching the standard": 0.9, "bad grouped ratio": 0.1,
           "durbin_watson_value": 2.0, "Kurtosis": 0.0, "skew": 0.0,
           "LM_p": 0.5, "F_p": 0.5},
    )
    base.update(kw)
    return base


def test_rank_picks_best_overall():
    good = _row("good")
    bad = _row("bad", alarm_acc=0.5, alarm_err=0.5, mape=0.5, r2_score=0.5,
               pearson_r_score=0.5, daily_r2_score=0.5,
               daily_pearson_r_score=0.5, rmse=0.5,
               **{"rate of reaching the standard": 0.5, "bad grouped ratio": 0.5,
                  "durbin_watson_value": 1.0})
    ranked = rank_models([bad, good])
    assert ranked[0]["model_type"] == "good"
    assert ranked[0]["rank"] < ranked[1]["rank"]


def test_identical_models_tiebreak_sum():
    # 两行指标完全相同：序位类分量（ordinal / r2 的正值段）会把并列拆成 0/1，
    # 其余竞赛分量并列取 0，因此两行 rank 之和恰为序位类分量数。
    from aimeta.pipelines.scoring import RANK_COMPONENTS
    a, b = _row("a"), _row("b")
    ranked = rank_models([a, b])
    n_ord = sum(1 for _, _, m in RANK_COMPONENTS if m in ("ordinal", "r2"))
    ranks = sorted(r["rank"] for r in ranked)
    assert abs((ranks[0] + ranks[1]) - n_ord) < 1e-9
    assert 0.0 <= ranks[0] <= n_ord
    assert 0.0 <= ranks[1] <= n_ord


def test_durbin_watson_abs2_direction():
    near = _row("near", durbin_watson_value=2.0)
    far = _row("far", durbin_watson_value=0.5)
    ranked = rank_models([far, near])
    assert ranked[0]["model_type"] == "near"


def test_negative_r2_gets_leveraged():
    # 负 R² 应按幅度加杠杆：R²=-10 与 R²=-0.1 不能只差一档（纯序位），前者应被显著重罚。
    from aimeta.pipelines.scoring import _r2_leveraged_ranks
    r = _r2_leveraged_ranks([0.9, -0.1, -10.0])
    assert r[0] < r[1] < r[2]          # 0.9 最优，-10 最差
    assert r[2] - r[1] > 1.0           # 幅度杠杆使差距大于纯序位的 1 档
    # 正 R² 不产生杠杆惩罚，仍为纯序位
    r2 = _r2_leveraged_ranks([0.5, 0.9])
    assert r2[1] == 0.0 and r2[0] == 1.0


def test_csv_roundtrip(tmp_path):
    rows = [_row("a"), _row("b", mape=0.3)]
    p = tmp_path / "metrics.csv"
    write_scoring_csv(rows, p)
    back = read_scoring_csv(p)
    assert len(back) == 2
    assert "rank" in back[0]
    assert set(SCORING_COLUMNS).issubset(back[0].keys())
    assert back[0]["rank"] <= back[1]["rank"]


def test_merge_keeps_better_mape(tmp_path):
    p = tmp_path / "metrics.csv"
    write_scoring_csv([_row("a", mape=0.4)], p)
    write_scoring_csv([_row("a", mape=0.5, r2_score=0.99)], p, merge=True)
    assert read_scoring_csv(p)[0]["mape"] == 0.4
    write_scoring_csv([_row("a", mape=0.2, r2_score=0.99)], p, merge=True)
    back = read_scoring_csv(p)
    assert back[0]["mape"] == 0.2
    assert back[0]["r2_score"] == 0.99
