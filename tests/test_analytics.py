"""Run with:  python tests/test_analytics.py   (pytest also works)."""
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import analytics as A  # noqa: E402

CSV = next((ROOT / "data").glob("*.csv"))


def load():
    p = A.prepare(A.read_csv(CSV))
    return p, A.derive(p.df)


def test_loads_all_rows_and_no_cleaning():
    p, d = load()
    assert len(d) == 1696 and p.notes == []


def test_channel_totals_match_independent_sum():
    _, d = load()
    ct = A.channel_totals(d).set_index("Channel")
    raw = pd.read_csv(CSV)
    assert ct.loc["Uber Eats", "Orders"] == raw["UberEatsOrders"].sum() == 800353
    assert abs(ct["Share"].sum() - 1) < 1e-12
    assert ct.loc["Uber Eats", "Rank"] == 1 and ct.loc["In-Store", "Rank"] == 4


def test_shares_sum_to_one_per_row_and_by_group():
    _, d = load()
    assert np.allclose(d[list(A.SHARE_COL.values())].sum(axis=1), 1)
    for col in ("Subregion", "CuisineType", "Segment"):
        assert np.allclose(A.share_matrix(d, col).sum(axis=1), 1)
        assert np.allclose(A.share_matrix(d, col, A.DELIVERY).sum(axis=1), 1)


def test_in_store_total_basis():
    _, d = load()
    m = A.share_matrix(d, "Segment", ["In-Store"], basis="total")
    raw = pd.read_csv(CSV).groupby("Segment")[["InStoreOrders", "MonthlyOrders"]].sum()
    assert np.allclose(m["In-Store"], raw["InStoreOrders"] / raw["MonthlyOrders"])


def test_diversification_bounds_and_extremes():
    row = pd.DataFrame(
        {"RestaurantID": [1, 2], "RestaurantName": ["a", "b"], "CuisineType": ["x", "x"], "Segment": ["s", "s"],
         "Subregion": ["r", "r"], "InStoreOrders": [100, 25], "UberEatsOrders": [0, 25],
         "DoorDashOrders": [0, 25], "SelfDeliveryOrders": [0, 25]}
    )
    d = A.derive(A.prepare(row).df)
    assert abs(d.loc[0, "Diversification"] - 0) < 1e-6      # single channel
    assert abs(d.loc[1, "Diversification"] - 1) < 1e-9      # perfectly even
    _, real = load()
    assert real["Diversification"].between(0, 1).all()


def test_risk_thresholds_move_flags():
    p, _ = load()
    spec = A.derive(p.df, A.RiskThresholds(high_combined=0.70, high_single=1.01))
    strict = A.derive(p.df, A.RiskThresholds(high_combined=0.60, high_single=1.01))
    assert (spec["RiskTier"] == "High risk").sum() == 603            # brief's 70% rule on combined aggregators
    assert (strict["RiskTier"] == "High risk").sum() > 603
    # no restaurant relies on ONE platform for 70%
    assert (spec["MaxAggShare"] >= 0.70).sum() == 0


def test_validation_report_detects_instore_share_mismatch():
    p, _ = load()
    v = A.validation_report(p.df, p.notes).set_index("Check")
    assert v.loc["Channel orders sum to MonthlyOrders", "Status"] == "Pass"
    assert v.loc["Delivery shares (UE+DD+SD) sum to ~100%", "Status"] == "Pass"
    assert v.loc["Revenue = orders x AOV", "Status"] == "Pass"
    assert v.loc["InStoreShare column matches order counts", "Status"] == "Warning"


def test_new_entries_and_new_categories_flow_through():
    p, _ = load()
    new = pd.DataFrame(
        {"RestaurantID": [99001, 99002], "RestaurantName": ["Harbour Ramen", "Hamilton Hot Pot"],
         "CuisineType": ["Korean", "Korean"], "Segment": ["Cafe", "Cafe"], "Subregion": ["Waiheke", "Waiheke"],
         "InStoreOrders": [300, 200], "UberEatsOrders": [200, 100], "DoorDashOrders": [100, 100], "SelfDeliveryOrders": [50, 50],
         "MonthlyOrders": [650, 450]}
    )
    merged = A.derive(A.prepare(A.merge_new_entries(p.df, new)).df)
    assert len(merged) == 1698
    assert "Waiheke" in set(A.dominance_table(merged, "Subregion")["Subregion"])
    assert "Korean" in set(A.orders_by(merged, "CuisineType").index)
    # re-uploading the same IDs must not double count
    again = A.prepare(A.merge_new_entries(merged, new)).df
    assert len(again) == 1698


def test_cleaning_drops_bad_rows_and_dedupes():
    raw = pd.read_csv(CSV).head(5).copy()
    raw.loc[0, "UberEatsOrders"] = -4
    raw.loc[1, "Subregion"] = None
    raw = pd.concat([raw, raw.iloc[[2]]], ignore_index=True)
    p = A.prepare(raw)
    assert len(p.df) == 3 and len(p.notes) == 2  # 5 rows - 2 bad = 3; duplicate resolved


def test_aliases_and_missing_columns():
    raw = pd.read_csv(CSV).head(3).rename(columns={"InStoreOrders": "InStoreOrdersCount"})
    df = A.read_csv(io.BytesIO(raw.to_csv(index=False).encode()))
    assert "InStoreOrders" in df.columns
    try:
        A.prepare(df.drop(columns=["Subregion"]))
        raise AssertionError("expected DataError")
    except A.DataError as e:
        assert "Subregion" in str(e)


def test_profit_and_projection():
    _, d = load()
    ps = A.profit_summary(d).set_index("Channel")
    assert abs(ps["Share of net profit"].sum() - 1) < 1e-9
    assert ps.loc["In-Store", "% loss-making restaurants"] == 0
    pr = A.projection(d, 6)
    assert len(pr) == 7 and abs(pr.loc[0, "Total"] - d["TotalOrders"].sum()) < 1e-6
    assert pr.loc[6, "Total"] > pr.loc[0, "Total"]


def test_insights_text():
    _, d = load()
    txt = A.insights(d)
    assert len(txt) >= 5 and "Uber Eats is the largest channel" in txt[0]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print(f"\n{len(fns)} tests passed")
