"""SkyCity Auckland - Order Channel Performance & Market Share analytics engine.

Pure pandas/numpy (no UI imports) so it can be unit-tested, re-used by the
Streamlit app, and by the report builder.  Everything is derived from the
*order counts* in the CSV, so new rows, new cuisines, new subregions or new
segments flow through automatically.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
CHANNELS = ["In-Store", "Uber Eats", "DoorDash", "Self-Delivery"]
DELIVERY = CHANNELS[1:]
AGGREGATORS = ["Uber Eats", "DoorDash"]

ORDER_COL = {
    "In-Store": "InStoreOrders",
    "Uber Eats": "UberEatsOrders",
    "DoorDash": "DoorDashOrders",
    "Self-Delivery": "SelfDeliveryOrders",
}
REV_COL = {
    "In-Store": "InStoreRevenue",
    "Uber Eats": "UberEatsRevenue",
    "DoorDash": "DoorDashRevenue",
    "Self-Delivery": "SelfDeliveryRevenue",
}
PROFIT_COL = {
    "In-Store": "InStoreNetProfit",
    "Uber Eats": "UberEatsNetProfit",
    "DoorDash": "DoorDashNetProfit",
    "Self-Delivery": "SelfDeliveryNetProfit",
}
SHARE_COL = {  # derived share-of-total-orders columns
    "In-Store": "Share_InStore",
    "Uber Eats": "Share_UberEats",
    "DoorDash": "Share_DoorDash",
    "Self-Delivery": "Share_SelfDelivery",
}

# Friendly dimension name -> column name
DIMENSIONS = {"Subregion": "Subregion", "Cuisine": "CuisineType", "Segment": "Segment"}

# Alternative header spellings (e.g. from the project brief) -> canonical names
ALIASES = {
    "InStoreOrdersCount": "InStoreOrders",
    "UberEatsOrdersCount": "UberEatsOrders",
    "DoorDashOrdersCount": "DoorDashOrders",
    "SelfDeliveryOrdersCount": "SelfDeliveryOrders",
    "DeliveryCostOrder": "DeliveryCostPerOrder",
    "Cuisine": "CuisineType",
}

REQUIRED = ["RestaurantID", "RestaurantName", "CuisineType", "Segment", "Subregion", *ORDER_COL.values()]

# Documented ranges from the project brief (used for outlier transparency only)
DOCUMENTED_RANGES = {
    "GrowthFactor": (0.99, 1.05),
    "AOV": (29.79, 47.23),
    "COGSRate": (0.20, 0.40),
    "OPEXRate": (0.20, 0.55),
    "DeliveryRadiusKM": (3, 18),
    "DeliveryCostPerOrder": (0.89, 5.31),
}

RISK_TIERS = ["High risk", "Elevated", "Balanced"]

# Diversification score (normalised Shannon entropy, 0..1) benchmark bands.
# Every restaurant here uses all four channels, so scores are compressed into ~0.75-0.98;
# the cut-offs below are calibrated to that spread (roughly top ~40% / middle ~45% / bottom ~15%).
# Re-calibrate if your data changes materially.
DIVERSIFICATION_BANDS = [
    (0.95, "Well diversified"),
    (0.88, "Moderately diversified"),
    (0.00, "Concentrated"),
]


@dataclass
class RiskThresholds:
    """Dependency-risk rules.  All values are fractions of a restaurant's TOTAL orders."""

    high_combined: float = 0.70   # Uber Eats + DoorDash together  (brief: 70% reliance)
    high_single: float = 0.50     # any single aggregator
    elevated_combined: float = 0.60


class DataError(ValueError):
    """Raised when the uploaded file cannot be used."""


@dataclass
class Prepared:
    df: pd.DataFrame
    notes: list[str] = field(default_factory=list)  # data-cleaning actions taken


# --------------------------------------------------------------------------- #
# Loading & cleaning
# --------------------------------------------------------------------------- #
def read_csv(source) -> pd.DataFrame:
    """Read a CSV from a path, bytes or file-like object (utf-8, with latin-1 fallback)."""
    if isinstance(source, (str, Path)):
        raw = Path(source).read_bytes()
    elif isinstance(source, (bytes, bytearray)):
        raw = bytes(source)
    else:
        raw = source.read()
        if isinstance(raw, str):
            raw = raw.encode("utf-8")
    for enc in ("utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=enc)
            break
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError as exc:
            raise DataError("The file is empty.") from exc
    else:  # pragma: no cover
        raise DataError("Could not decode the file as text.")
    df.columns = [str(c).strip() for c in df.columns]
    return df.rename(columns={k: v for k, v in ALIASES.items() if k in df.columns and v not in df.columns})


def prepare(df: pd.DataFrame) -> Prepared:
    """Validate required columns, coerce types, drop unusable rows and de-duplicate."""
    notes: list[str] = []
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise DataError("Missing required column(s): " + ", ".join(missing))

    out = df.copy()
    for c in ("CuisineType", "Segment", "Subregion", "RestaurantName"):
        out[c] = out[c].astype("string").str.strip()
    numeric_cols = [c for c in out.columns if c not in ("CuisineType", "Segment", "Subregion", "RestaurantName")]
    for c in numeric_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    n0 = len(out)
    bad_dim = out[["CuisineType", "Segment", "Subregion"]].isna().any(axis=1) | (
        out[["CuisineType", "Segment", "Subregion"]].apply(lambda s: s.str.len() == 0).any(axis=1)
    )
    order_cols = list(ORDER_COL.values())
    bad_orders = out[order_cols].isna().any(axis=1) | (out[order_cols] < 0).any(axis=1)
    bad_id = out["RestaurantID"].isna()
    keep = ~(bad_dim | bad_orders | bad_id)
    out = out[keep].copy()
    if n0 - len(out):
        notes.append(f"Removed {n0 - len(out)} row(s) with missing/negative order counts or missing identifiers.")

    out[order_cols] = out[order_cols].round().astype("int64")
    out["RestaurantID"] = out["RestaurantID"].astype("int64")
    for c in ("CuisineType", "Segment", "Subregion", "RestaurantName"):
        out[c] = out[c].astype(str)  # plain text columns chart/group more predictably

    zero = out[order_cols].sum(axis=1) == 0
    if zero.any():
        notes.append(f"Removed {int(zero.sum())} restaurant(s) with zero orders in every channel (shares undefined).")
        out = out[~zero]

    dups = out["RestaurantID"].duplicated(keep="last")
    if dups.any():
        notes.append(f"Resolved {int(dups.sum())} duplicate RestaurantID(s) - kept the most recent entry.")
        out = out[~dups]

    if out.empty:
        raise DataError("No usable rows remain after cleaning.")
    return Prepared(out.reset_index(drop=True), notes)


def merge_new_entries(base: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Append new rows to an existing dataset; the newer row wins on RestaurantID."""
    cols = list(dict.fromkeys([*base.columns, *new.columns]))
    both = pd.concat([base.reindex(columns=cols), new.reindex(columns=cols)], ignore_index=True)
    return both.drop_duplicates(subset="RestaurantID", keep="last").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Derived metrics
# --------------------------------------------------------------------------- #
def _entropy(shares: np.ndarray) -> np.ndarray:
    p = np.clip(shares, 1e-12, 1.0)
    h = -(shares * np.log(p)).sum(axis=1)
    return h / np.log(shares.shape[1])


def derive(df: pd.DataFrame, thr: RiskThresholds | None = None) -> pd.DataFrame:
    """Add per-restaurant share, dependence, diversification and risk columns."""
    thr = thr or RiskThresholds()
    d = df.copy()
    counts = d[[ORDER_COL[c] for c in CHANNELS]].to_numpy(dtype=float)
    total = counts.sum(axis=1)
    shares = counts / total[:, None]

    d["TotalOrders"] = total.astype("int64")
    for i, ch in enumerate(CHANNELS):
        d[SHARE_COL[ch]] = shares[:, i]
    d["DeliveryOrders"] = d[[ORDER_COL[c] for c in DELIVERY]].sum(axis=1)
    d["AggregatorOrders"] = d[[ORDER_COL[c] for c in AGGREGATORS]].sum(axis=1)
    d["DeliveryShare"] = d["DeliveryOrders"] / d["TotalOrders"]
    d["AggDependence"] = d["AggregatorOrders"] / d["TotalOrders"]          # Aggregator Dependence Index
    d["MaxAggShare"] = d[[SHARE_COL[c] for c in AGGREGATORS]].max(axis=1)
    d["TopAggregator"] = np.where(d[SHARE_COL["Uber Eats"]] >= d[SHARE_COL["DoorDash"]], "Uber Eats", "DoorDash")
    d["InStoreReliance"] = d[SHARE_COL["In-Store"]]                         # In-Store Reliance Ratio
    d["Diversification"] = _entropy(shares)                                # Channel Diversification Score
    d["TopChannel"] = np.array(CHANNELS)[shares.argmax(axis=1)]
    d["DiversificationBand"] = d["Diversification"].map(diversification_band)
    d["RiskTier"] = assign_risk(d, thr)
    return d


def diversification_band(score: float) -> str:
    for cutoff, label in DIVERSIFICATION_BANDS:
        if score >= cutoff:
            return label
    return DIVERSIFICATION_BANDS[-1][1]


def assign_risk(d: pd.DataFrame, thr: RiskThresholds) -> pd.Categorical:
    high = (d["AggDependence"] >= thr.high_combined) | (d["MaxAggShare"] >= thr.high_single)
    elev = (~high) & (d["AggDependence"] >= thr.elevated_combined)
    tier = np.where(high, "High risk", np.where(elev, "Elevated", "Balanced"))
    return pd.Categorical(tier, categories=RISK_TIERS, ordered=True)


# --------------------------------------------------------------------------- #
# Aggregations
# --------------------------------------------------------------------------- #
def channel_totals(df: pd.DataFrame) -> pd.DataFrame:
    """Overall volume, market share and rank by channel (volume-weighted)."""
    orders = pd.Series({c: df[ORDER_COL[c]].sum() for c in CHANNELS}, name="Orders")
    t = orders.to_frame()
    t["Share"] = t["Orders"] / t["Orders"].sum()
    t["Rank"] = t["Orders"].rank(ascending=False, method="min").astype(int)
    if all(REV_COL[c] in df.columns for c in CHANNELS):
        t["Revenue"] = pd.Series({c: df[REV_COL[c]].sum() for c in CHANNELS})
    t.index.name = "Channel"
    return t.reset_index().sort_values("Rank").reset_index(drop=True)


def orders_by(df: pd.DataFrame, dim_col: str) -> pd.DataFrame:
    """Order volume: rows = dimension values, columns = channels."""
    g = df.groupby(dim_col, observed=True)[[ORDER_COL[c] for c in CHANNELS]].sum()
    g.columns = CHANNELS
    return g


def share_matrix(df: pd.DataFrame, dim_col: str, channels: Iterable[str] = CHANNELS, basis: str = "subset") -> pd.DataFrame:
    """Channel share (0..1) per dimension value.

    basis='subset' -> normalise over the chosen channels (columns sum to 100%)
    basis='total'  -> divide by ALL orders (used for the In-Store-only view)
    """
    channels = list(channels)
    g = orders_by(df, dim_col)
    denom = g[channels].sum(axis=1) if basis == "subset" else g.sum(axis=1)
    return g[channels].div(denom.replace(0, np.nan), axis=0)


def dominance_table(df: pd.DataFrame, dim_col: str) -> pd.DataFrame:
    """Subregion/cuisine/segment channel dominance: leader, its share and lead margin."""
    g = orders_by(df, dim_col)
    sh = g.div(g.sum(axis=1), axis=0)
    ranked = np.sort(sh.to_numpy(), axis=1)[:, ::-1]
    t = pd.DataFrame(
        {
            "Restaurants": df.groupby(dim_col, observed=True).size(),
            "Orders": g.sum(axis=1),
            "Leading channel": sh.idxmax(axis=1),
            "Leader share": ranked[:, 0],
            "Lead over #2 (pp)": (ranked[:, 0] - ranked[:, 1]) * 100,
            "In-store share": sh["In-Store"],
            "Aggregator share": sh[AGGREGATORS].sum(axis=1),
        }
    )
    t.index.name = dim_col
    return t.sort_values("Aggregator share", ascending=False).reset_index()


def segment_in_group(df: pd.DataFrame, urban_subregions: Iterable[str]) -> pd.Series:
    urban = set(urban_subregions)
    return np.where(df["Subregion"].isin(urban), "Urban core", "Suburban")


def urban_vs_suburban(df: pd.DataFrame, urban_subregions: Iterable[str]) -> pd.DataFrame:
    d = df.assign(Area=segment_in_group(df, urban_subregions))
    g = d.groupby("Area")[[ORDER_COL[c] for c in CHANNELS]].sum()
    g.columns = CHANNELS
    sh = g.div(g.sum(axis=1), axis=0)
    sh["Restaurants"] = d.groupby("Area").size()
    return sh.reset_index()


def aggregator_index_by(df: pd.DataFrame, dim_col: str) -> pd.DataFrame:
    """Volume-weighted Aggregator Dependence Index + diversification per group."""
    g = df.groupby(dim_col, observed=True)
    out = pd.DataFrame(
        {
            "Restaurants": g.size(),
            "Aggregator dependence": g["AggregatorOrders"].sum() / g["TotalOrders"].sum(),
            "In-store reliance": g["InStoreOrders"].sum() / g["TotalOrders"].sum(),
            "Avg diversification": g["Diversification"].mean(),
            "% high risk": g["RiskTier"].apply(lambda s: (s == "High risk").mean()),
        }
    )
    return out.sort_values("Aggregator dependence", ascending=False).reset_index()


def risk_by(df: pd.DataFrame, dim_col: str) -> pd.DataFrame:
    """Share of restaurants in each risk tier per dimension value."""
    ct = pd.crosstab(df[dim_col], df["RiskTier"], normalize="index").reindex(columns=RISK_TIERS, fill_value=0)
    return ct


def kpi_summary(df: pd.DataFrame) -> dict:
    tot = df["TotalOrders"].sum()
    return {
        "restaurants": int(len(df)),
        "total_orders": int(tot),
        "channel_share": {c: float(df[ORDER_COL[c]].sum() / tot) for c in CHANNELS},
        "delivery_share": float(df["DeliveryOrders"].sum() / tot),
        "aggregator_dependence": float(df["AggregatorOrders"].sum() / tot),
        "aggregator_dependence_mean": float(df["AggDependence"].mean()),
        "in_store_reliance": float(df["InStoreOrders"].sum() / tot),
        "diversification_mean": float(df["Diversification"].mean()),
        "pct_high_risk": float((df["RiskTier"] == "High risk").mean()),
        "n_high_risk": int((df["RiskTier"] == "High risk").sum()),
        "pct_elevated": float((df["RiskTier"] == "Elevated").mean()),
        "pct_balanced": float((df["RiskTier"] == "Balanced").mean()),
    }


def profit_summary(df: pd.DataFrame) -> pd.DataFrame | None:
    """Revenue, net profit, margin and loss-making share by channel (needs revenue & profit columns)."""
    if not all(REV_COL[c] in df.columns and PROFIT_COL[c] in df.columns for c in CHANNELS):
        return None
    rows = []
    for c in CHANNELS:
        rev, prof, orders = df[REV_COL[c]].sum(), df[PROFIT_COL[c]].sum(), df[ORDER_COL[c]].sum()
        active = df[ORDER_COL[c]] > 0
        rows.append(
            {
                "Channel": c,
                "Orders": orders,
                "Revenue": rev,
                "Net profit": prof,
                "Margin": prof / rev if rev else np.nan,
                "Profit per order": prof / orders if orders else np.nan,
                "% loss-making restaurants": float((df.loc[active, PROFIT_COL[c]] < 0).mean()) if active.any() else np.nan,
            }
        )
    t = pd.DataFrame(rows)
    t["Share of orders"] = t["Orders"] / t["Orders"].sum()
    t["Share of net profit"] = t["Net profit"] / t["Net profit"].sum()
    return t


def projection(df: pd.DataFrame, months: int = 6) -> pd.DataFrame:
    """Scenario projection: each restaurant's orders compound at its own monthly GrowthFactor.

    NOTE: GrowthFactor is per restaurant (not per channel) so channel mix inside a restaurant
    is held constant; only the weighting between restaurants changes.
    """
    g = df["GrowthFactor"].fillna(1.0).to_numpy() if "GrowthFactor" in df.columns else np.ones(len(df))
    rows = []
    for m in range(months + 1):
        f = g ** m
        row = {"Month": m}
        for c in CHANNELS:
            row[c] = float((df[ORDER_COL[c]].to_numpy() * f).sum())
        rows.append(row)
    p = pd.DataFrame(rows)
    p["Total"] = p[CHANNELS].sum(axis=1)
    for c in CHANNELS:
        p[c + " share"] = p[c] / p["Total"]
    p["Aggregator share"] = (p["Uber Eats"] + p["DoorDash"]) / p["Total"]
    return p


# --------------------------------------------------------------------------- #
# Data-quality report
# --------------------------------------------------------------------------- #
def validation_report(df: pd.DataFrame, notes: list[str] | None = None) -> pd.DataFrame:
    """Transparency checks from the methodology (Step 1)."""
    rows: list[dict] = []

    def add(check, status, detail):
        rows.append({"Check": check, "Status": status, "Detail": detail})

    n = len(df)
    add("Required columns", "Pass", f"All {len(REQUIRED)} required columns present; {n:,} restaurants analysed.")
    for note in notes or []:
        add("Cleaning action", "Warning", note)

    if "MonthlyOrders" in df.columns:
        mism = int((df[[ORDER_COL[c] for c in CHANNELS]].sum(axis=1) != df["MonthlyOrders"]).sum())
        add("Channel orders sum to MonthlyOrders", "Pass" if mism == 0 else "Fail",
            "Every restaurant reconciles." if mism == 0 else f"{mism:,} restaurant(s) do not reconcile.")

    share_cols = ["UE_share", "DD_share", "SD_share"]
    if all(c in df.columns for c in share_cols):
        s = df[share_cols].sum(axis=1)
        off = int(((s - 1).abs() > 0.01).sum())
        add("Delivery shares (UE+DD+SD) sum to ~100%", "Pass" if off == 0 else "Fail",
            f"Range {s.min():.1%} - {s.max():.1%}; {off:,} outside +/-1pp.")
        total_d = df["UberEatsOrders"] + df["DoorDashOrders"] + df["SelfDeliveryOrders"]
        ue = (df["UberEatsOrders"] / total_d.replace(0, np.nan) - df["UE_share"]).abs()
        bad = int((ue > 0.01).sum())
        add("Delivery share columns match order counts", "Pass" if bad == 0 else "Warning",
            f"{bad:,} restaurant(s) differ by more than 1pp (max {ue.max() * 100:.2f}pp).")

    if "InStoreShare" in df.columns:
        recomputed = df["InStoreOrders"] / df[[ORDER_COL[c] for c in CHANNELS]].sum(axis=1)
        diff = df["InStoreShare"] - recomputed
        bad = int((diff.abs() > 0.01).sum())
        add("InStoreShare column matches order counts", "Pass" if bad == 0 else "Warning",
            f"{bad:,} of {n:,} restaurants differ by >1pp (mean gap {diff.mean() * 100:+.1f}pp). "
            "The dashboard uses order counts as the source of truth.")

    if all(REV_COL[c] in df.columns for c in CHANNELS) and "AOV" in df.columns:
        bad = 0
        for c in CHANNELS:
            bad += int(((df[REV_COL[c]] - df[ORDER_COL[c]] * df["AOV"]).abs() > 1.0).sum())
        add("Revenue = orders x AOV", "Pass" if bad == 0 else "Warning",
            "All channel revenues reconcile." if bad == 0 else f"{bad:,} channel-revenue values differ by more than $1.")

    ranges = []
    for col, (lo, hi) in DOCUMENTED_RANGES.items():
        if col in df.columns:
            out = int(((df[col] < lo - 1e-9) | (df[col] > hi + 1e-9)).sum())
            if out:
                ranges.append(f"{col}: {out}")
    add("Values within documented ranges", "Pass" if not ranges else "Warning",
        "All values inside the brief's ranges." if not ranges else "Outside range - " + "; ".join(ranges))

    for col in ("MonthlyOrders", "AOV"):
        if col in df.columns:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            out = int(((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum())
            add(f"Outliers in {col} (1.5 x IQR)", "Pass" if out == 0 else "Warning", f"{out:,} flagged for transparency (kept in analysis).")

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Auto-generated insights (re-computed for whatever data / filters are active)
# --------------------------------------------------------------------------- #
def insights(df: pd.DataFrame, thr: RiskThresholds | None = None) -> list[str]:
    thr = thr or RiskThresholds()
    if df.empty:
        return []
    k = kpi_summary(df)
    ct = channel_totals(df)
    lead = ct.iloc[0]
    out = [
        f"{lead['Channel']} is the largest channel with {lead['Share']:.1%} of {k['total_orders']:,} monthly orders; "
        f"delivery as a whole carries {k['delivery_share']:.1%} and in-store only {k['in_store_reliance']:.1%}.",
        f"Uber Eats and DoorDash together handle {k['aggregator_dependence']:.1%} of orders - "
        f"{k['n_high_risk']:,} restaurants ({k['pct_high_risk']:.1%}) breach the dependency thresholds "
        f"(aggregators >= {thr.high_combined:.0%} combined or any single platform >= {thr.high_single:.0%}).",
    ]
    for label, col in (("cuisine", "CuisineType"), ("segment", "Segment")):
        if df[col].nunique() > 1:
            a = aggregator_index_by(df, col)
            hi, lo = a.iloc[0], a.iloc[-1]
            out.append(
                f"Most aggregator-dependent {label}: {hi[col]} ({hi['Aggregator dependence']:.1%}); "
                f"least: {lo[col]} ({lo['Aggregator dependence']:.1%}) - a {abs(hi['Aggregator dependence'] - lo['Aggregator dependence']) * 100:.1f}pp spread."
            )
    if df["Subregion"].nunique() > 1:
        a = aggregator_index_by(df, "Subregion")
        spread = (a["Aggregator dependence"].max() - a["Aggregator dependence"].min()) * 100
        out.append(
            f"Subregions differ by only {spread:.1f}pp in aggregator dependence "
            f"({a.iloc[-1]['Subregion']} lowest, {a.iloc[0]['Subregion']} highest) - "
            + ("geography is a weak driver of channel mix compared with cuisine and segment."
               if spread < 3 else "geography is a meaningful driver of channel mix.")
        )
    ps = profit_summary(df)
    if ps is not None:
        agg = ps[ps["Channel"].isin(AGGREGATORS)]
        ins = ps[ps["Channel"] == "In-Store"].iloc[0]
        out.append(
            f"Aggregators deliver {agg['Share of orders'].sum():.1%} of orders but only {agg['Share of net profit'].sum():.1%} of net profit; "
            f"in-store earns ${ins['Profit per order']:.2f} per order versus ${agg['Net profit'].sum() / agg['Orders'].sum():.2f} on aggregator orders."
        )
    return out
