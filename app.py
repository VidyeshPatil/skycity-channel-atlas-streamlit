"""SkyCity Auckland - Order Channel Performance & Market Share Analytics (Streamlit).

Run:   streamlit run app.py

Dynamic by design
-----------------
* Drop extra CSV files into ./data  -> they are merged automatically (newest row wins per RestaurantID).
* Or upload CSVs in the Settings panel (replace the data, or append new entries to it).
* Every filter, chart, KPI, insight and risk flag is recomputed from the order counts, so new
  cuisines / subregions / segments / restaurants appear without touching the code.
"""
from __future__ import annotations

import base64
import html
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import analytics as A

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
DATA_DIR = ROOT / "data"

st.set_page_config(
    page_title="SkyCity Auckland | Channel Analytics",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------------------------------- #
# Palette (fresh teal / sky / coral / amber)
# --------------------------------------------------------------------------- #
INK = "#E9F5F3"
INK2 = "#A9C4C4"
GRID = "#1B4352"
TEAL = "#0FA596"
CHANNEL_COLORS = {"In-Store": "#0FA596", "Uber Eats": "#D4781A", "DoorDash": "#E0457B", "Self-Delivery": "#4A86F0"}
RISK_COLORS = {"High risk": "#FF6B86", "Elevated": "#F2B640", "Balanced": "#34D399"}
HEAT = [[0, "#103240"], [0.4, "#0B6F66"], [0.75, "#14AA98"], [1, "#7CF0DF"]]
HEAT_RISK = [[0, "#103240"], [0.5, "#B7842A"], [1, "#FF6B86"]]
FONT = "Figtree, Inter, 'Segoe UI', system-ui, sans-serif"


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def b64(name: str) -> str:
    p = ASSETS / name
    return base64.b64encode(p.read_bytes()).decode() if p.exists() else ""


_PLOTLY_NEW_WIDTH = "width" in inspect.signature(st.plotly_chart).parameters


def show(fig: go.Figure, key: str) -> None:
    kw = {"width": "stretch"} if _PLOTLY_NEW_WIDTH else {"use_container_width": True}
    st.plotly_chart(fig, key=key, **kw)


def table(df: pd.DataFrame, **kwargs) -> None:
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kwargs)
    except Exception:  # older Streamlit versions
        st.dataframe(df, use_container_width=True, hide_index=True, **kwargs)


def rerun() -> None:
    (getattr(st, "rerun", None) or st.experimental_rerun)()


def style(fig: go.Figure, h: int = 380, title: str | None = None) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        height=h,
        margin=dict(l=10, r=10, t=52 if title else 16, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=INK, size=13),
        legend=dict(orientation="h", y=-0.18, x=0, title_text=""),
        hoverlabel=dict(font_family=FONT),
    )
    if title:
        fig.update_layout(title=dict(text=title, x=0.01, font=dict(size=16, family=FONT)))
    return fig


def heatmap(m: pd.DataFrame, title: str, scale=HEAT, height: int | None = None) -> go.Figure:
    fig = go.Figure(
        go.Heatmap(
            z=m.to_numpy(dtype=float),
            x=[str(c) for c in m.columns],
            y=[str(i) for i in m.index],
            colorscale=scale,
            texttemplate="%{z:.1%}",
            hovertemplate="%{y} · %{x}<br><b>%{z:.1%}</b><extra></extra>",
            colorbar=dict(tickformat=".0%", thickness=12, len=0.9),
            xgap=3,
            ygap=3,
        )
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(side="top")
    return style(fig, h=height or max(250, 48 * len(m) + 110), title=title)


def pct(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = out[c] * 100
    return out


def esc(text: str) -> str:
    """HTML-escape and neutralise '$' so Streamlit's markdown never treats $...$ as LaTeX."""
    return html.escape(str(text)).replace("$", "&#36;")


def kpi_cards(items: list[tuple[str, str, str, str]]) -> None:
    cards = "".join(
        f'<div class="kpi" style="border-top-color:{c}"><div class="kpi-label">{esc(l)}</div>'
        f'<div class="kpi-value">{esc(v)}</div><div class="kpi-sub">{esc(s)}</div></div>'
        for l, v, s, c in items
    )
    st.markdown(f'<div class="kpi-grid">{cards}</div>', unsafe_allow_html=True)


def section(title: str, sub: str = "") -> None:
    st.markdown(
        f'<div class="sec"><div class="sec-title">{esc(title)}</div>'
        + (f'<div class="sec-sub">{esc(sub)}</div>' if sub else "")
        + "</div>",
        unsafe_allow_html=True,
    )


def callout(text: str, kind: str = "info") -> None:
    st.markdown(f'<div class="callout {kind}">{esc(text)}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# CSS (background imagery, glass panels, fresh palette)
# --------------------------------------------------------------------------- #
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@600;700;800&family=Figtree:wght@400;500;600;700&display=swap');
:root{--bg:#06141b;--surface:#0c2530;--surface2:#103240;--line:#1b4352;--ink:#e9f5f3;--ink2:#a9c4c4;--ink3:#78989b;--accent:#2fd4c0;--crit:#ff6b86;--warn:#f2b640;--good:#34d399;}
html, body, [class*="css"], .stMarkdown, button, input, textarea { font-family: 'Figtree','Segoe UI',system-ui,sans-serif; }
h1, h2, h3, .hero h1, .sec-title, .kpi-value, .brand b { font-family: 'Bricolage Grotesque','Figtree',sans-serif; }

.stApp {
  background: linear-gradient(180deg, rgba(6,20,27,.80) 0%, rgba(6,20,27,.94) 60%, #06141b 100%),
              url("data:image/jpeg;base64,__BG__") center top / cover fixed no-repeat !important;
  color: var(--ink);
}
.stApp > header, [data-testid="stHeader"] { background: transparent !important; }
[data-testid="stAppViewContainer"], [data-testid="stMain"], .stMain, section.main { background: transparent !important; }
.block-container { max-width: 1320px; padding: 1.2rem 2rem 2.4rem !important; }
.stApp p, .stApp li, .stApp label, .stApp span, .stMarkdown { color: var(--ink); }
.stCaption, [data-testid="stCaptionContainer"] { color: var(--ink2) !important; }

/* sidebar */
[data-testid="stSidebar"] > div:first-child {
  background: linear-gradient(rgba(6,20,27,.92), rgba(6,20,27,.97)), url("data:image/jpeg;base64,__BG__") center / cover no-repeat;
  border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] h4 { font-size: .74rem; letter-spacing: .14em; text-transform: uppercase; color: var(--ink3) !important; margin-top: 1rem; }
.brand { border-radius: 16px; padding: 16px; margin-bottom: 10px; color: #fff; border: 1px solid var(--line);
  background: linear-gradient(120deg, rgba(6,20,27,.85), rgba(15,165,150,.35)), url("data:image/jpeg;base64,__HERO__") center / cover; }
.brand b { font-size: 1.1rem; display:block; line-height:1.2; color:#fff; }
.brand span { font-size: .78rem; color: var(--ink2); }

/* hero */
.hero { border-radius: 22px; padding: 40px 38px 26px; margin-bottom: 14px; color: #fff; min-height: 250px;
  background-size: cover; background-position: center bottom; border: 1px solid var(--line); box-shadow: 0 18px 50px rgba(0,0,0,.45); }
.hero-badge { font-size:.72rem; letter-spacing:.16em; font-weight:700; color: var(--accent); text-transform: uppercase; }
.hero h1 { color:#fff; font-size: clamp(1.9rem, 4vw, 3rem); line-height:1.04; margin: 10px 0 10px; padding:0; font-weight:800; max-width: 760px; }
.hero p { max-width: 640px; color: #cfe3e3; font-size: 1rem; margin: 0 0 16px; }
.chips span { display:inline-block; margin: 0 8px 8px 0; padding: 5px 13px; border-radius: 999px; font-size:.8rem; font-weight:600;
  background: rgba(12,37,48,.7); border: 1px solid var(--line); color: var(--ink2); backdrop-filter: blur(4px); }

/* TABS - big, obvious, one-per-story */
.stTabs [data-baseweb="tab-list"] { gap: 6px; flex-wrap: wrap; padding: 6px; border-radius: 16px; background: rgba(10,32,43,.9);
  border: 1px solid var(--line); backdrop-filter: blur(14px); }
.stTabs [data-baseweb="tab"] { height: 46px; padding: 0 18px; border-radius: 11px; font-weight: 700; color: var(--ink2); background: transparent; }
.stTabs [data-baseweb="tab"]:hover { background: var(--surface2); color: var(--ink); }
.stTabs [aria-selected="true"] { background: var(--accent) !important; color: #04211d !important; }
.stTabs [aria-selected="true"] p { color: #04211d !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none !important; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.1rem; }

/* KPI cards */
.kpi-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 14px; margin: 4px 0 18px; }
.kpi { background: rgba(12,37,48,.86); border-radius:16px; padding:14px 16px 12px; border:1px solid var(--line); border-top: 4px solid var(--accent-c, var(--accent));
  box-shadow: 0 8px 24px rgba(0,0,0,.30); }
.kpi-label { font-size:.72rem; text-transform:uppercase; letter-spacing:.1em; color: var(--ink3); font-weight:700; }
.kpi-value { font-size:1.75rem; font-weight:800; color: var(--ink); margin: 2px 0; font-variant-numeric: tabular-nums; }
.kpi-sub { font-size:.78rem; color: var(--ink2); }

.sec { margin: 22px 0 8px; }
.sec-title { font-size:1.2rem; font-weight:700; color: var(--ink); }
.sec-sub { font-size:.86rem; color: var(--ink2); margin-top:2px; }

.callout { border-radius:14px; padding:12px 16px; margin: 8px 0 14px; font-size:.92rem; border:1px solid; }
.callout.info { background:rgba(74,134,240,.12); border-color:rgba(74,134,240,.45); color:#cfe0ff; }
.callout.warn { background:rgba(242,182,64,.12); border-color:rgba(242,182,64,.5); color:#ffe2a8; }
.callout.good { background:rgba(52,211,153,.12); border-color:rgba(52,211,153,.5); color:#b8f5dc; }

.insights { list-style:none; padding:0; margin:0; }
.insights li { background: rgba(12,37,48,.86); border:1px solid var(--line); border-left:5px solid var(--accent); border-radius:12px;
  padding:10px 14px; margin-bottom:9px; font-size:.93rem; color: var(--ink); }

/* cards around charts */
[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 18px; background: rgba(12,37,48,.78); border: 1px solid var(--line) !important; }
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
.stDownloadButton button, .stButton button { border-radius: 12px; border: 1px solid var(--line); background: var(--surface2); color: var(--ink); font-weight: 600; }
.stDownloadButton button:hover, .stButton button:hover { border-color: var(--accent); color: var(--accent); }

/* the left sidebar is not used: every control lives in the filter bar above the tabs */
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] { display: none !important; }
.fbar-title { font-size: .72rem; letter-spacing: .14em; text-transform: uppercase; color: var(--ink3); font-weight: 700; margin: 0 0 4px; }
[data-baseweb="select"] > div { max-height: 76px; overflow-y: auto; background: var(--surface); border-color: var(--line); }
/* selected values: tab-style chips with dark, high-contrast text on the teal */
[data-baseweb="tag"] { margin: 2px 5px 2px 0; padding: 0 6px 0 11px; font-size: .8rem; height: 28px; border-radius: 9px;
  background: var(--accent) !important; border: 1px solid rgba(255,255,255,.25); box-shadow: 0 2px 6px rgba(0,0,0,.3); }
[data-baseweb="tag"], [data-baseweb="tag"] span, [data-baseweb="tag"] p, [data-baseweb="tag"] div { color: #04211d !important; font-weight: 700 !important; }
[data-baseweb="tag"] svg, [data-baseweb="tag"] svg path { fill: #04211d !important; color: #04211d !important; }
[data-baseweb="tag"]:hover { background: #5ee6d5 !important; }
.stApp label p { font-size: .82rem; font-weight: 700; color: var(--ink2); }
[data-testid="stExpander"] { background: rgba(12,37,48,.78); border: 1px solid var(--line); border-radius: 14px; }
[data-testid="stExpander"] summary { font-weight: 700; }
[data-testid="stRadio"] [role="radiogroup"] { gap: .3rem .9rem; flex-wrap: wrap; }
footer, #MainMenu { visibility: hidden; }
.foot { text-align:center; color: var(--ink3); font-size:.8rem; margin-top: 26px; }
</style>
"""
st.markdown(
    CSS.replace("__BG__", b64("bg_main.jpg")).replace("__HERO__", b64("hero.jpg")),
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Data loading (cached; auto-refreshes when files in ./data change)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Reading dataset…")
def read_path(path_str: str, mtime: float) -> pd.DataFrame:  # mtime is part of the cache key
    return A.read_csv(path_str)


@st.cache_data(show_spinner="Reading upload…")
def read_bytes(raw: bytes) -> pd.DataFrame:
    return A.read_csv(raw)


@st.cache_data(show_spinner="Cleaning & validating…")
def prep(raw: pd.DataFrame):
    p = A.prepare(raw)
    return p.df, p.notes


@st.cache_data(show_spinner=False)
def derive_cached(df: pd.DataFrame, high_combined: float, high_single: float, elevated: float) -> pd.DataFrame:
    return A.derive(df, A.RiskThresholds(high_combined, high_single, elevated))


def folder_files() -> list[Path]:
    return sorted(DATA_DIR.glob("*.csv"), key=lambda p: p.stat().st_mtime) if DATA_DIR.exists() else []


def load_folder() -> tuple[pd.DataFrame | None, list[str]]:
    frames, names = [], []
    for f in folder_files():
        try:
            frames.append(read_path(str(f), f.stat().st_mtime))
            names.append(f.name)
        except Exception as exc:  # a bad extra file should not take the app down
            st.warning(f"Skipped {f.name}: {exc}")
    if not frames:
        return None, []
    out = frames[0]
    for nxt in frames[1:]:
        out = A.merge_new_entries(out, nxt)
    return out, names


# Page layout: hero, then the filter bar, then the tabs. Containers are created in display order and
# filled later, because the filters need the loaded data and the hero needs the filtered counts.
box_hero = st.container()
box_bar = st.container()
with box_bar:
    box_row = st.container(border=True)
    box_more = st.expander("⚙️  Settings · risk rules · urban core · data source", expanded=False)
with box_more:
    st.markdown('<div class="fbar-title">Data source</div>', unsafe_allow_html=True)
    mode = st.radio(
        "Data source",
        ["Built-in dataset (./data)", "Upload CSV · replace", "Upload CSV · append new entries"],
        horizontal=True,
        label_visibility="collapsed",
    )
    uploads = []
    if not mode.startswith("Built-in"):
        uploads = st.file_uploader("CSV file(s) – same columns as the original", type=["csv"], accept_multiple_files=True)
    if st.button("🔄 Reload data", help="Clear caches and re-read ./data"):
        st.cache_data.clear()
        rerun()
    st.markdown('<div class="fbar-title" style="margin-top:14px">Dependency-risk rules</div>', unsafe_allow_html=True)
    rule_cols = st.columns(3)
    urban_slot = st.container()

base_raw, base_names = load_folder()
raw, source_label = base_raw, ", ".join(base_names) if base_names else "no file"

if uploads:
    try:
        up = None
        for f in uploads:
            frame = read_bytes(f.getvalue())
            prep(frame)  # validate each file now so a bad upload is rejected instead of halting the app
            up = frame if up is None else A.merge_new_entries(up, frame)
        if mode.endswith("replace") or base_raw is None:
            raw, source_label = up, "uploaded: " + ", ".join(f.name for f in uploads)
        else:
            raw = A.merge_new_entries(base_raw, up)
            source_label = f"{', '.join(base_names)} + uploaded: " + ", ".join(f.name for f in uploads)
    except A.DataError as exc:
        st.error(f"Upload rejected – {exc}. Showing the built-in dataset.")
    except Exception as exc:
        st.error(f"Could not read the upload ({exc}). Showing the built-in dataset.")

if raw is None:
    st.error("No dataset found. Put a CSV in the `data/` folder or upload one in the Settings panel.")
    st.stop()

try:
    clean_df, clean_notes = prep(raw)
except A.DataError as exc:
    st.error(f"Dataset problem: {exc}")
    st.stop()

# --------------------------------------------------------------------------- #
# Sidebar: rules + filters
# --------------------------------------------------------------------------- #
all_sub = sorted(clean_df["Subregion"].unique())
all_cui = sorted(clean_df["CuisineType"].unique())
all_seg = sorted(clean_df["Segment"].unique())

with rule_cols[0]:
    hc = st.slider("High risk · Uber Eats + DoorDash ≥", 40, 95, 70, 1, format="%d%%",
                   help="Share of a restaurant's total orders coming from both aggregators combined. The project brief uses 70%.") / 100
with rule_cols[1]:
    hs = st.slider("High risk · any single platform ≥", 30, 80, 50, 1, format="%d%%",
                   help="Share of total orders from ONE platform. In this dataset no restaurant exceeds ~58%.") / 100
with rule_cols[2]:
    ec = min(st.slider("Elevated · combined aggregators ≥", 30, 90, 60, 1, format="%d%%"), int(hc * 100)) / 100


def reset_filters() -> None:
    st.session_state["fver"] = st.session_state.get("fver", 0) + 1  # new widget keys -> defaults (everything selected)


fv = st.session_state.get("fver", 0)
urban_default = [x for x in all_sub if x.strip().lower() in ("cbd", "central auckland", "auckland central")]
with urban_slot:
    urban_sel = st.multiselect("Treat as 'urban core' (everything else counts as suburban)", all_sub, default=urban_default,
                               key=f"f_urb_{fv}")

with box_row:
    st.markdown('<div class="fbar-title">Filters</div>', unsafe_allow_html=True)
    fc = st.columns([1.1, 1, 1, 1])
    with fc[0]:
        view = st.radio(
            "Channel view",
            ["All channels", "In-Store only", "Delivery only"],
            help="Applies to the Geography and Cuisine tabs. Delivery only re-bases shares on the three delivery channels; "
                 "In-Store only shows in-store orders as a % of all orders.",
        )
    with fc[1]:
        sel_sub = st.multiselect("Subregion", all_sub, default=all_sub, key=f"f_sub_{fv}")
    with fc[2]:
        sel_cui = st.multiselect("Cuisine", all_cui, default=all_cui, key=f"f_cui_{fv}")
    with fc[3]:
        sel_seg = st.multiselect("Segment", all_seg, default=all_seg, key=f"f_seg_{fv}")
    foot = st.columns([3, 1])
    with foot[1]:
        st.button("↺ Reset all filters", on_click=reset_filters)

df_all = derive_cached(clean_df, hc, hs, ec)
mask = df_all["Subregion"].isin(sel_sub) & df_all["CuisineType"].isin(sel_cui) & df_all["Segment"].isin(sel_seg)
df = df_all[mask].copy()
if df.empty:
    st.warning("No restaurants match the current filters – widen the filters above.")
    st.stop()

with foot[0]:
    st.caption(f"Showing **{len(df):,}** of {len(df_all):,} restaurants · channel view applies to the Geography and Cuisine tabs")

thr = A.RiskThresholds(hc, hs, ec)
view_channels = {"All channels": A.CHANNELS, "In-Store only": ["In-Store"], "Delivery only": A.DELIVERY}[view]
basis = "total" if view == "In-Store only" else "subset"
basis_text = {
    "All channels": "share of all orders",
    "In-Store only": "in-store orders as % of all orders",
    "Delivery only": "share of delivery orders (Uber Eats + DoorDash + Self-Delivery)",
}[view]
k = A.kpi_summary(df)

# --------------------------------------------------------------------------- #
# Hero
# --------------------------------------------------------------------------- #
chips = "".join(
    f"<span>{t}</span>"
    for t in (
        f"{len(df):,} of {len(df_all):,} restaurants",
        f"{df['Subregion'].nunique()} subregions",
        f"{df['CuisineType'].nunique()} cuisines",
        f"{df['Segment'].nunique()} segments",
        f"Source: {html.escape(source_label)}",
    )
)
with box_hero:
  st.markdown(
    f'<div class="hero" style="background-image:linear-gradient(90deg,rgba(6,20,27,.94) 0%,rgba(6,20,27,.58) 55%,rgba(6,20,27,.12) 100%),'
    f"url('data:image/jpeg;base64,{b64('hero.jpg')}')\">"
    '<div class="hero-badge">ORDER CHANNEL PERFORMANCE &amp; MARKET SHARE</div>'
    "<h1>Auckland Restaurants &amp; Bars — where do the orders come from?</h1>"
    "<p>Channel dependence, geographic ordering behaviour and aggregator risk across four parallel order channels: "
    "In-Store, Uber Eats, DoorDash and Self-Delivery.</p>"
    f'<div class="chips">{chips}</div></div>',
    unsafe_allow_html=True,
)

tabs = st.tabs(
    ["📊 Channel overview", "🗺️ Geography", "🍜 Cuisine & segment", "🚨 Dependency risk",
     "💰 Profitability", "📈 Outlook", "🧪 Data & methodology"]
)

# --------------------------------------------------------------------------- #
# 1. Overview
# --------------------------------------------------------------------------- #
with tabs[0]:
    kpi_cards(
        [
            ("Restaurants", f"{k['restaurants']:,}", "in current selection", TEAL),
            ("Monthly orders", f"{k['total_orders']:,}", "all four channels", "#4A86F0"),
            ("Delivery share", f"{k['delivery_share']:.1%}", "orders not taken in-store", "#8A7BF0"),
            ("Aggregator dependence", f"{k['aggregator_dependence']:.1%}", "Uber Eats + DoorDash (ADI)", "#D4781A"),
            ("In-store reliance", f"{k['in_store_reliance']:.1%}", "walk-in strength", "#34D399"),
            ("High-risk venues", f"{k['pct_high_risk']:.1%}", f"{k['n_high_risk']:,} restaurants flagged", "#E0457B"),
        ]
    )
    ct = A.channel_totals(df)
    c1, c2 = st.columns([1, 1.15])
    with c1:
        with st.container(border=True):
            fig = go.Figure(
                go.Pie(
                    labels=ct["Channel"], values=ct["Orders"], hole=0.64, sort=False,
                    marker=dict(colors=[CHANNEL_COLORS[c] for c in ct["Channel"]], line=dict(color="#0C2530", width=3)),
                    textinfo="percent", textfont=dict(size=14, color="#fff"),
                    hovertemplate="%{label}<br>%{value:,} orders<br>%{percent}<extra></extra>",
                )
            )
            fig.add_annotation(text=f"<b>{k['total_orders']:,}</b><br>orders / month", showarrow=False,
                               xref="paper", yref="paper", x=0.5, y=0.5, font=dict(size=15, color=INK))
            show(style(fig, 380, "Channel market share (volume-weighted)"), "ov_donut")
    with c2:
        with st.container(border=True):
            order = ct.sort_values("Orders")
            fig = go.Figure(
                go.Bar(
                    x=order["Orders"], y=order["Channel"], orientation="h",
                    marker_color=[CHANNEL_COLORS[c] for c in order["Channel"]],
                    text=[f"{v:,.0f}  ({s:.1%})" for v, s in zip(order["Orders"], order["Share"])],
                    textposition="outside", cliponaxis=False,
                    hovertemplate="%{y}<br>%{x:,} orders<extra></extra>",
                )
            )
            fig.update_xaxes(showgrid=True, gridcolor=GRID, range=[0, order["Orders"].max() * 1.35])
            show(style(fig, 380, "Order volume by channel · ranked"), "ov_bars")

    c3, c4 = st.columns([1.15, 1])
    with c3:
        with st.container(border=True):
            dvs = pd.DataFrame(
                {
                    "Group": ["In-Store", "Delivery (all)", "Aggregators (UE + DD)", "Self-Delivery"],
                    "Share": [k["in_store_reliance"], k["delivery_share"], k["aggregator_dependence"],
                              k["channel_share"]["Self-Delivery"]],
                }
            )
            fig = px.bar(dvs, x="Group", y="Share", color="Group", text_auto=".1%",
                         color_discrete_sequence=["#0FA596", "#8A7BF0", "#D4781A", "#4A86F0"])
            fig.update_yaxes(tickformat=".0%", range=[0, 1], gridcolor=GRID)
            fig.update_layout(showlegend=False)
            fig.update_xaxes(title_text="")
            fig.update_yaxes(title_text="")
            show(style(fig, 330, "Delivery vs in-store dominance"), "ov_dvs")
    with c4:
        section("Channel ranking")
        rk = ct.copy()
        rk["Share"] = rk["Share"] * 100
        cfg = {
            "Orders": st.column_config.NumberColumn(format="%d"),
            "Share": st.column_config.ProgressColumn("Market share", format="%.1f%%", min_value=0, max_value=100),
        }
        if "Revenue" in rk.columns:
            cfg["Revenue"] = st.column_config.NumberColumn("Revenue ($)", format="$%.0f")
        table(rk, column_config=cfg)

    section("Auto-generated insights", "Recomputed for whatever data and filters are active")
    st.markdown(
        '<ul class="insights">' + "".join(f"<li>{esc(t)}</li>" for t in A.insights(df, thr)) + "</ul>",
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------- #
# 2. Geography
# --------------------------------------------------------------------------- #
with tabs[1]:
    section("Subregion channel heatmap", f"Cells show {basis_text}")
    m = A.share_matrix(df, "Subregion", view_channels, basis)
    if view == "In-Store only":
        m = m.rename(columns={"In-Store": "In-Store (% of all orders)"})
    with st.container(border=True):
        show(heatmap(m, "Channel share by subregion"), "geo_heat")

    g1, g2 = st.columns(2)
    with g1:
        with st.container(border=True):
            vol = A.orders_by(df, "Subregion")[view_channels].reset_index().melt(id_vars="Subregion", var_name="Channel", value_name="Orders")
            fig = px.bar(vol, x="Subregion", y="Orders", color="Channel", barmode="group",
                         color_discrete_map=CHANNEL_COLORS)
            fig.update_yaxes(gridcolor=GRID, title_text="Orders / month")
            fig.update_xaxes(title_text="")
            show(style(fig, 380, "Order volume by subregion & channel"), "geo_vol")
    with g2:
        with st.container(border=True):
            ua = A.urban_vs_suburban(df, urban_sel)
            ua_long = ua.melt(id_vars=["Area", "Restaurants"], value_vars=view_channels, var_name="Channel", value_name="Share")
            if view == "Delivery only":
                ua_long["Share"] = ua_long["Share"] / ua_long.groupby("Area")["Share"].transform("sum")
            fig = px.bar(ua_long, x="Share", y="Area", color="Channel", orientation="h", barmode="stack" if view != "In-Store only" else "group",
                         color_discrete_map=CHANNEL_COLORS, text_auto=".1%")
            fig.update_xaxes(tickformat=".0%", gridcolor=GRID, title_text="")
            fig.update_yaxes(title_text="")
            show(style(fig, 380, "Urban core vs suburban ordering behaviour"), "geo_urban")
            if not urban_sel:
                st.caption("Pick the urban-core subregion(s) in Settings to populate this comparison.")

    section("Subregion channel dominance", "Which channel leads where, and by how much")
    dom = pct(A.dominance_table(df, "Subregion"), ["Leader share", "In-store share", "Aggregator share"])
    table(
        dom,
        column_config={
            "Orders": st.column_config.NumberColumn(format="%d"),
            "Leader share": st.column_config.ProgressColumn("Leader share", format="%.1f%%", min_value=0, max_value=100),
            "Lead over #2 (pp)": st.column_config.NumberColumn(format="%.1f"),
            "In-store share": st.column_config.NumberColumn(format="%.1f%%"),
            "Aggregator share": st.column_config.ProgressColumn("Aggregator share", format="%.1f%%", min_value=0, max_value=100),
        },
    )
    if df["Subregion"].nunique() > 1:
        ai = A.aggregator_index_by(df, "Subregion")
        spread = (ai["Aggregator dependence"].max() - ai["Aggregator dependence"].min()) * 100
        if spread < 3:
            callout(f"Subregions differ by only {spread:.1f} percentage points in aggregator dependence – "
                    "location is a weak driver of channel mix here; cuisine and business segment explain far more.", "info")
        else:
            callout(f"Aggregator dependence varies by {spread:.1f} percentage points across subregions – geography matters.", "warn")

# --------------------------------------------------------------------------- #
# 3. Cuisine & segment
# --------------------------------------------------------------------------- #
def stacked_share(dim_col: str, label: str, key: str) -> None:
    mm = A.share_matrix(df, dim_col, view_channels, basis)
    order = A.aggregator_index_by(df, dim_col)[dim_col].tolist()
    long = mm.reset_index().melt(id_vars=dim_col, var_name="Channel", value_name="Share")
    fig = px.bar(long, x="Share", y=dim_col, color="Channel", orientation="h", text_auto=".0%",
                 category_orders={dim_col: order}, color_discrete_map=CHANNEL_COLORS)
    fig.update_xaxes(tickformat=".0%", gridcolor=GRID, title_text="")
    fig.update_yaxes(title_text="", autorange="reversed")
    show(style(fig, max(300, 46 * len(mm) + 110), f"{label} · channel mix (sorted by aggregator reliance)"), key)


with tabs[2]:
    section("Cuisine vs channel", f"Cells show {basis_text}")
    mc = A.share_matrix(df, "CuisineType", view_channels, basis)
    cc1, cc2 = st.columns([1, 1.1])
    with cc1:
        with st.container(border=True):
            show(heatmap(mc, "Channel share by cuisine"), "cui_heat")
    with cc2:
        with st.container(border=True):
            stacked_share("CuisineType", "Cuisine", "cui_stack")

    section("Business segment vs channel")
    sc1, sc2 = st.columns([1, 1.1])
    with sc1:
        with st.container(border=True):
            show(heatmap(A.share_matrix(df, "Segment", view_channels, basis), "Channel share by segment"), "seg_heat")
    with sc2:
        with st.container(border=True):
            stacked_share("Segment", "Segment", "seg_stack")

    section("Aggregator-heavy categories", "Aggregator Dependence Index = (Uber Eats + DoorDash orders) ÷ all orders")
    a1, a2 = st.columns(2)
    for col_widget, dim, label, key in ((a1, "CuisineType", "cuisine", "adi_cui"), (a2, "Segment", "segment", "adi_seg")):
        with col_widget:
            with st.container(border=True):
                ai = A.aggregator_index_by(df, dim).sort_values("Aggregator dependence")
                colors = [RISK_COLORS["High risk"] if v >= hc else RISK_COLORS["Elevated"] if v >= ec else RISK_COLORS["Balanced"]
                          for v in ai["Aggregator dependence"]]
                fig = go.Figure(go.Bar(x=ai["Aggregator dependence"], y=ai[dim], orientation="h", marker_color=colors,
                                       text=[f"{v:.1%}" for v in ai["Aggregator dependence"]], textposition="outside", cliponaxis=False))
                fig.add_vline(x=hc, line_dash="dash", line_color="#FF6B86", annotation_text=f"{hc:.0%} high-risk line", annotation_position="top")
                fig.update_xaxes(tickformat=".0%", range=[0, 1], gridcolor=GRID)
                show(style(fig, max(280, 44 * len(ai) + 100), f"Aggregator dependence by {label}"), key)

    section("Cuisine × segment · aggregator dependence", "Where the dependence concentrates")
    gg = df.groupby(["CuisineType", "Segment"])[["AggregatorOrders", "TotalOrders"]].sum()
    piv = (gg["AggregatorOrders"] / gg["TotalOrders"]).unstack("Segment")
    with st.container(border=True):
        show(heatmap(piv, "Aggregator dependence · cuisine × segment", scale=HEAT_RISK), "cui_seg_heat")

# --------------------------------------------------------------------------- #
# 4. Dependency risk
# --------------------------------------------------------------------------- #
with tabs[3]:
    kpi_cards(
        [
            ("High risk", f"{k['n_high_risk']:,}", f"{k['pct_high_risk']:.1%} of restaurants", RISK_COLORS["High risk"]),
            ("Elevated", f"{int((df['RiskTier'] == 'Elevated').sum()):,}", f"{k['pct_elevated']:.1%} of restaurants", RISK_COLORS["Elevated"]),
            ("Balanced", f"{int((df['RiskTier'] == 'Balanced').sum()):,}", f"{k['pct_balanced']:.1%} of restaurants", RISK_COLORS["Balanced"]),
            ("Avg diversification", f"{k['diversification_mean']:.3f}", "0 = one channel · 1 = perfectly even", TEAL),
            ("Mean ADI", f"{k['aggregator_dependence_mean']:.1%}", "average restaurant", "#D4781A"),
        ]
    )
    callout(
        f"Rule in force → High risk: Uber Eats + DoorDash ≥ {hc:.0%} of orders, or any one platform ≥ {hs:.0%}. "
        f"Elevated: combined aggregators ≥ {ec:.0%}. Balanced: everything else. Adjust the rules in Settings above.", "info")
    if (df["MaxAggShare"] >= 0.70).sum() == 0:
        callout("No restaurant relies on a single aggregator for 70% of its orders (the highest single-platform share is "
                f"{df['MaxAggShare'].max():.1%}). The 70% rule therefore bites on the two platforms combined.", "good")

    r1, r2 = st.columns([1.25, 1])
    with r1:
        with st.container(border=True):
            sm = df if len(df) <= 3000 else df.sample(3000, random_state=7)
            fig = px.scatter(
                sm, x="InStoreReliance", y="AggDependence", color="RiskTier", opacity=0.65,
                color_discrete_map=RISK_COLORS, category_orders={"RiskTier": A.RISK_TIERS},
                hover_name="RestaurantName",
                hover_data={"CuisineType": True, "Segment": True, "Subregion": True, "TotalOrders": ":,",
                            "InStoreReliance": ":.1%", "AggDependence": ":.1%", "RiskTier": False},
            )
            fig.update_traces(marker=dict(size=7, line=dict(width=0)))
            fig.add_hline(y=hc, line_dash="dash", line_color="#FF6B86", annotation_text=f"{hc:.0%}", annotation_position="top left")
            fig.update_xaxes(tickformat=".0%", title_text="In-store reliance", gridcolor=GRID)
            fig.update_yaxes(tickformat=".0%", title_text="Aggregator dependence", gridcolor=GRID)
            show(style(fig, 420, "Risk map · each dot is a restaurant" + (" (random 3,000 shown)" if len(df) > 3000 else "")), "risk_scatter")
    with r2:
        with st.container(border=True):
            by = st.radio("Break risk down by", list(A.DIMENSIONS), horizontal=True, key="risk_dim")
            col = A.DIMENSIONS[by]
            rb = A.risk_by(df, col)
            rb = rb.loc[rb["High risk"].sort_values().index]
            long = rb.reset_index().melt(id_vars=col, var_name="Tier", value_name="Share")
            fig = px.bar(long, x="Share", y=col, color="Tier", orientation="h", text_auto=".0%",
                         color_discrete_map=RISK_COLORS, category_orders={"Tier": A.RISK_TIERS[::-1]})
            fig.update_xaxes(tickformat=".0%", title_text="", gridcolor=GRID)
            fig.update_yaxes(title_text="")
            show(style(fig, max(320, 42 * len(rb) + 130), f"Risk profile by {by.lower()}"), "risk_dim_bar")

    d1, d2 = st.columns([1.25, 1])
    with d1:
        with st.container(border=True):
            fig = px.histogram(df, x="Diversification", nbins=40, color_discrete_sequence=[TEAL])
            for cutoff, label in A.DIVERSIFICATION_BANDS[:-1]:
                fig.add_vline(x=cutoff, line_dash="dot", line_color="#A9C4C4", annotation_text=label, annotation_position="top")
            fig.update_xaxes(title_text="Channel Diversification Score (normalised entropy)", gridcolor=GRID)
            fig.update_yaxes(title_text="Restaurants", gridcolor=GRID)
            show(style(fig, 330, "Diversification score distribution & benchmark bands"), "div_hist")
    with d2:
        section("Balanced vs high-risk profile")
        prof = (
            df.groupby("RiskTier", observed=True)
            .agg(Restaurants=("RestaurantID", "count"), Avg_orders=("TotalOrders", "mean"),
                 In_store=("InStoreReliance", "mean"), Aggregators=("AggDependence", "mean"),
                 Diversification=("Diversification", "mean"))
            .reset_index()
        )
        prof["In_store"] *= 100
        prof["Aggregators"] *= 100
        table(prof, column_config={
            "Avg_orders": st.column_config.NumberColumn("Avg orders", format="%.0f"),
            "In_store": st.column_config.NumberColumn("In-store %", format="%.1f%%"),
            "Aggregators": st.column_config.NumberColumn("Aggregators %", format="%.1f%%"),
            "Diversification": st.column_config.NumberColumn(format="%.3f"),
        })
        bands = df["DiversificationBand"].value_counts().reindex([b for _, b in A.DIVERSIFICATION_BANDS], fill_value=0).reset_index()
        bands.columns = ["Benchmark band", "Restaurants"]
        bands["Share"] = bands["Restaurants"] / bands["Restaurants"].sum() * 100
        table(bands, column_config={"Share": st.column_config.NumberColumn(format="%.1f%%")})

    section("Restaurants to prioritise", "Highest aggregator dependence first")
    n_show = st.slider("Rows to show", 10, 200, 25, 5, key="topn")
    cols = ["RestaurantName", "CuisineType", "Segment", "Subregion", "TotalOrders", "AggDependence", "TopAggregator",
            "MaxAggShare", "InStoreReliance", "Diversification", "RiskTier"]
    risky = df[df["RiskTier"] == "High risk"].sort_values(["AggDependence", "MaxAggShare"], ascending=False)
    shown = pct(risky[cols].head(n_show), ["AggDependence", "MaxAggShare", "InStoreReliance"])
    table(shown, column_config={
        "TotalOrders": st.column_config.NumberColumn("Orders", format="%d"),
        "AggDependence": st.column_config.ProgressColumn("Aggregator dep.", format="%.1f%%", min_value=0, max_value=100),
        "MaxAggShare": st.column_config.NumberColumn("Top platform %", format="%.1f%%"),
        "InStoreReliance": st.column_config.NumberColumn("In-store %", format="%.1f%%"),
        "Diversification": st.column_config.NumberColumn(format="%.3f"),
    })
    st.download_button("⬇️ Download all high-risk restaurants (CSV)", risky[cols].to_csv(index=False).encode(),
                       file_name="high_risk_restaurants.csv", mime="text/csv")

# --------------------------------------------------------------------------- #
# 5. Profitability
# --------------------------------------------------------------------------- #
with tabs[4]:
    ps = A.profit_summary(df)
    if ps is None:
        callout("Revenue / net-profit columns were not found in the data, so the profitability view is unavailable.", "warn")
    else:
        agg = ps[ps["Channel"].isin(A.AGGREGATORS)]
        ins = ps[ps["Channel"] == "In-Store"].iloc[0]
        kpi_cards(
            [
                ("Net profit / month", f"${ps['Net profit'].sum():,.0f}", "all channels", TEAL),
                ("Aggregator order share", f"{agg['Share of orders'].sum():.1%}", "of all orders", "#D4781A"),
                ("Aggregator profit share", f"{agg['Share of net profit'].sum():.1%}", "of all net profit", "#E0457B"),
                ("In-store profit / order", f"${ins['Profit per order']:.2f}", "walk-in orders", "#34D399"),
                ("Aggregator profit / order", f"${agg['Net profit'].sum() / agg['Orders'].sum():.2f}", "after ~30% commission", "#E0457B"),
            ]
        )
        p1, p2 = st.columns(2)
        with p1:
            with st.container(border=True):
                fig = go.Figure()
                fig.add_bar(name="Share of orders", x=ps["Channel"], y=ps["Share of orders"], marker_color="#94A3B8", text=[f"{v:.1%}" for v in ps["Share of orders"]])
                fig.add_bar(name="Share of net profit", x=ps["Channel"], y=ps["Share of net profit"], marker_color=TEAL, text=[f"{v:.1%}" for v in ps["Share of net profit"]])
                fig.update_layout(barmode="group")
                fig.update_yaxes(tickformat=".0%", gridcolor=GRID)
                show(style(fig, 380, "Volume share vs profit share"), "pf_share")
        with p2:
            with st.container(border=True):
                fig = go.Figure(go.Bar(x=ps["Channel"], y=ps["Profit per order"], marker_color=[CHANNEL_COLORS[c] for c in ps["Channel"]],
                                       text=[f"${v:.2f}" for v in ps["Profit per order"]]))
                fig.update_yaxes(title_text="$ per order", gridcolor=GRID)
                show(style(fig, 380, "Net profit per order by channel"), "pf_order")
        p3, p4 = st.columns(2)
        with p3:
            with st.container(border=True):
                fig = go.Figure(go.Bar(x=ps["Channel"], y=ps["% loss-making restaurants"], marker_color=[CHANNEL_COLORS[c] for c in ps["Channel"]],
                                       text=[f"{v:.1%}" for v in ps["% loss-making restaurants"]]))
                fig.update_yaxes(tickformat=".0%", gridcolor=GRID)
                show(style(fig, 340, "Restaurants losing money on the channel"), "pf_loss")
        with p4:
            section("Channel economics")
            t = ps.copy()
            t["Margin"] *= 100
            t["% loss-making restaurants"] *= 100
            t["Share of orders"] *= 100
            t["Share of net profit"] *= 100
            table(t, column_config={
                "Orders": st.column_config.NumberColumn(format="%d"),
                "Revenue": st.column_config.NumberColumn(format="$%.0f"),
                "Net profit": st.column_config.NumberColumn(format="$%.0f"),
                "Margin": st.column_config.NumberColumn(format="%.1f%%"),
                "Profit per order": st.column_config.NumberColumn(format="$%.2f"),
                "% loss-making restaurants": st.column_config.NumberColumn("% loss-making", format="%.1f%%"),
                "Share of orders": st.column_config.NumberColumn(format="%.1f%%"),
                "Share of net profit": st.column_config.NumberColumn(format="%.1f%%"),
            })
        callout("Net profit follows the dataset's own definition: revenue − COGS − OPEX − commission (aggregators) or − delivery cost (self-delivery).", "info")

# --------------------------------------------------------------------------- #
# 6. Outlook
# --------------------------------------------------------------------------- #
with tabs[5]:
    section("Scenario outlook", "Each restaurant's orders compound at its own monthly GrowthFactor; channel mix inside a restaurant is held constant")
    months = st.slider("Months ahead", 3, 24, 6, key="months")
    pj = A.projection(df, months)
    o1, o2 = st.columns([1.4, 1])
    with o1:
        with st.container(border=True):
            fig = go.Figure()
            for ch in A.CHANNELS:
                fig.add_scatter(x=pj["Month"], y=pj[ch], name=ch, mode="lines+markers", line=dict(color=CHANNEL_COLORS[ch], width=3))
            fig.update_xaxes(title_text="Months from now", dtick=1 if months <= 12 else 2, gridcolor=GRID)
            fig.update_yaxes(title_text="Orders / month", gridcolor=GRID)
            show(style(fig, 400, "Projected monthly orders by channel"), "outlook_lines")
    with o2:
        first, last = pj.iloc[0], pj.iloc[-1]
        kpi_cards(
            [
                ("Orders now", f"{first['Total']:,.0f}", "per month", TEAL),
                (f"Orders in {months} months", f"{last['Total']:,.0f}", f"{last['Total'] / first['Total'] - 1:+.1%}", "#4A86F0"),
                ("Aggregator share", f"{last['Aggregator share']:.1%}", f"{(last['Aggregator share'] - first['Aggregator share']) * 100:+.2f}pp vs now", "#D4781A"),
            ]
        )
        callout("Growth is spread almost evenly across channels, so aggregator dependence will not shrink on its own – "
                "it only falls if restaurants actively shift orders to in-store or self-delivery. This is a scenario, not a forecast.", "warn")
    show_pj = pj[["Month", *A.CHANNELS, "Total"]].round(0)
    table(show_pj, column_config={c: st.column_config.NumberColumn(format="%d") for c in [*A.CHANNELS, "Total"]})

# --------------------------------------------------------------------------- #
# 7. Data & methodology
# --------------------------------------------------------------------------- #
with tabs[6]:
    section("Step 1 · Data validation & consistency checks")
    rep = A.validation_report(clean_df, clean_notes)
    rep["Status"] = rep["Status"].map({"Pass": "✅ Pass", "Warning": "⚠️ Warning", "Fail": "❌ Fail"})
    table(rep)
    st.caption("Order counts are treated as the source of truth; every share, KPI and flag is recomputed from them.")

    section("Detected schema", "New categories in the data appear automatically in filters and charts")
    e1, e2, e3 = st.columns(3)
    for w, title, vals in ((e1, "Subregions", all_sub), (e2, "Cuisines", all_cui), (e3, "Segments", all_seg)):
        with w:
            st.markdown(f"**{title} ({len(vals)})**")
            st.write(", ".join(vals))

    section("Key performance indicators")
    kdef = pd.DataFrame(
        [
            ("Channel Order Share (%)", "Orders in a channel ÷ all orders", "Volume contribution by channel"),
            ("Aggregator Dependence Index", "(Uber Eats + DoorDash orders) ÷ all orders", "Reliance on Uber Eats / DoorDash"),
            ("In-Store Reliance Ratio", "In-store orders ÷ all orders", "Walk-in strength"),
            ("Channel Diversification Score", "Shannon entropy of the 4 channel shares ÷ ln 4 (0–1)", "Operational resilience"),
            ("Subregion Channel Dominance", "Leading channel, its share and lead over #2, per subregion", "Geographic ordering behaviour"),
        ],
        columns=["KPI", "Formula", "Meaning"],
    )
    table(kdef)

    with st.expander("Methodology & assumptions"):
        st.markdown(
            "- **Volume aggregation** by channel, subregion, cuisine and segment uses the four order-count columns.\n"
            "- **Market share** is volume-weighted (total orders), not an average of restaurant percentages.\n"
            "- **Dependency risk** flags restaurants by the thresholds in Settings (defaults: combined aggregators ≥ 70%, "
            "single platform ≥ 50%, elevated ≥ 60%).\n"
            "- **Diversification bands** (≥ 0.95 well diversified, 0.88–0.95 moderate, < 0.88 concentrated) are calibrated to this "
            "dataset's spread and can be edited in `analytics.py`.\n"
            "- **Outlook** compounds each restaurant's GrowthFactor; it is a scenario, not a statistical forecast."
        )

    section("Explore & export", "Search, filter and sort (click a column header) · filtered, with all derived metrics")
    q1, q2 = st.columns([2, 1])
    with q1:
        query = st.text_input("Search restaurant, cuisine, segment or subregion", key="ex_q", placeholder="e.g. korean, ghost kitchen, Parnell")
    with q2:
        tier_pick = st.multiselect("Risk tier", A.RISK_TIERS, default=A.RISK_TIERS, key="ex_tier")
    ex = df[df["RiskTier"].astype(str).isin(tier_pick)]
    if query.strip():
        hay = (ex["RestaurantName"] + " " + ex["CuisineType"] + " " + ex["Segment"] + " " + ex["Subregion"]).str.lower()
        ex = ex[hay.str.contains(query.strip().lower(), regex=False)]
    st.caption(f"{len(ex):,} restaurants match" + (" · showing the first 500" if len(ex) > 500 else ""))
    table(ex.head(500).round(4))
    st.download_button("⬇️ Download filtered data with derived metrics (CSV)", ex.to_csv(index=False).encode(),
                       file_name="skycity_channel_metrics.csv", mime="text/csv")

st.markdown('<div class="foot">SkyCity Auckland · Order Channel Performance &amp; Market Share Analytics</div>', unsafe_allow_html=True)
