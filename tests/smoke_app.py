"""Headless smoke test for app.py using stub streamlit/plotly modules.

It cannot validate rendering, but it executes every line of app.py (all tabs, all channel views)
against real data so pandas/logic/name errors surface.  Run:  python tests/smoke_app.py
"""
import io
import runpy
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class Ctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __getattr__(self, name):
        return lambda *a, **k: Ctx()


class Stop(Exception):
    pass


def build_streamlit(radio_choice=0, uploads=None, mode_idx=0, sub_subset=None):
    st = types.ModuleType("streamlit")
    st.__version__ = "1.50.0"
    st.calls = {"plotly_chart": 0, "dataframe": 0, "markdown": 0, "errors": []}

    def deco(*a, **k):
        if a and callable(a[0]) and not k:
            f = a[0]
            f.clear = lambda: None
            return f
        return lambda f: deco(f)

    st.cache_data = deco
    st.cache_data.clear = lambda: None
    st.cache_resource = deco
    st.set_page_config = lambda **k: None

    def markdown(*a, **k):
        st.calls["markdown"] += 1

    st.markdown = markdown
    st.sidebar = Ctx()
    st.sidebar.warning = lambda *a, **k: st.calls["errors"].append(("sidebar.warning", a))
    st.sidebar.error = lambda *a, **k: st.calls["errors"].append(("sidebar.error", a))

    def radio(label, options, **k):
        options = list(options)
        if label.startswith("Data source"):
            return options[mode_idx]
        if label.startswith("Channel view"):
            return options[radio_choice]
        return options[0]

    st.radio = radio
    st.file_uploader = lambda *a, **k: uploads or []
    st.button = lambda *a, **k: False
    st.slider = lambda label, lo, hi, value, step=1, **k: value
    st.multiselect = lambda label, options, default=None, **k: (
        [o for o in options if sub_subset is None or label != "Subregion" or o in sub_subset] if default is not None and label == "Subregion" else (default or [])
    )
    st.session_state = {}
    st.text_input = lambda *a, **k: ''
    st.tabs = lambda labels: [Ctx() for _ in labels]
    st.columns = lambda spec, **k: [Ctx() for _ in (spec if isinstance(spec, (list, tuple)) else range(spec))]
    st.container = lambda **k: Ctx()
    st.expander = lambda *a, **k: Ctx()

    def plotly_chart(fig, key=None, **kw):
        st.calls["plotly_chart"] += 1

    st.plotly_chart = plotly_chart

    def dataframe(df, **k):
        st.calls["dataframe"] += 1
        assert hasattr(df, "shape")

    st.dataframe = dataframe
    st.download_button = lambda *a, **k: False
    st.caption = st.write = lambda *a, **k: None
    st.error = lambda *a, **k: st.calls["errors"].append(("error", a))
    st.warning = lambda *a, **k: st.calls["errors"].append(("warning", a))

    def stop():
        raise Stop()

    st.stop = stop
    st.rerun = lambda: None
    st.column_config = types.SimpleNamespace(
        NumberColumn=lambda *a, **k: ("num", a, k), ProgressColumn=lambda *a, **k: ("prog", a, k)
    )
    return st


def run(**kw):
    st = build_streamlit(**kw)
    sys.modules["streamlit"] = st
    px, go, plotly = MagicMock(name="px"), MagicMock(name="go"), types.ModuleType("plotly")
    plotly.express, plotly.graph_objects = px, go
    sys.modules.update({"plotly": plotly, "plotly.express": px, "plotly.graph_objects": go})
    sys.modules.pop("analytics", None)
    try:
        runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
    except Stop:
        st.calls["stopped"] = True
    return st


if __name__ == "__main__":
    import pandas as pd

    csv = next((ROOT / "data").glob("*.csv"))
    for idx, name in enumerate(["All channels", "In-Store only", "Delivery only"]):
        st = run(radio_choice=idx)
        assert not st.calls.get("stopped") and not st.calls["errors"], (name, st.calls)
        print(f"OK  view={name:14s} charts={st.calls['plotly_chart']} tables={st.calls['dataframe']}")

    # upload that adds a brand-new subregion, cuisine and segment (append mode)
    raw = pd.read_csv(csv).head(40).copy()
    raw["RestaurantID"] += 100000
    raw["Subregion"] = "Waiheke"
    raw["CuisineType"] = "Korean"
    raw["Segment"] = "Food Truck"
    up = types.SimpleNamespace(name="new_entries.csv", getvalue=lambda: raw.to_csv(index=False).encode())
    st = run(uploads=[up], mode_idx=2)
    assert not st.calls["errors"], st.calls
    print("OK  append upload with new subregion/cuisine/segment")

    # upload with a missing column must be rejected gracefully
    bad = types.SimpleNamespace(name="bad.csv", getvalue=lambda: raw.drop(columns=["Subregion"]).to_csv(index=False).encode())
    st = run(uploads=[bad], mode_idx=1)
    assert any(c[0] == "error" for c in st.calls["errors"]), st.calls
    print("OK  bad upload rejected gracefully:", st.calls["errors"][0][1][0][:70])

    # replace mode with a good upload
    st = run(uploads=[up], mode_idx=1)
    assert not st.calls["errors"], st.calls
    print("OK  replace upload")

    # a single-subregion filter must still work
    st = run(sub_subset=["CBD"])
    assert not st.calls["errors"], st.calls
    print("OK  single-subregion filter")
    print("\nsmoke tests passed")
