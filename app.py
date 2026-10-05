"""
Wine Quality Intelligence — a premium AI-powered wine analytics platform.

Pages
-----
🏠 Dashboard    - hero, live KPIs, AI briefing
📊 Analytics    - filters, charts, correlation map, feature importance
🤖 Prediction   - manual feature sliders -> quality prediction panel
📷 Wine Scanner - OCR label scanner (EasyOCR) with graceful fallback
📈 Insights     - food pairing, taste profile, radar chart, premium/budget
⚙ Settings     - preferences, data source, about

Run with:  streamlit run wine_dashboard.py
Place winequality-red.csv, wine_model.pkl and wine_features.pkl in the
same directory as this file.

Developed by: Drashti Panchal
"""

import re
import io
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="Wine Quality Intelligence",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY = "#6D071A"
SECONDARY = "#8B0000"
ACCENT = "#C9A227"
BG = "#0F1117"
CARD = "#1A1D26"
WHITE = "#F8FAFC"
MUTED = "#9CA8BC"
SUCCESS = "#10B981"
WARNING = "#F59E0B"
DANGER = "#EF4444"

SEQ_WINE = [ACCENT, "#B5473B", SECONDARY, PRIMARY, "#3D0410"]
DIVERGING = [CARD, "#2A2E3C", "#5C1C2E", SECONDARY, PRIMARY, ACCENT]

FEATURES = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "ph", "sulphates", "alcohol",
]

FEATURE_ICONS = {
    "fixed acidity": "🍋", "volatile acidity": "🧪", "citric acid": "🍊",
    "residual sugar": "🍬", "chlorides": "🧂", "free sulfur dioxide": "💨",
    "total sulfur dioxide": "☁️", "density": "⚖️", "ph": "🔬",
    "sulphates": "✨", "alcohol": "🥂",
}


def style_fig(fig, height=None):
    """Apply the house dark/glass styling to a Plotly figure."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=WHITE, size=13),
        title=dict(font=dict(family="Playfair Display, serif", size=20, color=WHITE)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED)),
        margin=dict(l=10, r=10, t=60, b=10),
        hoverlabel=dict(bgcolor=CARD, font_size=13, font_family="Inter, sans-serif",
                         font_color=WHITE, bordercolor=ACCENT),
        transition=dict(duration=450, easing="cubic-in-out"),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.08)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.08)")
    if height:
        fig.update_layout(height=height)
    return fig


st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,500;0,600;0,700;1,500&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {{
    --primary: {PRIMARY};
    --secondary: {SECONDARY};
    --accent: {ACCENT};
    --bg: {BG};
    --card: {CARD};
    --white: {WHITE};
    --muted: {MUTED};
    --success: {SUCCESS};
    --warning: {WARNING};
    --danger: {DANGER};
}}

html, body, .stApp {{
    background: radial-gradient(ellipse 80% 60% at 50% -10%, rgba(109,7,26,0.20), transparent),
                radial-gradient(ellipse 60% 50% at 100% 10%, rgba(201,162,39,0.07), transparent),
                var(--bg);
    color: var(--white);
    font-family: 'Inter', sans-serif;
}}
* {{ scroll-behavior: smooth; }}
#MainMenu, footer, header {{visibility: hidden;}}
.block-container {{ padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1320px; }}
h1, h2, h3, h4 {{ font-family: 'Playfair Display', serif; color: var(--white); letter-spacing: 0.3px; }}
p, span, label, div {{ font-family: 'Inter', sans-serif; }}

::-webkit-scrollbar {{ width: 10px; }}
::-webkit-scrollbar-track {{ background: var(--bg); }}
::-webkit-scrollbar-thumb {{ background: linear-gradient(var(--primary), var(--accent)); border-radius: 10px; }}

@keyframes fadeInUp {{ from {{ opacity:0; transform: translateY(18px); }} to {{ opacity:1; transform: translateY(0); }} }}
@keyframes floatGlass {{ 0%,100% {{ transform: translateY(0) rotate(-3deg); }} 50% {{ transform: translateY(-14px) rotate(3deg); }} }}
@keyframes pulseGlow {{ 0%,100% {{ opacity:0.55; }} 50% {{ opacity:1; }} }}
@keyframes shimmer {{ 0% {{ background-position:-400px 0; }} 100% {{ background-position:400px 0; }} }}
@keyframes gradientMove {{ 0% {{ background-position:0% 50%; }} 50% {{ background-position:100% 50%; }} 100% {{ background-position:0% 50%; }} }}
@keyframes shineSweep {{ 0% {{ transform: translateX(-120%) skewX(-15deg); }} 100% {{ transform: translateX(220%) skewX(-15deg); }} }}
@keyframes barGrow {{ from {{ width:0%; }} }}
@keyframes spinSlow {{ from {{ transform: rotate(0deg); }} to {{ transform: rotate(360deg); }} }}
@keyframes pulseRing {{ 0% {{ box-shadow:0 0 0 0 rgba(201,162,39,0.45); }} 100% {{ box-shadow:0 0 0 16px rgba(201,162,39,0); }} }}

.fade-in {{ animation: fadeInUp 0.7s ease both; }}
.d-0{{animation-delay:0s !important;}} .d-1{{animation-delay:.08s !important;}}
.d-2{{animation-delay:.16s !important;}} .d-3{{animation-delay:.24s !important;}}
.d-4{{animation-delay:.32s !important;}} .d-5{{animation-delay:.40s !important;}}

/* ---------------- HERO ---------------- */
.hero-wrap {{
    position: relative; border-radius: 28px; padding: 56px 48px; margin-bottom: 32px; overflow: hidden;
    background: linear-gradient(135deg, rgba(109,7,26,0.38), rgba(26,29,38,0.6), rgba(201,162,39,0.08), rgba(109,7,26,0.38)),
                linear-gradient(180deg, var(--card), #12141c);
    background-size: 300% 300%, 100% 100%;
    animation: fadeInUp 0.8s ease both, gradientMove 16s ease-in-out infinite;
    border: 1px solid rgba(201,162,39,0.25);
    box-shadow: 0 30px 80px -20px rgba(109,7,26,0.45), inset 0 1px 0 rgba(255,255,255,0.05);
}}
.hero-eyebrow {{
    display:inline-flex; align-items:center; gap:8px; font-family:'JetBrains Mono',monospace; font-size:12px;
    letter-spacing:3px; text-transform:uppercase; color:var(--accent);
    background:rgba(201,162,39,0.08); border:1px solid rgba(201,162,39,0.3); padding:6px 16px; border-radius:100px;
    margin-bottom:20px;
}}
.hero-title {{
    font-size:52px; font-weight:700; line-height:1.08; margin:0 0 12px 0;
    background: linear-gradient(120deg, #FFFFFF 30%, var(--accent) 75%, #E7A0AE 100%);
    -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;
}}
.hero-sub {{ font-size:17px; color:var(--muted); max-width:600px; line-height:1.6; font-style:italic; }}
.hero-glass {{ position:absolute; right:56px; top:50%; transform:translateY(-50%); font-size:100px;
    filter: drop-shadow(0 20px 30px rgba(0,0,0,0.5)); animation: floatGlass 6s ease-in-out infinite; opacity:0.9; }}
@media (max-width:900px) {{ .hero-title{{font-size:34px;}} .hero-glass{{display:none;}} .hero-wrap{{padding:36px 22px;}} }}

/* ---------------- SECTION LABEL ---------------- */
.section-label {{ display:flex; align-items:center; gap:14px; margin:40px 0 18px 0; }}
.section-label .tag {{ font-family:'JetBrains Mono',monospace; font-size:11px; letter-spacing:2.5px; text-transform:uppercase;
    color:var(--accent); border:1px solid rgba(201,162,39,0.35); padding:4px 12px; border-radius:100px;
    background:rgba(201,162,39,0.06); white-space:nowrap; }}
.section-label .title {{ font-size:23px; font-family:'Playfair Display',serif; color:var(--white); margin:0; }}
.section-label .line {{ flex:1; height:1px; background:linear-gradient(90deg, rgba(201,162,39,0.4), transparent); }}

/* ---------------- KPI CARDS ---------------- */
.kpi-card {{
    position:relative; background:linear-gradient(160deg, rgba(255,255,255,0.05), rgba(255,255,255,0.015));
    backdrop-filter:blur(18px); border:1px solid rgba(255,255,255,0.08); border-radius:24px;
    padding:24px 22px 20px 22px; overflow:hidden; animation:fadeInUp 0.7s ease both;
    transition:transform .35s cubic-bezier(.2,.8,.2,1), box-shadow .35s ease, border-color .35s ease;
}}
.kpi-card::after {{ content:""; position:absolute; top:0; left:0; width:60%; height:100%;
    background:linear-gradient(100deg, transparent, rgba(255,255,255,0.16), transparent);
    transform: translateX(-120%) skewX(-15deg); pointer-events:none; }}
.kpi-card:hover::after {{ animation: shineSweep 1s ease; }}
.kpi-card:hover {{ transform: translateY(-8px) scale(1.015);
    box-shadow: 0 24px 50px -10px rgba(109,7,26,0.45), 0 0 0 1px rgba(201,162,39,0.4); border-color: rgba(201,162,39,0.5); }}
.kpi-top {{ display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:12px; }}
.kpi-icon {{ font-size:24px; filter: drop-shadow(0 4px 10px rgba(201,162,39,0.35)); }}
.kpi-badge {{ font-family:'JetBrains Mono',monospace; font-size:10px; color:var(--accent);
    border:1px solid rgba(201,162,39,0.4); padding:2px 8px; border-radius:100px; letter-spacing:1px; }}
.kpi-trend {{ font-size:11px; font-family:'JetBrains Mono',monospace; margin-top:6px; }}
.kpi-trend.up {{ color: var(--success); }}
.kpi-trend.down {{ color: var(--danger); }}
.kpi-value {{ font-family:'JetBrains Mono',monospace; font-size:34px; font-weight:600; color:var(--white); line-height:1; margin-bottom:6px; }}
.kpi-label {{ color:var(--muted); font-size:13px; letter-spacing:.5px; text-transform:uppercase; }}
.kpi-glow-gold .kpi-value {{ color:var(--accent); text-shadow:0 0 18px rgba(201,162,39,0.5); }}
.kpi-glow-rose .kpi-value {{ color:#e7a0ae; }}
.kpi-glow-wine .kpi-value {{ color:#ff6b8a; }}

/* ---------------- GLASS PANEL ---------------- */
.glass-panel {{
    background:linear-gradient(160deg, rgba(255,255,255,0.045), rgba(255,255,255,0.012));
    backdrop-filter:blur(16px); border:1px solid rgba(255,255,255,0.07); border-radius:24px;
    padding:22px 22px 8px 22px; margin-bottom:16px; box-shadow:0 18px 40px -18px rgba(0,0,0,0.6);
    transition:border-color .3s ease; animation:fadeInUp 0.55s ease both;
}}
.glass-panel:hover {{ border-color: rgba(201,162,39,0.22); }}

/* ---------------- INSIGHT CARDS ---------------- */
.insight-card {{
    position:relative; border-radius:22px; padding:22px 22px 20px 22px;
    background:linear-gradient(160deg, rgba(255,255,255,0.05), rgba(255,255,255,0.01));
    border:1px solid rgba(255,255,255,0.08); border-left:3px solid var(--accent); height:100%; overflow:hidden;
    animation:fadeInUp 0.7s ease both; transition:transform .4s cubic-bezier(.2,.8,.2,1), box-shadow .3s ease, border-color .3s ease;
}}
.insight-card:hover {{ transform: translateY(-6px) scale(1.01); box-shadow:0 18px 36px -14px rgba(201,162,39,0.3); border-color: rgba(201,162,39,0.4); }}
.insight-eyebrow {{ font-family:'JetBrains Mono',monospace; font-size:11px; letter-spacing:2px; text-transform:uppercase; color:var(--accent); margin-bottom:10px; display:block; }}
.insight-title {{ font-family:'Playfair Display',serif; font-size:19px; color:var(--white); margin-bottom:8px; }}
.insight-body {{ color:var(--muted); font-size:14px; line-height:1.55; }}

/* ---------------- RANKING BARS ---------------- */
.rank-row {{ display:flex; align-items:center; gap:14px; margin-bottom:14px; padding:6px 8px; border-radius:12px; transition:background .25s ease, transform .25s ease; }}
.rank-row:hover {{ background: rgba(255,255,255,0.04); transform: translateX(4px); }}
.rank-num {{ font-family:'JetBrains Mono',monospace; font-size:13px; color:var(--muted); width:22px; }}
.rank-num.top {{ color:var(--accent); font-weight:700; }}
.rank-label {{ width:175px; font-size:13.5px; color:var(--white); flex-shrink:0; }}
.rank-row:hover .rank-label {{ color: var(--accent); }}
.rank-bar-track {{ flex:1; height:12px; border-radius:100px; background:rgba(255,255,255,0.06); overflow:hidden; position:relative; }}
.rank-bar-fill {{ height:100%; border-radius:100px; background:linear-gradient(90deg, var(--primary), #e7a0ae); animation:barGrow 1.1s cubic-bezier(.2,.8,.2,1) both; }}
.rank-bar-fill.top {{ background:linear-gradient(90deg, var(--accent), #fff3c4); box-shadow:0 0 14px rgba(201,162,39,0.6); }}
.rank-pct {{ font-family:'JetBrains Mono',monospace; font-size:12px; color:var(--muted); width:50px; text-align:right; }}

/* ---------------- SIDEBAR ---------------- */
section[data-testid="stSidebar"] {{ background:linear-gradient(180deg, var(--card), var(--bg)); border-right:1px solid rgba(201,162,39,0.15); }}
section[data-testid="stSidebar"] .block-container {{ padding-top:1.5rem; }}
.sidebar-brand {{ text-align:center; padding:4px 0 18px 0; }}
.sidebar-bottle {{ font-size:44px; animation: floatGlass 7s ease-in-out infinite; display:block; margin-bottom:4px; }}
.sidebar-brand .name {{ font-family:'Playfair Display',serif; font-size:19px; color:var(--accent); letter-spacing:1px; }}
.sidebar-brand .tag {{ font-family:'JetBrains Mono',monospace; font-size:10px; color:var(--muted); letter-spacing:2px; text-transform:uppercase; }}
.sidebar-divider {{ height:1px; background:linear-gradient(90deg, transparent, rgba(201,162,39,0.4), transparent); margin:18px 0 16px 0; }}
.sidebar-section-title {{ font-family:'JetBrains Mono',monospace; font-size:11px; letter-spacing:2px; text-transform:uppercase; color:#e7a0ae; margin-bottom:10px; }}
div[role="radiogroup"] label {{
    background: rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06); border-radius:14px;
    padding:8px 12px !important; margin-bottom:6px; transition: all .25s ease; width:100%;
}}
div[role="radiogroup"] label:hover {{ border-color: rgba(201,162,39,0.5); background: rgba(201,162,39,0.06); }}

/* Streamlit widget overrides */
div[data-baseweb="select"] > div {{ background-color: rgba(255,255,255,0.04) !important; border-color: rgba(201,162,39,0.3) !important; border-radius:14px !important; }}
div[data-baseweb="select"]:hover > div {{ border-color: rgba(201,162,39,0.6) !important; }}
span[data-baseweb="tag"] {{ background-color: var(--primary) !important; border-radius:8px !important; }}
.stSlider [data-baseweb="slider"] > div > div {{ background: rgba(255,255,255,0.08) !important; }}
.stSlider [data-baseweb="slider"] div[role="slider"] {{ background-color: var(--accent) !important; box-shadow:0 0 10px rgba(201,162,39,0.6) !important; }}
.stSlider [data-baseweb="slider"] div[role="slider"]:hover {{ box-shadow: 0 0 0 8px rgba(201,162,39,0.18), 0 0 14px rgba(201,162,39,0.7) !important; transform: scale(1.15) !important; }}
.stSlider [data-baseweb="slider"] > div > div > div {{ background: linear-gradient(90deg, var(--primary), var(--accent)) !important; }}

.stButton > button {{
    position:relative; background:linear-gradient(135deg, var(--primary), var(--secondary)); color:var(--white);
    border:1px solid rgba(201,162,39,0.4); border-radius:14px; font-weight:600; letter-spacing:.3px;
    padding:.6rem 1.1rem; transition: all .25s ease; width:100%;
}}
.stButton > button:hover {{ border-color: var(--accent); box-shadow: 0 8px 22px -6px rgba(201,162,39,0.45); transform: translateY(-2px); color: var(--accent); }}
.stButton > button:active {{ transform: translateY(0) scale(0.97); }}
.stDownloadButton > button {{
    background:linear-gradient(135deg, var(--card), #12141c); color:var(--accent); border:1px solid rgba(201,162,39,0.4);
    border-radius:14px; font-weight:600; width:100%; transition: all .25s ease;
}}
.stDownloadButton > button:hover {{ box-shadow: 0 8px 22px -6px rgba(201,162,39,0.45); transform: translateY(-2px); }}

[data-testid="stDataFrame"] {{ border-radius:18px; overflow:hidden; border:1px solid rgba(255,255,255,0.08); }}
[data-testid="stFileUploadDropzone"] {{
    background: linear-gradient(160deg, rgba(201,162,39,0.05), rgba(109,7,26,0.05)) !important;
    border: 2px dashed rgba(201,162,39,0.35) !important; border-radius: 20px !important;
}}
[data-testid="stFileUploadDropzone"]:hover {{ border-color: var(--accent) !important; }}
[data-testid="stCameraInput"] video {{ border-radius: 18px; }}

/* ---------------- BADGES ---------------- */
.badge {{ display:inline-flex; align-items:center; gap:6px; padding:8px 18px; border-radius:100px; font-weight:700;
    font-size:14px; letter-spacing:.4px; }}
.badge-excellent {{ background: rgba(16,185,129,0.14); color:var(--success); border:1px solid rgba(16,185,129,0.4); }}
.badge-average {{ background: rgba(245,158,11,0.14); color:var(--warning); border:1px solid rgba(245,158,11,0.4); }}
.badge-poor {{ background: rgba(239,68,68,0.14); color:var(--danger); border:1px solid rgba(239,68,68,0.4); }}

/* ---------------- warning card (OCR fallback) ---------------- */
.warn-card {{
    border-radius:20px; padding:22px; background: linear-gradient(160deg, rgba(245,158,11,0.1), rgba(245,158,11,0.02));
    border:1px solid rgba(245,158,11,0.35); animation: fadeInUp .5s ease both;
}}
.warn-card .icon {{ font-size:28px; margin-bottom:8px; }}
.warn-card .title {{ font-family:'Playfair Display',serif; font-size:18px; color:var(--warning); margin-bottom:6px; }}
.warn-card .body {{ color:var(--muted); font-size:14px; line-height:1.55; }}

/* ---------------- scanner result card ---------------- */
.scan-card {{ background: linear-gradient(160deg, rgba(255,255,255,0.05), rgba(255,255,255,0.01));
    border:1px solid rgba(255,255,255,0.08); border-radius:18px; padding:16px 18px; text-align:center;
    transition: transform .3s ease, border-color .3s ease; animation: fadeInUp .55s ease both; }}
.scan-card:hover {{ transform: translateY(-4px); border-color: rgba(201,162,39,0.4); }}
.scan-card .lbl {{ font-size:11px; letter-spacing:1.5px; text-transform:uppercase; color:var(--muted); margin-bottom:6px; }}
.scan-card .val {{ font-family:'Playfair Display',serif; font-size:18px; color:var(--white); }}

/* footer */
.app-footer {{ text-align:center; margin-top:52px; padding:26px 0 8px 0; border-top:1px solid rgba(255,255,255,0.07);
    color:var(--muted); font-family:'JetBrains Mono',monospace; font-size:12px; letter-spacing:1.4px; }}
.app-footer .heart {{ color: var(--secondary); }}
.app-footer .credit {{ color: var(--accent); font-weight:600; }}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_data(path: str = "winequality-red.csv") -> pd.DataFrame:
    data = pd.read_csv(path)
    data.columns = data.columns.str.strip().str.lower() #strip
    return data


@st.cache_resource(show_spinner=False)
def load_model_and_features():
    """Load the pre-trained SVC pipeline and the feature order it expects.
    Returns (model, features, error_message)."""
    model, features, error = None, FEATURES, None
    try:
        model = joblib.load("wine_model.pkl")
    except Exception as exc:
        model = None
        error = f"{type(exc).__name__}: {exc}"
    try:
        loaded_features = joblib.load("wine_features.pkl")
        if isinstance(loaded_features, (list, tuple, np.ndarray)) and len(loaded_features) > 0:
            features = list(loaded_features)
    except Exception:
        pass
    # Normalize casing/whitespace so feature names always match the
    # lowercased dataframe columns (e.g. "pH" -> "ph").
    features = [str(f).strip().lower() for f in features]
    return model, features, error


@st.cache_resource(show_spinner=False)
def train_importance_model(data: pd.DataFrame):
    X = data.drop("quality", axis=1)
    y = data["quality"]
    rf = RandomForestRegressor(n_estimators=150, random_state=42)
    rf.fit(X, y)
    importance = pd.DataFrame(
        {"Feature": X.columns, "Importance": rf.feature_importances_}
    ).sort_values("Importance", ascending=False)
    importance["Importance"] = importance["Importance"] / importance["Importance"].sum()
    return importance


@st.cache_resource(show_spinner=False)
def compute_model_accuracy(_model, data: pd.DataFrame, features):
    """Best-effort accuracy read-out for the hero KPI (evaluated on the
    training data itself, since no held-out split ships with the model)."""
    if _model is None:
        return None
    try:
        X = data[features].to_numpy()
        y = data["quality"]
        preds = _model.predict(X)
        return round(accuracy_score(y, preds) * 100, 1)
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def get_ocr_reader():
    """Lazily build an EasyOCR reader. Returns None if EasyOCR/torch is
    not installed, so the UI can fall back gracefully."""
    try:
        import easyocr
        return easyocr.Reader(["en"], gpu=False, verbose=False)
    except Exception:
        return None


def predict_quality(model, features, values: dict):
    """Run the pipeline on a single sample and return score, confidence,
    and a per-class score distribution (approximate, since the shipped
    SVC was trained with probability=False)."""
    X = np.array([[values[f] for f in features]])
    pred = int(model.predict(X)[0])

    classes = getattr(model.named_steps.get("model", model), "classes_", None)
    dist = None
    confidence = None
    try:
        proba_model = model.named_steps.get("model", model)
        if hasattr(proba_model, "predict_proba"):
            probs = model.predict_proba(X)[0]
            dist = dict(zip(classes, probs))
            confidence = float(np.max(probs) * 100)
        elif hasattr(proba_model, "decision_function"):
            scores = model.decision_function(X)[0]
            scores = np.atleast_1d(scores)
            exp = np.exp(scores - np.max(scores))
            softmax = exp / exp.sum()
            dist = dict(zip(classes, softmax))
            confidence = float(np.max(softmax) * 100)
    except Exception:
        pass

    if confidence is None:
        confidence = 75.0  # neutral fallback so the UI still renders

    return pred, confidence, dist


def quality_badge_html(score: int) -> str:
    if score >= 7:
        return '<span class="badge badge-excellent">🟢 Excellent</span>'
    if score >= 5:
        return '<span class="badge badge-average">🟡 Average</span>'
    return '<span class="badge badge-poor">🔴 Poor</span>'


def stars_for_score(score: int) -> str:
    filled = max(1, min(5, round(score / 10 * 5)))
    return "★" * filled + "☆" * (5 - filled)


def make_gauge(score: int):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": " / 10", "font": {"color": WHITE, "family": "JetBrains Mono"}},
        gauge={
            "axis": {"range": [0, 10], "tickcolor": MUTED},
            "bar": {"color": ACCENT, "thickness": 0.28},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 5], "color": "rgba(239,68,68,0.35)"},
                {"range": [5, 7], "color": "rgba(245,158,11,0.35)"},
                {"range": [7, 10], "color": "rgba(16,185,129,0.35)"},
            ],
            "threshold": {"line": {"color": WHITE, "width": 3}, "thickness": 0.75, "value": score},
        },
    ))
    style_fig(fig, height=260)
    fig.update_layout(margin=dict(l=20, r=20, t=30, b=10))
    return fig


def make_probability_bar(dist: dict):
    classes = list(dist.keys())
    probs = [round(v * 100, 1) for v in dist.values()]
    fig = px.bar(
        x=classes, y=probs, color=probs, color_continuous_scale=[CARD, PRIMARY, ACCENT],
        labels={"x": "Quality class", "y": "Probability (%)"}, title="Class Probability Distribution",
    )
    fig.update_traces(marker_line_width=0)
    fig.update_coloraxes(showscale=False)
    style_fig(fig, height=300)
    return fig


def make_radar(sample: dict, avg: dict):
    cats = list(sample.keys())

    def norm(d):
        vals = []
        for c in cats:
            lo, hi = df[c].min(), df[c].max()
            v = (d[c] - lo) / (hi - lo) if hi > lo else 0.5
            vals.append(v)
        return vals

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=norm(avg), theta=cats, fill="toself", name="Dataset Average",
                                   line_color=MUTED, opacity=0.5))
    fig.add_trace(go.Scatterpolar(r=norm(sample), theta=cats, fill="toself", name="This Wine",
                                   line_color=ACCENT, opacity=0.75))
    fig.update_layout(
        polar=dict(bgcolor="rgba(0,0,0,0)",
                   radialaxis=dict(visible=True, showticklabels=False, gridcolor="rgba(255,255,255,0.08)"),
                   angularaxis=dict(gridcolor="rgba(255,255,255,0.08)")),
        showlegend=True, title="Chemical Profile Radar",
    )
    style_fig(fig, height=420)
    return fig


def render_section_header(tag: str, title: str):
    st.markdown(
        f"""
        <div class="section-label">
            <span class="tag">{tag}</span>
            <h3 class="title">{title}</h3>
            <div class="line"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_row(cards: list):
    cols = st.columns(len(cards))
    for idx, (col, (icon, value, label, glow, trend)) in enumerate(zip(cols, cards)):
        with col:
            trend_html = f'<div class="kpi-trend {trend[1]}">{trend[0]}</div>' if trend else ""
            st.markdown(
                f"""
                <div class="kpi-card {glow} d-{idx}">
                    <div class="kpi-top">
                        <span class="kpi-icon">{icon}</span>
                        <span class="kpi-badge">LIVE</span>
                    </div>
                    <div class="kpi-value">{value}</div>
                    <div class="kpi-label">{label}</div>
                    {trend_html}
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_footer():
    st.markdown(
        """
        <div class="app-footer">
            
            
        </div>
        """,
        unsafe_allow_html=True,
    )


# =====================================================================
# LABEL TEXT PARSING (for the OCR wine scanner)
# =====================================================================
def parse_label_text(text: str) -> dict:
    """Best-effort heuristic extraction of common wine-label fields from
    raw OCR text. Anything not confidently found is left as 'Unknown'."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    joined = " ".join(lines)

    vintage_match = re.search(r"\b(19|20)\d{2}\b", joined)
    vintage = vintage_match.group(0) if vintage_match else "Unknown"

    alcohol_match = re.search(r"(\d{1,2}(?:\.\d)?)\s*%\s*(?:vol|alc|abv)?", joined, re.IGNORECASE)
    alcohol = f"{alcohol_match.group(1)}%" if alcohol_match else "Unknown"

    type_keywords = {
        "red": "Red", "white": "White", "rosé": "Rosé", "rose": "Rosé",
        "sparkling": "Sparkling", "champagne": "Sparkling", "merlot": "Red",
        "cabernet": "Red", "shiraz": "Red", "chardonnay": "White",
        "sauvignon": "White", "pinot noir": "Red", "pinot grigio": "White",
    }
    wine_type = "Unknown"
    for kw, label in type_keywords.items():
        if kw in joined.lower():
            wine_type = label
            break

    country_keywords = [
        "france", "italy", "spain", "india", "usa", "united states", "chile",
        "argentina", "australia", "germany", "portugal", "south africa",
    ]
    country = "Unknown"
    for kw in country_keywords:
        if kw in joined.lower():
            country = kw.title()
            break

    # Heuristic: the longest all-letters line near the top is likely the
    # wine name; the next distinct line is treated as the brand/producer.
    candidate_lines = [l for l in lines if len(l) > 2 and re.search(r"[A-Za-z]", l)]
    name = candidate_lines[0] if candidate_lines else "Unknown"
    brand = candidate_lines[1] if len(candidate_lines) > 1 else "Unknown"

    return {
        "name": name, "brand": brand, "country": country, "vintage": vintage,
        "type": wine_type, "alcohol": alcohol, "raw_text": text,
    }


# =====================================================================
# SESSION STATE
# =====================================================================
if "prediction_count" not in st.session_state:
    st.session_state.prediction_count = 0
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

# =====================================================================
# LOAD RESOURCES
# =====================================================================
try:
    df = load_data("winequality-red.csv")
except FileNotFoundError:
    st.error("Couldn't find **winequality-red.csv**. Place it next to this app and rerun.")
    st.stop()

model, model_features, model_load_error = load_model_and_features()
model_accuracy = compute_model_accuracy(model, df, model_features)

# =====================================================================
# SIDEBAR NAVIGATION
# =====================================================================
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <span class="sidebar-bottle">🍾</span>
            <div class="name">Wine Intelligence</div>
            <div class="tag">AI Powered Platform</div>
        </div>
        <div class="sidebar-divider"></div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="sidebar-section-title">Navigate</div>', unsafe_allow_html=True)
    page = st.radio(
        label="",
        options=["🏠 Dashboard", "📊 Analytics", "🤖 Prediction", "📷 Wine Scanner", "📈 Insights", "⚙ Settings"],
        label_visibility="collapsed",
    )
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="sidebar-section-title">Session</div>
        <div style="color:var(--muted);font-size:12.5px;line-height:1.7;">
            Predictions run: <b style="color:var(--accent);">{st.session_state.prediction_count}</b><br>
            Dataset rows: <b style="color:var(--accent);">{len(df)}</b><br>
            Model status: <b style="color:{'var(--success)' if model is not None else 'var(--danger)'};">
                {'Loaded' if model is not None else 'Missing'}</b>
            {f'<br><span style="color:var(--danger);font-size:11px;">{model_load_error}</span>' if model_load_error else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )

# =====================================================================
# PAGE: DASHBOARD
# =====================================================================
if page == "🏠 Dashboard":
    st.markdown(
        f"""
        <div class="hero-wrap">
            <div class="hero-eyebrow">● Live Intelligence</div>
            <h1 class="hero-title">🍷 Wine Quality Prediction</h1>
            <div class="hero-sub">AI Powered Wine Intelligence Platform</div>
            <div class="hero-glass">🍷</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_kpi_row([
        ("📅", datetime.now().strftime("%b %d, %Y"), "Current Date", "kpi-glow-rose", None),
        ("🎯", f"{model_accuracy if model_accuracy is not None else '—'}%", "Model Accuracy", "kpi-glow-gold", None),
        ("🍇", f"{len(df):,}", "Dataset Size", "kpi-glow-wine", None),
        ("🔮", f"{st.session_state.prediction_count}", "Total Predictions", "kpi-glow-gold", None),
    ])

    render_section_header("Sommelier AI", "Intelligence Briefing")
    avg_quality = round(df["quality"].mean(), 2)
    if avg_quality >= 7:
        summary = "The full dataset skews premium — most vintages cluster in the top quality band."
    elif avg_quality >= 5.5:
        summary = "A solidly drinkable, everyday-quality dataset with several standout bottles."
    else:
        summary = "The dataset trends toward lower quality scores, with room for improvement."

    i1, i2 = st.columns(2)
    with i1:
        st.markdown(
            f"""
            <div class="insight-card">
                <span class="insight-eyebrow">📊 Overview</span>
                <div class="insight-title">Dataset Summary</div>
                <div class="insight-body">{summary} Average quality across all {len(df)} samples is
                <b style="color:var(--accent);">{avg_quality}</b> out of 10.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with i2:
        st.markdown(
            f"""
            <div class="insight-card">
                <span class="insight-eyebrow">🤖 Model</span>
                <div class="insight-title">Prediction Engine</div>
                <div class="insight-body">A trained SVC pipeline (StandardScaler + SVC) is loaded and ready.
                Head to the <b>Prediction</b> tab to score a custom sample, or the <b>Wine Scanner</b> tab
                to read a label photo automatically.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    render_footer()

# =====================================================================
# PAGE: ANALYTICS
# =====================================================================
elif page == "📊 Analytics":
    st.markdown('<h2 class="fade-in">📊 Wine Analytics</h2>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section-title">Filters</div>', unsafe_allow_html=True)
        quality_filter = st.multiselect(
            "Select Quality", options=sorted(df["quality"].unique()), default=sorted(df["quality"].unique())
        )
        alcohol_filter = st.slider(
            "Alcohol Range", float(df["alcohol"].min()), float(df["alcohol"].max()),
            (float(df["alcohol"].min()), float(df["alcohol"].max())),
        )

    filtered_df = df[
        (df["quality"].isin(quality_filter))
        & (df["alcohol"] >= alcohol_filter[0])
        & (df["alcohol"] <= alcohol_filter[1])
    ]

    total_wines = filtered_df.shape[0]
    avg_quality = round(filtered_df["quality"].mean(), 2) if total_wines else 0
    avg_alcohol = round(filtered_df["alcohol"].mean(), 2) if total_wines else 0
    max_quality = filtered_df["quality"].max() if total_wines else 0

    render_kpi_row([
        ("🍷", total_wines, "Total Wines", "kpi-glow-rose", None),
        ("⭐", avg_quality, "Average Quality", "kpi-glow-gold", None),
        ("🥂", f"{avg_alcohol}%", "Average Alcohol", "kpi-glow-wine", None),
        ("🏆", max_quality, "Best Quality", "kpi-glow-gold", None),
    ])

    render_section_header("Tasting Notes", "Quality & Alcohol Relationships")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        fig1 = px.scatter(filtered_df, x="alcohol", y="quality", size="citric acid", color="quality",
                           hover_data=["ph", "sulphates"], color_continuous_scale=SEQ_WINE,
                           title="Alcohol vs Quality", size_max=18)
        style_fig(fig1, height=380)
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        fig2 = px.histogram(filtered_df, x="quality", color="quality", color_discrete_sequence=SEQ_WINE,
                             title="Quality Distribution")
        style_fig(fig2, height=380)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        avg_by_alcohol = filtered_df.groupby("quality")["alcohol"].mean().reset_index()
        fig3 = px.bar(avg_by_alcohol, x="quality", y="alcohol", color="alcohol",
                       color_continuous_scale=[CARD, PRIMARY, ACCENT], title="Average Alcohol by Quality")
        style_fig(fig3, height=360)
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        fig4 = px.box(filtered_df, x="quality", y="alcohol", color="quality", color_discrete_sequence=SEQ_WINE,
                       title="Alcohol Spread by Quality")
        style_fig(fig4, height=360)
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    render_section_header("Chemistry", "Correlation Map")
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    corr = filtered_df.corr(numeric_only=True)
    fig5 = px.imshow(corr, text_auto=".2f", color_continuous_scale=DIVERGING, title="Feature Correlation",
                      aspect="auto")
    fig5.update_traces(hovertemplate="%{x} ↔ %{y}<br>r = %{z:.2f}<extra></extra>", textfont=dict(size=10))
    style_fig(fig5, height=540)
    st.plotly_chart(fig5, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    render_section_header("Model Insight", "Factors Affecting Wine Quality")
    importance_df = train_importance_model(df)
    max_importance = importance_df["Importance"].max()
    rows_html = []
    for i, row in enumerate(importance_df.itertuples(), start=1):
        pct_of_max = (row.Importance / max_importance) * 100
        top_class = "top" if i == 1 else ""
        rows_html.append(
            f'<div class="rank-row"><div class="rank-num {top_class}">{i:02d}</div>'
            f'<div class="rank-label">{row.Feature.title()}</div>'
            f'<div class="rank-bar-track"><div class="rank-bar-fill {top_class}" style="width:{pct_of_max:.1f}%;"></div></div>'
            f'<div class="rank-pct">{row.Importance * 100:.1f}%</div></div>'
        )
    st.markdown('<div class="glass-panel"><div style="padding:8px 4px;">' + "".join(rows_html) + "</div></div>",
                unsafe_allow_html=True)

    render_section_header("Ledger", "Dataset Preview")
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.dataframe(filtered_df, use_container_width=True, height=380)
    st.markdown("</div>", unsafe_allow_html=True)

    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button("⬇ Export filtered CSV", data=filtered_df.to_csv(index=False).encode("utf-8"),
                            file_name="wine_quality_filtered.csv", mime="text/csv", use_container_width=True)
    with dl2:
        report_text = (
            f"WINE QUALITY INTELLIGENCE — SNAPSHOT REPORT\n"
            f"Generated: {datetime.now().strftime('%B %d, %Y %H:%M')}\n\n"
            f"Total wines in view : {total_wines}\n"
            f"Average quality      : {avg_quality}\n"
            f"Average alcohol      : {avg_alcohol}%\n"
            f"Best quality score   : {max_quality}\n"
        )
        st.download_button("📄 Download report", data=report_text.encode("utf-8"),
                            file_name="wine_quality_report.txt", mime="text/plain", use_container_width=True)

    render_footer()

# =====================================================================
# PAGE: PREDICTION
# =====================================================================
elif page == "🤖 Prediction":
    st.markdown('<h2 class="fade-in">🤖 Wine Quality Prediction</h2>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color:var(--muted);margin-top:-10px;">Adjust the chemical properties below and let the '
        'model score the wine.</p>', unsafe_allow_html=True,
    )

    if model is None:
        error_line = f'<div class="body" style="margin-top:10px;"><code>{model_load_error}</code></div>' if model_load_error else ""
        st.markdown(
            f"""
            <div class="warn-card">
                <div class="icon">⚠️</div>
                <div class="title">Model Not Found</div>
                <div class="body">wine_model.pkl could not be loaded. Place it in the same folder as this
                app and rerun to enable predictions.</div>
                {error_line}
            </div>
            """,
              unsafe_allow_html=True,
        )
        st.stop()

    render_section_header("Input", "Chemical Properties")
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    input_values = {}
    cols = st.columns(3)
    for i, feature in enumerate(model_features):
        lo, hi, mean = float(df[feature].min()), float(df[feature].max()), float(df[feature].mean())
        with cols[i % 3]:
            input_values[feature] = st.slider(
                f"{FEATURE_ICONS.get(feature, '🍷')} {feature.title()}", lo, hi, mean,
                step=round((hi - lo) / 100, 4) if hi > lo else 0.01,
            )
    st.markdown("</div>", unsafe_allow_html=True)

    predict_clicked = st.button("🔮 Predict Wine Quality", use_container_width=True)

    if predict_clicked:
        st.session_state.prediction_count += 1
        score, confidence, dist = predict_quality(model, model_features, input_values)
        st.session_state.last_prediction = {"score": score, "confidence": confidence, "dist": dist,
                                             "values": input_values}

    if st.session_state.last_prediction:
        result = st.session_state.last_prediction
        score, confidence, dist = result["score"], result["confidence"], result["dist"]

        render_section_header("Result", "Prediction Panel")
        r1, r2, r3, r4 = st.columns(4)
        with r1:
            st.markdown(
                f'<div class="scan-card"><div class="lbl">Quality Score</div>'
                f'<div class="val">{score} / 10</div></div>', unsafe_allow_html=True,
            )
        with r2:
            st.markdown(
                f'<div class="scan-card"><div class="lbl">Confidence</div>'
                f'<div class="val">{confidence:.1f}%</div></div>', unsafe_allow_html=True,
            )
        with r3:
            st.markdown(
                f'<div class="scan-card"><div class="lbl">Status</div>'
                f'<div class="val">{quality_badge_html(score)}</div></div>', unsafe_allow_html=True,
            )
        with r4:
            st.markdown(
                f'<div class="scan-card"><div class="lbl">Wine Rating</div>'
                f'<div class="val" style="color:var(--accent);">{stars_for_score(score)}</div></div>',
                unsafe_allow_html=True,
            )

        if score >= 7:
            recommendation = "Excellent balance — suitable for premium tastings and cellar-worthy collections."
        elif score >= 5:
            recommendation = "Good everyday quality — pairs well with casual meals and gatherings."
        else:
            recommendation = "Below-average profile — consider adjusting acidity, sulphates, or alcohol balance."

        st.markdown(
            f"""
            <div class="insight-card" style="margin-top:14px;">
                <span class="insight-eyebrow">💡 Recommendation</span>
                <div class="insight-body">{recommendation}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        g1, g2 = st.columns(2)
        with g1:
            st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
            st.plotly_chart(make_gauge(score), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with g2:
            st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
            if dist:
                st.plotly_chart(make_probability_bar(dist), use_container_width=True)
            else:
                st.info("Probability distribution unavailable for this model configuration.")
            st.markdown("</div>", unsafe_allow_html=True)

    render_footer()

# =====================================================================
# PAGE: WINE SCANNER
# =====================================================================
elif page == "📷 Wine Scanner":
    st.markdown('<h2 class="fade-in">📷 Wine Label Scanner</h2>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color:var(--muted);margin-top:-10px;">Upload a photo of a wine label or use your camera — '
        'AI OCR will try to read the details.</p>', unsafe_allow_html=True,
    )

    tab1, tab2 = st.tabs(["⬆ Upload Image", "📸 Camera Capture"])
    image_bytes = None
    with tab1:
        uploaded = st.file_uploader("Drag & drop a wine label image", type=["png", "jpg", "jpeg"])
        if uploaded is not None:
            image_bytes = uploaded.read()
    with tab2:
        captured = st.camera_input("Take a photo of the label")
        if captured is not None:
            image_bytes = captured.getvalue()

    if image_bytes:
        render_section_header("Preview", "Captured Label")
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.image(image_bytes, use_container_width=True, caption="Uploaded wine label")
        st.markdown("</div>", unsafe_allow_html=True)

        with st.spinner("🍷 Analyzing Wine... reading label with AI OCR"):
            reader = get_ocr_reader()
            ocr_text = ""
            if reader is not None:
                try:
                    from PIL import Image
                    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    result = reader.readtext(np.array(img), detail=0)
                    ocr_text = "\n".join(result)
                except Exception:
                    ocr_text = ""

        if not ocr_text.strip():
            st.markdown(
                """
                <div class="warn-card">
                    <div class="icon">⚠️</div>
                    <div class="title">OCR Could Not Read This Label</div>
                    <div class="body">Either EasyOCR is not installed in this environment, or the text on the
                    label wasn't clear enough to extract. Try a sharper, well-lit photo — or install
                    <code>easyocr</code> (<code>pip install easyocr</code>) to enable automatic reading.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            parsed = parse_label_text(ocr_text)
            render_section_header("Result", "Extracted Wine Details")
            fields = [
                ("🍷", "Wine Name", parsed["name"]), ("🏷", "Brand", parsed["brand"]),
                ("🌍", "Country", parsed["country"]), ("📅", "Vintage", parsed["vintage"]),
                ("🍇", "Type", parsed["type"]), ("🥂", "Alcohol %", parsed["alcohol"]),
            ]
            cols = st.columns(3)
            for i, (icon, label, val) in enumerate(fields):
                with cols[i % 3]:
                    st.markdown(
                        f'<div class="scan-card"><div class="lbl">{icon} {label}</div>'
                        f'<div class="val">{val}</div></div>', unsafe_allow_html=True,
                    )

            with st.expander("📄 Raw OCR text"):
                st.text(ocr_text)

            if model is not None and parsed["alcohol"] != "Unknown":
                try:
                    alcohol_val = float(re.sub("[^0-9.]", "", parsed["alcohol"]))
                    sample = {f: float(df[f].mean()) for f in model_features}
                    sample["alcohol"] = alcohol_val
                    score, confidence, dist = predict_quality(model, model_features, sample)
                    st.markdown(
                        f"""
                        <div class="insight-card" style="margin-top:14px;">
                            <span class="insight-eyebrow">⭐ Estimated Quality</span>
                            <div class="insight-body">Based on the detected alcohol content and dataset averages
                            for other properties, the model estimates a quality score of
                            <b style="color:var(--accent);">{score}/10</b> ({confidence:.1f}% confidence)
                            {quality_badge_html(score)}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                except Exception:
                    pass
    else:
        st.info("Upload an image or use the camera tab above to begin scanning a wine label.")

    render_footer()

# =====================================================================
# PAGE: INSIGHTS
# =====================================================================
elif page == "📈 Insights":
    st.markdown('<h2 class="fade-in">📈 AI Insights</h2>', unsafe_allow_html=True)

    importance_df = train_importance_model(df)
    top3 = importance_df.head(3).reset_index(drop=True)

    render_section_header("Model Insight", "Top Quality Drivers")
    st.markdown(
        f"""
        <div class="insight-card">
            <span class="insight-eyebrow">🏆 Feature Importance</span>
            <div class="insight-body">
                <b style="color:var(--accent);">1.</b> {top3.iloc[0]['Feature'].title()} &nbsp;·&nbsp;
                <b style="color:#e7a0ae;">2.</b> {top3.iloc[1]['Feature'].title()} &nbsp;·&nbsp;
                <b style="color:var(--muted);">3.</b> {top3.iloc[2]['Feature'].title()}
                <br><br>These three chemical properties carry the most predictive weight in the model.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    avg_alcohol = df["alcohol"].mean()
    avg_sugar = df["residual sugar"].mean()
    avg_acidity = df["fixed acidity"].mean()
    avg_sulphates = df["sulphates"].mean()

    sample = st.session_state.last_prediction["values"] if st.session_state.last_prediction else {
        f: float(df[f].mean()) for f in FEATURES
    }
    avg_profile = {f: float(df[f].mean()) for f in FEATURES}

    if sample["residual sugar"] > avg_sugar * 1.15:
        taste = "Noticeably sweet, with a soft, rounded finish."
    elif sample["fixed acidity"] > avg_acidity * 1.1:
        taste = "Crisp and tart, with a bright acidic edge."
    else:
        taste = "Balanced and smooth, with medium body."

    serving_temp = "14–16°C (cellar temperature)" if sample["alcohol"] >= avg_alcohol else "10–12°C (lightly chilled)"

    if sample["alcohol"] >= avg_alcohol and sample["sulphates"] >= avg_sulphates:
        pairing = "Grilled red meats, aged cheeses, or dark chocolate."
    else:
        pairing = "Light pasta dishes, poultry, or soft cheeses."

    tier = "Premium" if (sample.get("alcohol", 0) >= avg_alcohol and sample.get("sulphates", 0) >= avg_sulphates) else "Budget-Friendly"

    render_section_header("Sommelier AI", "Taste & Pairing Profile")
    p1, p2, p3, p4 = st.columns(4)
    for col, icon, label, val in [
        (p1, "🍇", "Estimated Taste", taste), (p2, "🌡", "Serving Temperature", serving_temp),
        (p3, "🍽", "Food Pairing", pairing), (p4, "💎", "Tier", tier),
    ]:
        with col:
            st.markdown(
                f'<div class="insight-card"><span class="insight-eyebrow">{icon} {label}</span>'
                f'<div class="insight-body">{val}</div></div>', unsafe_allow_html=True,
            )

    render_section_header("Chemistry", "Chemical Profile Radar")
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.plotly_chart(make_radar(sample, avg_profile), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.caption("Radar compares the most recent prediction sample (or dataset defaults) against the dataset average, min-max normalized per feature.")

    render_footer()

# =====================================================================
# PAGE: SETTINGS
# =====================================================================
elif page == "⚙ Settings":
    st.markdown('<h2 class="fade-in">⚙ Settings</h2>', unsafe_allow_html=True)

    render_section_header("Preferences", "Application Settings")
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    s1, s2 = st.columns(2)
    with s1:
        st.toggle("Enable animations", value=True, help="Purely cosmetic in this build — kept for future theming.")
        st.toggle("Show raw OCR text by default", value=False)
    with s2:
        st.selectbox("Chart color theme", ["Wine (default)", "Monochrome Gold", "Slate"])
        st.selectbox("Units", ["Metric (default)", "Imperial"])
    st.markdown("</div>", unsafe_allow_html=True)

    render_section_header("Data Source", "Files & Model")
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="color:var(--muted);font-size:14px;line-height:1.8;">
            📄 Dataset file: <b style="color:var(--white);">winequality-red.csv</b> ({len(df)} rows)<br>
            🤖 Model file: <b style="color:var(--white);">wine_model.pkl</b>
            ({'loaded' if model is not None else 'not found'})<br>
            {f'<span style="color:var(--danger);">Load error: {model_load_error}</span><br>' if model_load_error else ''}
            🧬 Feature order: <b style="color:var(--white);">{', '.join(model_features)}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🔄 Reset session (clear prediction history)"):
        st.session_state.prediction_count = 0
        st.session_state.last_prediction = None
        st.success("Session reset.")

    render_section_header("About", "Wine Quality Intelligence")
    st.markdown(
        """
        <div class="insight-card">
            <span class="insight-eyebrow">ℹ️ About</span>
            <div class="insight-body">
                Wine Quality Intelligence is an AI-powered analytics and prediction platform built on the
                UCI Red Wine Quality dataset. It combines exploratory analytics, a trained SVC prediction
                pipeline, an OCR-based label scanner, and AI-generated tasting insights into a single
                premium dashboard.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_footer()