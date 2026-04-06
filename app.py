import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import io
import json
from datetime import datetime

# ── PAGE CONFIG ──────────────────────────────────────────
st.set_page_config(
    page_title="DataPrep Studio",
    layout="wide",
    page_icon="⬡",
    initial_sidebar_state="expanded",
)

# ── SESSION STATE ────────────────────────────────────────
def _init():
    defaults = {
        "raw_df":       None,
        "current_df":   None,
        "log":          [],
        "file_name":    "",
        "theme":        "Light",
        "chart_colors": ["#4F6CF7"],   # default = 1 color
        "num_colors":   1,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
_init()

# ── THEME TOKENS ─────────────────────────────────────────
THEMES = {
    "Light": {
        "bg":         "#F8F9FC",
        "surface":    "#FFFFFF",
        "surface2":   "#F1F3F9",
        "border":     "#E2E5EF",
        "border2":    "#C8CCDB",
        "accent":     "#4F6CF7",
        "accent2":    "#7289FA",
        "green":      "#059669",
        "amber":      "#D97706",
        "red":        "#DC2626",
        "text":       "#111827",
        "text2":      "#4B5563",
        "text3":      "#9CA3AF",
        "shadow":     "rgba(0,0,0,0.07)",
        "plot_bg":    "#F8F9FC",
        "grid":       "#E5E7EB",
        "font_color": "#4B5563",
    },
    "Dark": {
        "bg":         "#0B0D11",
        "surface":    "#111318",
        "surface2":   "#181B22",
        "border":     "#252830",
        "border2":    "#2E3240",
        "accent":     "#5B6AF0",
        "accent2":    "#7C8FF5",
        "green":      "#34D399",
        "amber":      "#FBBF24",
        "red":        "#F87171",
        "text":       "#E8EAF0",
        "text2":      "#8B90A0",
        "text3":      "#555A6A",
        "shadow":     "rgba(0,0,0,0.5)",
        "plot_bg":    "#0B0D11",
        "grid":       "#252830",
        "font_color": "#8B90A0",
    },
    "Warm Sand": {
        "bg":         "#FAF6EF",
        "surface":    "#FFFDF8",
        "surface2":   "#F2EBE0",
        "border":     "#DDD0BE",
        "border2":    "#C9B99A",
        "accent":     "#A0522D",
        "accent2":    "#C0652A",
        "green":      "#3B7A57",
        "amber":      "#C17B1A",
        "red":        "#B22222",
        "text":       "#2C1A0E",
        "text2":      "#5C4033",
        "text3":      "#9E8272",
        "shadow":     "rgba(80,40,10,0.08)",
        "plot_bg":    "#FAF6EF",
        "grid":       "#E8DDD0",
        "font_color": "#5C4033",
    },
    "Slate Blue": {
        "bg":         "#E8EDF8",
        "surface":    "#F4F6FF",
        "surface2":   "#DCE2F5",
        "border":     "#B8C3E8",
        "border2":    "#8FA0D8",
        "accent":     "#2D3A9E",
        "accent2":    "#3D4DBE",
        "green":      "#1A6B4A",
        "amber":      "#7A5200",
        "red":        "#8B1A1A",
        "text":       "#0A0F2E",
        "text2":      "#2A3260",
        "text3":      "#7080B0",
        "shadow":     "rgba(20,30,100,0.1)",
        "plot_bg":    "#E8EDF8",
        "grid":       "#C0CCEE",
        "font_color": "#2A3260",
    },
}

T = THEMES[st.session_state.theme]

# ── CSS INJECTION ────────────────────────────────────────
# We inject CSS via a <script> tag that builds a <style> element at runtime.
# This is necessary because Streamlit's markdown renderer can expose the raw
# text of a <style> block as visible content on the page. Using script injection
# bypasses the markdown parser entirely so nothing leaks as visible text.

def _make_css(t):
    return (
        # Google Fonts loaded via @import inside the injected style
        "@import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,300;9..40,400;"
        "9..40,500;9..40,600;9..40,700&family=JetBrains+Mono:wght@300;400;500&display=swap');"

        # ── full-page background ──
        "html,body{background-color:" + t["bg"] + " !important;}"
        ".stApp,.stApp>div,"
        "[data-testid='stAppViewContainer'],"
        "[data-testid='stAppViewBlockContainer'],"
        "[data-testid='stMain'],"
        "[data-testid='stMainBlockContainer'],"
        ".main,.main>div,.block-container{"
        "background-color:" + t["bg"] + " !important;}"
        ".block-container{padding:2rem 2.5rem 4rem !important;max-width:1400px !important;}"

        # ── sidebar (all wrapper layers) ──
        "section[data-testid='stSidebar'],"
        "section[data-testid='stSidebar']>div,"
        "section[data-testid='stSidebar']>div>div,"
        "section[data-testid='stSidebar']>div>div>div{"
        "background-color:" + t["surface"] + " !important;"
        "border-right:1px solid " + t["border"] + " !important;}"

        # ── sidebar toggle always visible ──
        "[data-testid='collapsedControl']{"
        "display:flex !important;visibility:visible !important;"
        "opacity:1 !important;pointer-events:auto !important;"
        "background:" + t["surface"] + " !important;"
        "border:1px solid " + t["border2"] + " !important;"
        "border-radius:50% !important;"
        "box-shadow:0 2px 8px " + t["shadow"] + " !important;"
        "z-index:9999 !important;}"

        # ── typography ──
        "html,body,p,div,span,li,label{font-family:'DM Sans',sans-serif !important;}"
        "h1,h2,h3,h4,h5{"
        "font-family:'DM Sans',sans-serif !important;"
        "font-weight:700 !important;letter-spacing:-0.01em !important;"
        "color:" + t["text"] + " !important;line-height:1.2 !important;}"

        # ── hide chrome ──
        "#MainMenu,footer{visibility:hidden;}"
        ".stDeployButton{display:none;}"
        "header[data-testid='stHeader']{background:transparent !important;}"

        # ── metric cards ──
        "[data-testid='stMetric']{"
        "background:" + t["surface"] + " !important;"
        "border:1px solid " + t["border"] + " !important;"
        "border-radius:12px !important;padding:1.1rem 1.3rem !important;"
        "box-shadow:0 2px 8px " + t["shadow"] + " !important;}"
        "[data-testid='stMetricLabel'] p{"
        "font-family:'JetBrains Mono',monospace !important;"
        "font-size:0.68rem !important;text-transform:uppercase;"
        "letter-spacing:0.09em;color:" + t["text3"] + " !important;}"
        "[data-testid='stMetricValue']{"
        "font-family:'JetBrains Mono',monospace !important;"
        "font-size:1.6rem !important;font-weight:500 !important;"
        "color:" + t["text"] + " !important;}"

        # ── dataframe ──
        "[data-testid='stDataFrame']{"
        "border:1px solid " + t["border"] + " !important;"
        "border-radius:12px !important;overflow:hidden;"
        "background:" + t["surface"] + " !important;}"

        # ── buttons ──
        ".stButton>button{"
        "font-family:'DM Sans',sans-serif !important;"
        "font-size:0.85rem !important;font-weight:600 !important;"
        "background:" + t["accent"] + " !important;color:#fff !important;"
        "border:none !important;border-radius:8px !important;"
        "padding:0.45rem 1.2rem !important;"
        "box-shadow:0 2px 6px " + t["shadow"] + " !important;}"
        ".stButton>button:hover{"
        "background:" + t["accent2"] + " !important;transform:translateY(-1px);}"
        ".danger-btn .stButton>button{"
        "background:transparent !important;"
        "border:1px solid " + t["red"] + " !important;"
        "color:" + t["red"] + " !important;box-shadow:none !important;}"
        ".danger-btn .stButton>button:hover{"
        "background:" + t["red"] + " !important;color:#fff !important;}"

        # ── download buttons ──
        ".stDownloadButton>button{"
        "font-family:'DM Sans',sans-serif !important;font-size:0.85rem !important;"
        "background:transparent !important;"
        "border:1px solid " + t["border2"] + " !important;"
        "color:" + t["text2"] + " !important;border-radius:8px !important;}"
        ".stDownloadButton>button:hover{"
        "border-color:" + t["accent"] + " !important;color:" + t["accent"] + " !important;}"

        # ── inputs ──
        ".stTextInput input,.stNumberInput input,.stTextArea textarea{"
        "font-family:'JetBrains Mono',monospace !important;font-size:0.82rem !important;"
        "background:" + t["surface2"] + " !important;"
        "border:1px solid " + t["border"] + " !important;"
        "color:" + t["text"] + " !important;border-radius:8px !important;}"
        ".stTextInput input:focus,.stNumberInput input:focus{"
        "border-color:" + t["accent"] + " !important;outline:none !important;}"

        # ── selectbox / multiselect ──
        ".stSelectbox div[data-baseweb='select']>div,"
        ".stMultiSelect div[data-baseweb='select']>div{"
        "background:" + t["surface2"] + " !important;"
        "border:1px solid " + t["border"] + " !important;"
        "border-radius:8px !important;"
        "font-family:'DM Sans',sans-serif !important;"
        "font-size:0.85rem !important;color:" + t["text"] + " !important;}"
        "[data-baseweb='popover']{"
        "background:" + t["surface"] + " !important;"
        "border:1px solid " + t["border2"] + " !important;border-radius:10px !important;}"
        "[data-baseweb='menu']{background:" + t["surface"] + " !important;}"
        "[data-baseweb='option']{"
        "color:" + t["text"] + " !important;"
        "font-family:'DM Sans',sans-serif !important;"
        "background:" + t["surface"] + " !important;}"
        "[data-baseweb='option']:hover{background:" + t["surface2"] + " !important;}"

        # ── tabs ──
        ".stTabs [data-baseweb='tab-list']{"
        "background:" + t["surface2"] + " !important;"
        "border-radius:12px !important;padding:4px !important;"
        "gap:2px;border:1px solid " + t["border"] + ";flex-wrap:wrap;}"
        ".stTabs [data-baseweb='tab']{"
        "font-family:'DM Sans',sans-serif !important;"
        "font-size:0.82rem !important;font-weight:500 !important;"
        "color:" + t["text3"] + " !important;background:transparent !important;"
        "border-radius:8px !important;padding:0.38rem 0.95rem !important;white-space:nowrap;}"
        ".stTabs [aria-selected='true']{"
        "background:" + t["accent"] + " !important;color:#fff !important;}"
        ".stTabs [data-baseweb='tab-panel']{"
        "background:" + t["surface"] + " !important;"
        "border:1px solid " + t["border"] + ";"
        "border-radius:12px !important;padding:1.5rem !important;margin-top:0.5rem;}"

        # ── alerts ──
        ".stAlert{"
        "border-radius:8px !important;"
        "font-family:'DM Sans',sans-serif !important;"
        "font-size:0.85rem !important;background:" + t["surface"] + " !important;}"

        # ── expander ──
        ".stExpander{"
        "background:" + t["surface"] + " !important;"
        "border:1px solid " + t["border"] + " !important;border-radius:12px !important;}"
        ".stExpander summary p{"
        "font-family:'DM Sans',sans-serif !important;"
        "font-size:0.88rem !important;font-weight:600 !important;"
        "color:" + t["text2"] + " !important;}"

        # ── radio / checkbox ──
        ".stRadio label,.stCheckbox label{"
        "font-family:'DM Sans',sans-serif !important;"
        "font-size:0.85rem !important;color:" + t["text2"] + " !important;}"
        ".stRadio [data-testid='stMarkdownContainer'] p{color:" + t["text2"] + " !important;}"

        # ── file uploader ──
        "[data-testid='stFileUploader']{"
        "background:" + t["surface2"] + " !important;"
        "border:1.5px dashed " + t["border2"] + " !important;"
        "border-radius:12px !important;padding:1rem !important;}"

        # ── caption ──
        ".stCaption p{"
        "font-family:'DM Sans',sans-serif !important;"
        "font-size:0.78rem !important;color:" + t["text3"] + " !important;}"

        # ── custom component classes ──
        ".section-hdr{"
        "display:flex;align-items:center;gap:0.6rem;"
        "margin-bottom:1.1rem;padding-bottom:0.65rem;"
        "border-bottom:1px solid " + t["border"] + ";}"
        ".section-hdr h3{"
        "margin:0 !important;font-size:0.97rem !important;"
        "font-weight:600 !important;color:" + t["text"] + " !important;}"
        ".section-tag{"
        "font-family:'JetBrains Mono',monospace !important;"
        "font-size:0.6rem !important;text-transform:uppercase;"
        "letter-spacing:0.09em;background:transparent;"
        "color:" + t["accent"] + ";border:1px solid " + t["accent"] + ";"
        "padding:0.1rem 0.45rem;border-radius:4px;opacity:0.7;}"
        ".impact-box{"
        "font-family:'JetBrains Mono',monospace;"
        "font-size:0.77rem;padding:0.6rem 1rem;"
        "background:" + t["surface2"] + ";"
        "border:1px solid " + t["border2"] + ";"
        "border-left:3px solid " + t["green"] + ";"
        "border-radius:8px;color:" + t["green"] + ";margin:0.5rem 0;}"
        ".log-item{"
        "font-family:'JetBrains Mono',monospace;"
        "font-size:0.73rem;padding:0.42rem 0.8rem;margin:0.18rem 0;"
        "background:" + t["surface2"] + ";"
        "border-left:2px solid " + t["accent"] + ";"
        "border-radius:0 8px 8px 0;color:" + t["text2"] + ";}"
        ".log-num{color:" + t["accent"] + ";font-weight:600;}"
        ".welcome-card{"
        "background:" + t["surface"] + ";border:1px solid " + t["border"] + ";"
        "border-radius:18px;padding:2.8rem 3rem 2.3rem;"
        "text-align:center;box-shadow:0 6px 32px " + t["shadow"] + ";"
        "max-width:680px;margin:0 auto;}"
        ".chip{"
        "display:inline-block;font-family:'DM Sans',sans-serif;"
        "font-size:0.8rem;font-weight:500;padding:0.28rem 0.85rem;"
        "background:" + t["surface2"] + ";border:1px solid " + t["border2"] + ";"
        "border-radius:20px;color:" + t["text2"] + ";margin:0.2rem;}"
        ".page-label{"
        "font-family:'JetBrains Mono',monospace;"
        "font-size:0.6rem;text-transform:uppercase;"
        "letter-spacing:0.12em;color:" + t["accent"] + ";margin-bottom:0.3rem;}"
        ".page-sub{"
        "font-family:'DM Sans',sans-serif;"
        "font-size:0.88rem;color:" + t["text3"] + ";margin-top:0.25rem;}"
        ".swatch{"
        "display:inline-block;width:26px;height:26px;"
        "border-radius:6px;margin:2px;"
        "border:1px solid rgba(128,128,128,0.2);vertical-align:middle;}"
    )

# Inject the CSS via a <script> that creates a <style> node.
# json.dumps() safely escapes the CSS string so it can be embedded in JS.
_css_str = _make_css(T)
st.markdown(
    "<script>(function(){"
    "var s=document.createElement('style');"
    "s.id='dps-theme';"
    "s.textContent=" + json.dumps(_css_str) + ";"
    "var existing=document.getElementById('dps-theme');"
    "if(existing){existing.remove();}"
    "document.head.appendChild(s);"
    "})();</script>",
    unsafe_allow_html=True,
)

# ── HELPERS ──────────────────────────────────────────────
def log_action(action, details, columns=None):
    st.session_state.log.append({
        "step":      len(st.session_state.log) + 1,
        "action":    action,
        "details":   details,
        "columns":   ", ".join(columns) if columns else "—",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })

def reset_session():
    for k in ["raw_df","current_df","log","file_name"]:
        st.session_state[k] = [] if k=="log" else (None if k!="file_name" else "")
    st.rerun()

def push_df(df, action, details, columns=None):
    st.session_state.current_df = df.copy()
    log_action(action, details, columns)

@st.cache_data(show_spinner=False)
def _load(file_bytes, file_name):
    try:
        if file_name.endswith(".csv"):    return pd.read_csv(io.BytesIO(file_bytes))
        elif file_name.endswith(".xlsx"): return pd.read_excel(io.BytesIO(file_bytes))
        elif file_name.endswith(".json"): return pd.read_json(io.BytesIO(file_bytes))
    except Exception as e:
        st.error(f"Could not parse file: {e}")
    return None

def section(icon, title, tag=None):
    tag_html = f'<span class="section-tag">{tag}</span>' if tag else ""
    st.markdown(
        f'<div class="section-hdr"><span>{icon}</span>'
        f'<h3>{title}</h3>{tag_html}</div>',
        unsafe_allow_html=True,
    )

def impact(msg):
    st.markdown(f'<div class="impact-box">✓ {msg}</div>', unsafe_allow_html=True)

def page_header(label, title, subtitle=""):
    st.markdown(
        f'<div class="page-label">{label}</div>'
        f'<h1 style="font-size:1.85rem;font-weight:700;margin:0 0 0.1rem;">{title}</h1>'
        + (f'<div class="page-sub">{subtitle}</div>' if subtitle else ""),
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

# ── PLOTLY THEME ─────────────────────────────────────────
def apply_theme(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=T["plot_bg"],
        font=dict(family="JetBrains Mono,monospace", color=T["font_color"], size=11),
        title_font=dict(family="DM Sans,sans-serif", color=T["text"], size=14),
        xaxis=dict(gridcolor=T["grid"], linecolor=T["border"],
                   tickfont=dict(size=10, color=T["font_color"])),
        yaxis=dict(gridcolor=T["grid"], linecolor=T["border"],
                   tickfont=dict(size=10, color=T["font_color"])),
        legend=dict(bgcolor=T["surface"], bordercolor=T["border"], borderwidth=1,
                    font=dict(color=T["text2"])),
        margin=dict(t=45, b=40, l=50, r=20),
    )
    return fig

def set_mpl():
    plt.rcParams.update({
        "figure.facecolor": T["surface"],
        "axes.facecolor":   T["plot_bg"],
        "axes.edgecolor":   T["border"],
        "axes.labelcolor":  T["font_color"],
        "text.color":       T["font_color"],
        "xtick.color":      T["font_color"],
        "ytick.color":      T["font_color"],
        "grid.color":       T["grid"],
        "grid.linewidth":   0.6,
        "font.family":      "sans-serif",
    })

# ── SIDEBAR ──────────────────────────────────────────────
data_loaded = st.session_state.current_df is not None

with st.sidebar:
    st.markdown(
        f'<div style="padding:0.6rem 0 0.8rem;border-bottom:1px solid {T["border"]};margin-bottom:0.9rem;">'
        f'<div style="font-family:DM Sans,sans-serif;font-size:1.1rem;font-weight:700;color:{T["text"]};">⬡ DataPrep Studio</div>'
        f'<div style="font-family:JetBrains Mono,monospace;font-size:0.58rem;color:{T["text3"]};'
        f'text-transform:uppercase;letter-spacing:0.1em;margin-top:0.2rem;">Data Preparation &amp; Visualization</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div style="font-family:JetBrains Mono,monospace;font-size:0.6rem;'
        f'text-transform:uppercase;letter-spacing:0.08em;color:{T["text3"]};margin-bottom:0.3rem;">Theme</div>',
        unsafe_allow_html=True,
    )
    new_theme = st.selectbox(
        "Theme", list(THEMES.keys()),
        index=list(THEMES.keys()).index(st.session_state.theme),
        label_visibility="collapsed",
    )
    if new_theme != st.session_state.theme:
        st.session_state.theme = new_theme
        st.rerun()

    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

    st.markdown(
        f'<div style="font-family:JetBrains Mono,monospace;font-size:0.6rem;'
        f'text-transform:uppercase;letter-spacing:0.08em;color:{T["text3"]};margin-bottom:0.3rem;">Navigate</div>',
        unsafe_allow_html=True,
    )
    NAV = ["Upload & Overview","Cleaning Studio","Visualization Builder","Export & Report"]
    page = st.radio("nav", NAV, label_visibility="collapsed")

    if page != "Upload & Overview" and not data_loaded:
        st.warning("Upload a dataset first.")
        page = "Upload & Overview"

    if data_loaded:
        dfs = st.session_state.current_df
        st.markdown(
            f'<div style="margin-top:1rem;padding:0.8rem;background:{T["surface2"]};'
            f'border:1px solid {T["border"]};border-radius:10px;">'
            f'<div style="font-family:JetBrains Mono,monospace;font-size:0.58rem;'
            f'text-transform:uppercase;letter-spacing:0.08em;color:{T["text3"]};margin-bottom:0.4rem;">Active Dataset</div>'
            f'<div style="font-size:0.8rem;color:{T["text2"]};margin-bottom:0.15rem;">📄 {st.session_state.file_name}</div>'
            f'<div style="font-family:JetBrains Mono,monospace;font-size:0.78rem;color:{T["accent"]};">'
            f'{dfs.shape[0]:,} rows × {dfs.shape[1]} cols</div>'
            f'<div style="font-family:JetBrains Mono,monospace;font-size:0.7rem;color:{T["green"]};margin-top:0.2rem;">'
            f'{len(st.session_state.log)} steps applied</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:0.4rem;'></div>", unsafe_allow_html=True)
        st.markdown('<div class="danger-btn">', unsafe_allow_html=True)
        if st.button("↺  Reset Session", key="btn_reset", use_container_width=True):
            reset_session()
        st.markdown("</div>", unsafe_allow_html=True)
        if st.session_state.log:
            if st.button("↩  Undo Last Step", key="btn_undo", use_container_width=True):
                st.session_state.log.pop()
                st.rerun()


# ═══════════════════════════════════════════
# PAGE A — UPLOAD & OVERVIEW
# ═══════════════════════════════════════════
if page == "Upload & Overview":

    if not data_loaded:
        st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)
        _, mid, _ = st.columns([1, 2.5, 1])
        with mid:
            st.markdown(
                f'<div class="welcome-card">'
                f'<div style="font-size:2.5rem;margin-bottom:0.6rem;">⬡</div>'
                f'<h1 style="font-size:1.9rem;font-weight:700;margin:0 0 0.65rem;color:{T["text"]};">'
                f'Welcome to DataPrep Studio</h1>'
                f'<p style="font-size:0.95rem;color:{T["text2"]};line-height:1.7;margin:0 0 0.4rem;">'
                f'Your all-in-one workspace for cleaning, transforming, and visualizing data — '
                f'without writing a single line of code.</p>'
                f'<p style="font-size:0.86rem;color:{T["text3"]};line-height:1.65;margin:0;">'
                f'Profile your dataset, fix missing values, handle outliers, encode categories, '
                f'build interactive charts, and export a reproducible transformation recipe.</p>'
                f'<div style="margin-top:1.3rem;">'
                f'<span class="chip">📥 CSV / Excel / JSON</span>'
                f'<span class="chip">🧹 Smart Cleaning</span>'
                f'<span class="chip">📊 6 Chart Types</span>'
                f'<span class="chip">📤 Export Recipe</span>'
                f'</div>'
                f'<div style="margin-top:1.8rem;font-size:0.9rem;font-weight:600;color:{T["accent"]};">'
                f'↓ &nbsp;Upload your file below to get started</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)
        _, mid2, _ = st.columns([1, 2.5, 1])
        with mid2:
            uf = st.file_uploader("Upload CSV, Excel (.xlsx) or JSON",
                                  type=["csv","xlsx","json"], key="uploader_welcome")
    else:
        page_header("01 / INGEST","Upload & Overview")
        uf = st.file_uploader("Replace dataset", type=["csv","xlsx","json"],
                              label_visibility="collapsed", key="uploader_replace")

    if uf is not None:
        if st.session_state.raw_df is None or uf.name != st.session_state.file_name:
            with st.spinner("Loading…"):
                df_new = _load(uf.read(), uf.name)
            if df_new is not None:
                st.session_state.raw_df     = df_new.copy()
                st.session_state.current_df = df_new.copy()
                st.session_state.file_name  = uf.name
                st.session_state.log        = []
                log_action("Upload", f"Loaded {uf.name}", [])
                st.rerun()

    if data_loaded:
        df = st.session_state.current_df

        m1,m2,m3,m4,m5 = st.columns(5)
        m1.metric("Rows",           f"{df.shape[0]:,}")
        m2.metric("Columns",        f"{df.shape[1]}")
        m3.metric("Missing Values", f"{df.isna().sum().sum():,}")
        pct = round(df.isna().sum().sum() / max(df.shape[0]*df.shape[1],1)*100,1)
        m4.metric("Missing %",      f"{pct}%")
        m5.metric("Duplicate Rows", f"{df.duplicated().sum():,}")

        st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)
        t_prev, t_prof, t_stats = st.tabs(["  Preview  ","  Column Profile  ","  Summary Stats  "])

        with t_prev:
            st.dataframe(df.head(20), use_container_width=True, height=380)

        with t_prof:
            rows = []
            for col in df.columns:
                dt   = str(df[col].dtype)
                miss = int(df[col].isna().sum())
                mp   = round(miss/max(len(df),1)*100,1)
                kind = ("Numeric"     if any(x in dt for x in ("int","float"))
                        else "Datetime"    if "datetime" in dt
                        else "Categorical" if dt in ("object","category")
                        else dt)
                rows.append({"Column":col,"Kind":kind,"Dtype":dt,
                             "Non-Null":int(df[col].notna().sum()),
                             "Missing":miss,"Missing %":mp,
                             "Unique":int(df[col].nunique())})
            prof = pd.DataFrame(rows)
            def _cm(v):
                if v>30: return f"color:{T['red']}"
                elif v>10: return f"color:{T['amber']}"
                return f"color:{T['green']}"
            st.dataframe(prof.style.applymap(_cm, subset=["Missing %"]),
                         use_container_width=True, height=420, hide_index=True)

        with t_stats:
            nd = df.select_dtypes(include=np.number)
            cd = df.select_dtypes(include=["object","category"])
            if not nd.empty:
                st.markdown("**Numeric columns**")
                st.dataframe(nd.describe().T.round(3), use_container_width=True)
            if not cd.empty:
                st.markdown("**Categorical columns**")
                st.dataframe(cd.describe(include="all").T, use_container_width=True)
            if nd.empty and cd.empty:
                st.info("No summary statistics available.")


# ═══════════════════════════════════════════
# PAGE B — CLEANING STUDIO
# ═══════════════════════════════════════════
elif page == "Cleaning Studio":
    page_header("02 / PREPARE","Cleaning Studio","Transform, clean, and reshape your dataset.")
    df = st.session_state.current_df.copy()

    tabs = st.tabs(["Missing Values","Duplicates","Data Types",
                    "Categorical","Outliers","Scaling","Column Ops","Validation"])

    # ── Missing Values ──────────────────────
    with tabs[0]:
        section("◌","Missing Value Handler","NULL TREATMENT")
        na_c    = df.isna().sum()
        na_pct  = (na_c/max(len(df),1)*100).round(1)
        cols_na = na_c[na_c>0].index.tolist()

        if not cols_na:
            st.success("✓ No missing values detected.")
        else:
            fig = px.bar(
                pd.DataFrame({"Column":cols_na,"Pct":na_pct[cols_na].values}),
                x="Column",y="Pct",labels={"Pct":"Missing %"},
                color="Pct",
                color_continuous_scale=[[0,"#34D399"],[0.35,"#FBBF24"],[1,"#F87171"]]
            )
            apply_theme(fig)
            fig.update_layout(height=200,showlegend=False,margin=dict(t=10,b=30,l=30,r=10))
            fig.update_coloraxes(showscale=False)
            st.plotly_chart(fig, use_container_width=True)

            action = st.radio("Action",["Fill Values","Drop Rows","Drop Columns by Threshold"],
                              horizontal=True, key="mv_action")

            if action=="Drop Rows":
                sel = st.multiselect("Drop rows where NULL in:", cols_na,
                                     default=cols_na[:1], key="mv_sel")
                if st.button("Apply — Drop Rows", key="btn_mv_drop_rows") and sel:
                    before = len(df)
                    df = df.dropna(subset=sel)
                    push_df(df,"Drop NaN Rows",f"Removed {before-len(df)} rows",sel)
                    impact(f"Removed {before-len(df)} rows → {len(df):,} remain")
                    st.rerun()

            elif action=="Drop Columns by Threshold":
                thr = st.slider("Drop columns with > X% missing:",0,100,50,key="mv_thr")
                to_drop = [c for c in df.columns if na_pct.get(c,0)>thr]
                if to_drop: st.info(f"Will drop: **{', '.join(to_drop)}**")
                if st.button("Apply — Drop Columns", key="btn_mv_drop_cols") and to_drop:
                    df = df.drop(columns=to_drop)
                    push_df(df,"Drop NaN Cols",f"Dropped: {to_drop}",to_drop)
                    impact(f"Dropped {len(to_drop)} columns"); st.rerun()

            else:
                c1,c2 = st.columns(2)
                fc   = c1.selectbox("Column",cols_na,key="mv_col")
                is_n = any(x in str(df[fc].dtype) for x in ("int","float"))
                mths = (["Mean","Median","Mode","Constant","Forward Fill","Backward Fill"]
                        if is_n else ["Mode","Constant","Forward Fill","Backward Fill"])
                meth = c2.selectbox("Method",mths,key="mv_meth")
                cv   = st.text_input("Constant value",key="mv_const") if meth=="Constant" else ""
                prev = int(df[fc].isna().sum())
                if st.button("Apply Fill", key="btn_mv_fill"):
                    try:
                        if   meth=="Mean":          df[fc]=df[fc].fillna(df[fc].mean())
                        elif meth=="Median":        df[fc]=df[fc].fillna(df[fc].median())
                        elif meth=="Mode":          df[fc]=df[fc].fillna(df[fc].mode()[0])
                        elif meth=="Constant":      df[fc]=df[fc].fillna(cv)
                        elif meth=="Forward Fill":  df[fc]=df[fc].ffill()
                        elif meth=="Backward Fill": df[fc]=df[fc].bfill()
                        push_df(df,"Fill NaN",f"{fc} → {meth}",[fc])
                        impact(f"Filled {prev} nulls in '{fc}' using {meth}"); st.rerun()
                    except Exception as e: st.error(f"Fill error: {e}")

    # ── Duplicates ──────────────────────────
    with tabs[1]:
        section("⧉","Duplicate Detector","DEDUPLICATION")
        subset   = st.multiselect("Check on columns (empty=full row):",df.columns.tolist(),key="dup_sub")
        dsub     = subset or None
        dm       = df.duplicated(subset=dsub,keep=False)
        dn       = df.duplicated(subset=dsub).sum()
        c1,c2    = st.columns(2)
        c1.metric("Duplicate Groups",dn)
        c2.metric("Rows in Groups",int(dm.sum()))
        if dn>0:
            if st.checkbox("Inspect duplicate groups",key="dup_inspect"):
                st.dataframe(df[dm].sort_values(by=(dsub or df.columns[:2].tolist())),
                             use_container_width=True,height=260)
            kp = st.radio("Keep strategy:",["first","last","none (drop all)"],
                          horizontal=True,key="dup_keep")
            ka = False if "none" in kp else kp
            if st.button("Remove Duplicates",key="btn_dup_remove"):
                bef=len(df); df=df.drop_duplicates(subset=dsub,keep=ka)
                push_df(df,"Remove Duplicates",f"Removed {bef-len(df)} rows",subset)
                impact(f"Removed {bef-len(df)} rows → {len(df):,} remain"); st.rerun()
        else:
            st.success("✓ No duplicates detected.")

    # ── Data Types ──────────────────────────
    with tabs[2]:
        section("⌥","Type & Format Parser","TYPE CASTING")
        c1,c2,c3 = st.columns(3)
        col_c    = c1.selectbox("Column",df.columns,key="tc_col")
        cur_t    = str(df[col_c].dtype)
        c2.markdown(
            f'<div style="padding-top:0.3rem;">'
            f'<div style="font-family:JetBrains Mono,monospace;font-size:0.6rem;color:{T["text3"]};'
            f'text-transform:uppercase;letter-spacing:0.06em;">Current type</div>'
            f'<div style="font-family:JetBrains Mono,monospace;font-size:0.9rem;color:{T["amber"]};'
            f'margin-top:0.2rem;">{cur_t}</div></div>',
            unsafe_allow_html=True)
        new_t = c3.selectbox("Target Type",["Numeric","Datetime","Categorical","String"],key="tc_tgt")
        o1,_  = st.columns(2)
        clean_d,dt_fmt = False,""
        if new_t=="Numeric":    clean_d = o1.checkbox("Strip dirty chars (commas, £, $…)",key="tc_clean")
        elif new_t=="Datetime": dt_fmt  = o1.text_input("Date format (optional, e.g. %d/%m/%Y)",key="tc_fmt")
        sample = df[col_c].dropna().head(5).tolist()
        st.markdown(
            f'<div style="font-family:JetBrains Mono,monospace;font-size:0.7rem;color:{T["text3"]};'
            f'margin:0.3rem 0 0.7rem;">Sample: {" · ".join(str(s) for s in sample)}</div>',
            unsafe_allow_html=True)
        if st.button("Convert Type",key="btn_tc"):
            try:
                if new_t=="Numeric":
                    if clean_d:
                        df[col_c]=df[col_c].astype(str).str.replace(r"[^\d.\-]","",regex=True)
                        df[col_c]=df[col_c].replace("",np.nan)
                    df[col_c]=pd.to_numeric(df[col_c],errors="coerce")
                    push_df(df,"Type Cast",f"{col_c} → Numeric",[col_c])
                    impact(f"Converted '{col_c}' to Numeric.")
                elif new_t=="Datetime":
                    kw={"errors":"coerce"}
                    if dt_fmt: kw["format"]=dt_fmt
                    df[col_c]=pd.to_datetime(df[col_c],**kw)
                    push_df(df,"Type Cast",f"{col_c} → Datetime",[col_c])
                    impact(f"Parsed '{col_c}' as Datetime.")
                elif new_t=="Categorical":
                    df[col_c]=df[col_c].astype("category")
                    push_df(df,"Type Cast",f"{col_c} → category",[col_c])
                    impact(f"'{col_c}' is now Categorical.")
                elif new_t=="String":
                    df[col_c]=df[col_c].astype(str)
                    push_df(df,"Type Cast",f"{col_c} → string",[col_c])
                    impact(f"'{col_c}' converted to String.")
                st.rerun()
            except Exception as e: st.error(f"Conversion error: {e}")

    # ── Categorical ─────────────────────────
    with tabs[3]:
        section("⊞","Categorical Toolkit","ENCODING")
        cc = df.select_dtypes(include=["object","category"]).columns.tolist()
        if not cc:
            st.info("No categorical columns found.")
        else:
            cat_col = st.selectbox("Column:",cc,key="cat_col")
            vc  = df[cat_col].value_counts().head(12)
            fig = px.bar(x=vc.index.astype(str),y=vc.values,
                         labels={"x":cat_col,"y":"Count"},
                         color_discrete_sequence=[T["accent"]])
            apply_theme(fig)
            fig.update_layout(height=170,margin=dict(t=10,b=30,l=30,r=10))
            st.plotly_chart(fig, use_container_width=True)

            op = st.radio("Operation:",["Standardize Text","Map / Replace Values",
                                         "Group Rare Categories","One-Hot Encoding"],
                          horizontal=True, key="cat_op")

            if op=="Standardize Text":
                ops = st.multiselect("Apply:",["Trim whitespace","Lowercase","Uppercase","Title Case"],
                                     key="cat_std_ops")
                if st.button("Apply Text Standardization", key="btn_cat_std") and ops:
                    for o in ops:
                        if   o=="Trim whitespace": df[cat_col]=df[cat_col].astype(str).str.strip()
                        elif o=="Lowercase":       df[cat_col]=df[cat_col].astype(str).str.lower()
                        elif o=="Uppercase":       df[cat_col]=df[cat_col].astype(str).str.upper()
                        elif o=="Title Case":      df[cat_col]=df[cat_col].astype(str).str.title()
                    push_df(df,"Text Standardize",f"{cat_col}: {ops}",[cat_col])
                    impact(f"Applied {ops} to '{cat_col}'"); st.rerun()

            elif op=="Map / Replace Values":
                uv = df[cat_col].dropna().unique()[:50]
                ed = st.data_editor(pd.DataFrame({"Original":uv,"New_Value":uv}),
                                    hide_index=True, use_container_width=True, key="cat_map_ed")
                um = st.radio("Unmatched:",["Keep original","Set to 'Other'"],key="cat_map_um")
                if st.button("Apply Mapping", key="btn_cat_map"):
                    md = dict(zip(ed["Original"].astype(str),ed["New_Value"].astype(str)))
                    if um=="Keep original":
                        df[cat_col]=df[cat_col].astype(str).map(md).fillna(df[cat_col].astype(str))
                    else:
                        df[cat_col]=df[cat_col].astype(str).map(md).fillna("Other")
                    push_df(df,"Map Values",f"Remapped {cat_col}",[cat_col])
                    impact(f"Mapping applied to '{cat_col}'"); st.rerun()

            elif op=="Group Rare Categories":
                thr  = st.slider("Group below (%):",1,30,5,key="cat_rare_thr")
                freq = df[cat_col].value_counts(normalize=True)*100
                rare = freq[freq<thr].index.tolist()
                if rare:
                    st.info(f"Will group **{len(rare)}** → 'Other': "
                            f"{', '.join(str(c) for c in rare[:8])}{'…' if len(rare)>8 else ''}")
                if st.button("Group → 'Other'", key="btn_cat_group") and rare:
                    df[cat_col]=df[cat_col].replace(rare,"Other")
                    push_df(df,"Group Rare",f"{cat_col}: {len(rare)} → Other",[cat_col])
                    impact(f"Grouped {len(rare)} rare categories"); st.rerun()

            elif op=="One-Hot Encoding":
                df1  = st.checkbox("Drop first category",key="cat_ohe_drop")
                pref = st.text_input("Prefix:",value=cat_col,key="cat_ohe_pref")
                st.caption(f"Will add {df[cat_col].nunique()} new columns.")
                if st.button("Apply One-Hot Encoding", key="btn_cat_ohe"):
                    df=pd.get_dummies(df,columns=[cat_col],drop_first=df1,prefix=pref)
                    push_df(df,"One-Hot Encode",f"Encoded {cat_col}",[cat_col])
                    impact("One-hot encoding applied"); st.rerun()

    # ── Outliers ────────────────────────────
    with tabs[4]:
        section("◈","Outlier Detection & Treatment","NUMERIC CLEANING")
        nc = df.select_dtypes(include=np.number).columns.tolist()
        if not nc:
            st.info("No numeric columns available.")
        else:
            c1,c2 = st.columns(2)
            oc    = c1.selectbox("Column:",nc,key="out_col")
            meth  = c2.radio("Method:",["IQR (×1.5)","Z-Score (|z|>3)"],
                             horizontal=True, key="out_meth")
            cd    = df[oc].dropna()
            if "IQR" in meth:
                Q1,Q3 = cd.quantile(0.25),cd.quantile(0.75)
                lo,hi = Q1-1.5*(Q3-Q1),Q3+1.5*(Q3-Q1)
            else:
                mu,sig = cd.mean(),cd.std()
                lo,hi  = mu-3*sig,mu+3*sig
            om = (df[oc]<lo)|(df[oc]>hi)
            st.metric("Outliers Detected",
                      f"{int(om.sum())}  ({round(om.sum()/max(len(df),1)*100,2)}%)")
            fig=go.Figure(go.Box(
                y=cd,name=oc,
                marker_color=T["accent"],line_color=T["accent2"],
                boxpoints="outliers",marker_outliercolor=T["red"],marker_size=4,
            ))
            apply_theme(fig)
            fig.update_layout(height=250,showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            oa = st.radio("Action:",["Do Nothing","Remove Outlier Rows","Cap / Winsorize"],
                          horizontal=True, key="out_action")
            if st.button("Apply Outlier Treatment", key="btn_out_apply") and oa!="Do Nothing":
                if oa=="Remove Outlier Rows":
                    bef=len(df); df=df[~om]
                    push_df(df,"Remove Outliers",f"Removed {bef-len(df)} rows",[oc])
                    impact(f"Removed {bef-len(df)} outlier rows")
                else:
                    cl=int((df[oc]<lo).sum()); ch=int((df[oc]>hi).sum())
                    df.loc[df[oc]<lo,oc]=lo; df.loc[df[oc]>hi,oc]=hi
                    push_df(df,"Winsorize",f"Capped {oc}",[oc])
                    impact(f"Capped {cl} low + {ch} high values")
                st.rerun()

    # ── Scaling ─────────────────────────────
    with tabs[5]:
        section("⇅","Normalization & Scaling","FEATURE SCALING")
        nc = df.select_dtypes(include=np.number).columns.tolist()
        if not nc:
            st.info("No numeric columns available.")
        else:
            sc = st.multiselect("Columns to scale:",nc,key="sc_cols")
            sm = st.radio("Method:",["Min-Max (0–1)","Z-Score (mean=0, std=1)"],
                          horizontal=True, key="sc_meth")
            if sc:
                st.markdown("**Before:**")
                st.dataframe(df[sc].describe().T[["mean","std","min","max"]].round(3),
                             use_container_width=True,height=150)
            if st.button("Apply Scaling", key="btn_sc_apply") and sc:
                for c in sc:
                    if "Min" in sm:
                        mn,mx=df[c].min(),df[c].max()
                        if mx!=mn: df[c]=(df[c]-mn)/(mx-mn)
                        else: st.warning(f"Skipped '{c}' — constant.")
                    else:
                        mu,sig=df[c].mean(),df[c].std()
                        if sig: df[c]=(df[c]-mu)/sig
                        else: st.warning(f"Skipped '{c}' — zero std.")
                push_df(df,"Scale",f"{sm} on {sc}",sc)
                st.markdown("**After:**")
                st.dataframe(df[sc].describe().T[["mean","std","min","max"]].round(3),
                             use_container_width=True,height=150)
                impact(f"Scaled {len(sc)} column(s)"); st.rerun()

    # ── Column Ops ──────────────────────────
    with tabs[6]:
        section("⊕","Column Operations","RESHAPE")
        cop = st.radio("Operation:",["Rename","Drop","Formula","Binning"],
                       horizontal=True, key="cop_op")

        if cop=="Rename":
            c1,c2 = st.columns(2)
            rc = c1.selectbox("Column to rename:",df.columns,key="cop_ren_col")
            nn = c2.text_input("New name:",key="cop_ren_name")
            if st.button("Rename Column", key="btn_cop_rename") and nn and nn!=rc:
                df=df.rename(columns={rc:nn})
                push_df(df,"Rename",f"'{rc}' → '{nn}'",[rc])
                impact(f"Renamed '{rc}' → '{nn}'"); st.rerun()

        elif cop=="Drop":
            dc = st.multiselect("Columns to drop:",df.columns,key="cop_drop_cols")
            if dc: st.caption(f"Will remove: {', '.join(dc)}")
            if st.button("Drop Columns", key="btn_cop_drop") and dc:
                df=df.drop(columns=dc)
                push_df(df,"Drop Columns",f"Dropped: {dc}",dc)
                impact(f"Dropped {len(dc)} column(s)"); st.rerun()

        elif cop=="Formula":
            nf = df.select_dtypes(include=np.number).columns.tolist()
            if not nf: st.info("No numeric columns.")
            else:
                c1,c2,c3 = st.columns(3)
                ca = c1.selectbox("Column A:",nf,key="cop_fml_a")
                op = c2.selectbox("Operator:",["+","−","×","÷"],key="cop_fml_op")
                cb = c3.selectbox("Column B:",nf,key="cop_fml_b")
                nn = st.text_input("New column name:",key="cop_fml_name")
                if st.button("Create Column", key="btn_cop_formula") and nn:
                    try:
                        if op=="+":  df[nn]=df[ca]+df[cb]
                        elif op=="−":df[nn]=df[ca]-df[cb]
                        elif op=="×":df[nn]=df[ca]*df[cb]
                        elif op=="÷":df[nn]=df[ca]/df[cb].replace(0,np.nan)
                        push_df(df,"Formula",f"{nn}={ca}{op}{cb}",[nn])
                        impact(f"Created '{nn}'"); st.rerun()
                    except Exception as e: st.error(f"Formula error: {e}")

        elif cop=="Binning":
            nb = df.select_dtypes(include=np.number).columns.tolist()
            if not nb: st.info("No numeric columns.")
            else:
                c1,c2,c3 = st.columns(3)
                bc    = c1.selectbox("Column to bin:",nb,key="cop_bin_col")
                bins  = c2.number_input("Bins:",2,20,4,key="cop_bin_n")
                strat = c3.radio("Strategy:",["Equal-width","Quantile"],key="cop_bin_strat")
                on    = st.text_input("Output name:",value=f"{bc}_bin",key="cop_bin_out")
                if st.button("Apply Binning", key="btn_cop_bin") and on:
                    try:
                        if strat=="Equal-width":
                            df[on]=pd.cut(df[bc],bins=int(bins),labels=False)
                        else:
                            df[on]=pd.qcut(df[bc],q=int(bins),labels=False,duplicates="drop")
                        push_df(df,"Binning",f"{bc}→{bins} bins→{on}",[bc])
                        impact(f"Created '{on}'"); st.rerun()
                    except Exception as e: st.error(f"Binning error: {e}")

    # ── Validation ──────────────────────────
    with tabs[7]:
        section("✓","Data Validation Rules","QUALITY CHECKS")
        vc_  = st.selectbox("Column to validate:",df.columns,key="val_col")
        rule = st.radio("Rule:",["Non-Null Constraint","Numeric Range","Allowed Categories List"],
                        horizontal=True, key="val_rule")
        vmin_,vmax_,astr_ = 0.0,100.0,""
        if rule=="Numeric Range":
            r1,r2 = st.columns(2)
            vmin_ = r1.number_input("Min:",value=0.0,key="val_min")
            vmax_ = r2.number_input("Max:",value=100.0,key="val_max")
        elif rule=="Allowed Categories List":
            astr_ = st.text_input("Allowed values (comma-separated):",key="val_cats")
        viols = None
        if st.button("Run Validation Check", key="btn_val_run"):
            if rule=="Non-Null Constraint":
                viols=df[df[vc_].isna()]
            elif rule=="Numeric Range":
                nc2=pd.to_numeric(df[vc_],errors="coerce")
                viols=df[(nc2<vmin_)|(nc2>vmax_)]
            elif rule=="Allowed Categories List" and astr_.strip():
                al=[x.strip() for x in astr_.split(",")]
                viols=df[~df[vc_].astype(str).isin(al)]
            else:
                st.warning("Fill in rule parameters first.")
        if viols is not None:
            if len(viols)==0:
                st.success("✓ No violations.")
            else:
                st.error(f"⚠ {len(viols)} violations ({round(len(viols)/max(len(df),1)*100,1)}%)")
                st.dataframe(viols.head(200),use_container_width=True,height=280)
                st.download_button("Export Violations (CSV)",
                                   viols.to_csv(index=False).encode(),
                                   "violations.csv","text/csv",key="btn_val_export")


# ═══════════════════════════════════════════
# PAGE C — VISUALIZATION BUILDER
# ═══════════════════════════════════════════
elif page == "Visualization Builder":
    page_header("03 / EXPLORE","Visualization Builder",
                "Build interactive charts from your prepared dataset.")

    df_orig  = st.session_state.current_df.copy()
    cat_cols = df_orig.select_dtypes(include=["object","category"]).columns.tolist()
    num_cols = df_orig.select_dtypes(include=np.number).columns.tolist()
    all_cols = df_orig.columns.tolist()

    # ── Filters ──────────────────────────
    with st.expander("⚙  Filters & Data Scope", expanded=False):
        fc1,fc2,fc3 = st.columns(3)
        df_view = df_orig.copy()
        with fc1:
            if cat_cols:
                cf = st.selectbox("Filter by category:",["— none —"]+cat_cols,key="vcf")
                if cf!="— none —":
                    cv = st.multiselect("Keep values:",df_view[cf].dropna().unique().tolist(),key="vcv")
                    if cv: df_view=df_view[df_view[cf].isin(cv)]
        with fc2:
            if num_cols:
                nf = st.selectbox("Filter by range:",["— none —"]+num_cols,key="vnf")
                if nf!="— none —":
                    mnv=float(df_view[nf].min()); mxv=float(df_view[nf].max())
                    if mnv<mxv:
                        lo2,hi2=st.slider("Range:",mnv,mxv,(mnv,mxv),key="vrng")
                        df_view=df_view[(df_view[nf]>=lo2)&(df_view[nf]<=hi2)]
        with fc3:
            st.markdown(
                f'<div style="font-family:JetBrains Mono,monospace;font-size:0.75rem;padding-top:0.3rem;">'
                f'<div style="color:{T["text3"]};font-size:0.62rem;text-transform:uppercase;'
                f'letter-spacing:0.06em;">Rows visible</div>'
                f'<div style="color:{T["accent"]};font-size:1.05rem;margin-top:0.2rem;">{len(df_view):,}</div>'
                f'<div style="color:{T["text3"]};font-size:0.7rem;">{df_orig.shape[0]-len(df_view):,} filtered out</div>'
                f'</div>',
                unsafe_allow_html=True)

    # ── Color Palette ─────────────────────
    with st.expander("🎨  Color Palette", expanded=False):
        st.markdown(
            f'<div style="font-size:0.85rem;color:{T["text2"]};margin-bottom:0.6rem;">'
            f'Set how many colors to use, then pick each one. These are applied to every chart.</div>',
            unsafe_allow_html=True)

        n_input = st.number_input(
            "Number of colors:", min_value=1, max_value=10,
            value=st.session_state.num_colors, step=1, key="nc_input",
        )
        n_int = int(n_input)
        if n_int != st.session_state.num_colors:
            st.session_state.num_colors = n_int
            while len(st.session_state.chart_colors) < n_int:
                st.session_state.chart_colors.append("#AAAAAA")
            st.session_state.chart_colors = st.session_state.chart_colors[:n_int]
            st.rerun()

        cols_per_row = 5
        n_rows = (n_int + cols_per_row - 1) // cols_per_row
        for row_i in range(n_rows):
            pcols = st.columns(cols_per_row)
            for col_i in range(cols_per_row):
                idx = row_i * cols_per_row + col_i
                if idx >= n_int:
                    break
                with pcols[col_i]:
                    picked = st.color_picker(
                        f"Color {idx+1}",
                        value=st.session_state.chart_colors[idx],
                        key=f"cp_{idx}",
                    )
                    st.session_state.chart_colors[idx] = picked

        swatches = "".join(
            f'<span class="swatch" title="{c}" style="background:{c};"></span>'
            for c in st.session_state.chart_colors
        )
        st.markdown(f'<div style="margin-top:0.4rem;">{swatches}</div>',
                    unsafe_allow_html=True)

    PALETTE = list(st.session_state.chart_colors)

    # ── Chart config ──────────────────────
    left_p, right_p = st.columns([1, 3], gap="large")

    with left_p:
        st.markdown(
            f'<div style="font-family:JetBrains Mono,monospace;font-size:0.6rem;'
            f'text-transform:uppercase;letter-spacing:0.08em;color:{T["text3"]};margin-bottom:0.5rem;">Chart Type</div>',
            unsafe_allow_html=True)
        chart_type = st.selectbox("ctype",[
            "Histogram","Box Plot","Scatter Plot",
            "Line Chart","Bar Chart","Correlation Heatmap"
        ], label_visibility="collapsed")

        st.markdown("<div style='height:0.4rem;'></div>", unsafe_allow_html=True)
        x_col     = st.selectbox("X axis:",all_cols)
        y_col     = st.selectbox("Y axis:",["— none —"]+all_cols)
        group_col = st.selectbox("Color / Group:",["— none —"]+cat_cols)

        agg_m,top_n = "sum",0
        if chart_type=="Bar Chart":
            agg_m  = st.selectbox("Aggregation:",["sum","mean","count","median"])
            top_n  = int(st.number_input("Top N (0=all):",0,200,0))

    with right_p:
        gc = None if group_col=="— none —" else group_col
        yc = None if y_col=="— none —" else y_col
        try:
            if chart_type=="Histogram":
                fig=px.histogram(df_view,x=x_col,color=gc,
                                 barmode="overlay",opacity=0.8,
                                 color_discrete_sequence=PALETTE)
                apply_theme(fig); st.plotly_chart(fig,use_container_width=True)

            elif chart_type=="Box Plot":
                fig=px.box(df_view,x=gc,y=yc if yc else x_col,color=gc,
                           color_discrete_sequence=PALETTE)
                apply_theme(fig); st.plotly_chart(fig,use_container_width=True)

            elif chart_type=="Scatter Plot":
                if not yc: st.error("Select a Y axis column.")
                else:
                    fig=px.scatter(df_view,x=x_col,y=yc,color=gc,
                                   opacity=0.75,color_discrete_sequence=PALETTE)
                    apply_theme(fig); st.plotly_chart(fig,use_container_width=True)

            elif chart_type=="Line Chart":
                if not yc: st.error("Select a Y axis column.")
                else:
                    fig=px.line(df_view.sort_values(by=x_col),x=x_col,y=yc,
                                color=gc,color_discrete_sequence=PALETTE)
                    apply_theme(fig); st.plotly_chart(fig,use_container_width=True)

            elif chart_type=="Bar Chart":
                if not yc: st.error("Select a Y axis column.")
                else:
                    if gc:
                        agg_df=df_view.groupby([x_col,gc])[yc].agg(agg_m).reset_index()
                        if top_n>0:
                            top_cats=agg_df.groupby(x_col)[yc].sum().nlargest(top_n).index
                            agg_df=agg_df[agg_df[x_col].isin(top_cats)]
                        fig=px.bar(agg_df,x=x_col,y=yc,color=gc,
                                   barmode="group",color_discrete_sequence=PALETTE)
                    else:
                        agg_df=df_view.groupby(x_col)[yc].agg(agg_m).reset_index()
                        if top_n>0: agg_df=agg_df.nlargest(top_n,yc)
                        fig=px.bar(agg_df,x=x_col,y=yc,
                                   color=x_col,color_discrete_sequence=PALETTE)
                    apply_theme(fig); st.plotly_chart(fig,use_container_width=True)

            elif chart_type=="Correlation Heatmap":
                num_only=df_view.select_dtypes(include=np.number)
                if len(num_only.columns)<2:
                    st.error("Need at least 2 numeric columns.")
                else:
                    set_mpl()
                    corr=num_only.corr()
                    fig_m,ax=plt.subplots(
                        figsize=(max(6,len(corr)*0.85),max(5,len(corr)*0.7)))
                    sns.heatmap(corr,annot=True,fmt=".2f",ax=ax,
                                cmap=sns.diverging_palette(220,10,as_cmap=True),
                                linewidths=0.5,linecolor=T["border"],
                                annot_kws={"size":8},cbar_kws={"shrink":0.8})
                    ax.set_title("Correlation Matrix",pad=12)
                    plt.tight_layout()
                    st.pyplot(fig_m); plt.close(fig_m)

        except Exception as e:
            st.error(f"Chart error: {e}. Check your column selections.")


# ═══════════════════════════════════════════
# PAGE D — EXPORT & REPORT
# ═══════════════════════════════════════════
elif page == "Export & Report":
    page_header("04 / EXPORT","Export & Report",
                "Download your cleaned dataset and transformation recipe.")
    df = st.session_state.current_df

    section("⟳","Transformation Log",f"{len(st.session_state.log)} STEPS")
    if not st.session_state.log:
        st.info("No transformations recorded yet.")
    else:
        for s in st.session_state.log:
            st.markdown(
                f'<div class="log-item">'
                f'<span class="log-num">#{s["step"]:02d}</span>  {s["action"]} — {s["details"]}'
                f'<span style="float:right;color:{T["text3"]};">{s["timestamp"]}</span>'
                f'</div>',
                unsafe_allow_html=True)
        st.markdown("<div style='height:0.7rem;'></div>", unsafe_allow_html=True)
        recipe = {
            "generated":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source_file": st.session_state.file_name,
            "steps":       st.session_state.log,
        }
        snippet = "\n".join([
            "import pandas as pd","import numpy as np","",
            f"# Auto-generated by DataPrep Studio — {datetime.now().strftime('%Y-%m-%d')}",
            f"df = pd.read_csv('{st.session_state.file_name}')","",
        ] + [f"# Step {s['step']}: {s['action']} — {s['details']}" for s in st.session_state.log])

        dl1,dl2 = st.columns(2)
        with dl1:
            st.download_button("⬇  Download JSON Recipe",
                               json.dumps(recipe,indent=2),
                               "transformation_recipe.json","application/json",
                               use_container_width=True,key="btn_dl_recipe")
        with dl2:
            st.download_button("⬇  Download Python Snippet",
                               snippet.encode(),"pipeline.py","text/x-python",
                               use_container_width=True,key="btn_dl_snippet")
        with st.expander("View Python Snippet"):
            st.code(snippet,language="python")

    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
    section("⬇","Export Dataset","FINAL OUTPUT")

    if st.session_state.raw_df is not None:
        raw=st.session_state.raw_df
        d1,d2,d3,d4=st.columns(4)
        d1.metric("Original Rows",f"{raw.shape[0]:,}")
        d2.metric("Final Rows",f"{df.shape[0]:,}",delta=f"{df.shape[0]-raw.shape[0]:+,}")
        d3.metric("Original Cols",f"{raw.shape[1]}")
        d4.metric("Final Cols",f"{df.shape[1]}",delta=f"{df.shape[1]-raw.shape[1]:+,}")

    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
    st.dataframe(df.head(10),use_container_width=True,height=220)

    c1,c2=st.columns(2)
    with c1:
        st.download_button("⬇  Export as CSV",
                           df.to_csv(index=False).encode(),
                           "cleaned_dataset.csv","text/csv",
                           use_container_width=True,key="btn_dl_csv")
    with c2:
        buf=io.BytesIO()
        with pd.ExcelWriter(buf,engine="openpyxl") as w:
            df.to_excel(w,index=False,sheet_name="Cleaned_Data")
        st.download_button("⬇  Export as Excel (.xlsx)",
                           buf.getvalue(),"cleaned_dataset.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           use_container_width=True,key="btn_dl_xlsx")