import html
import os
import streamlit as st
import requests
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

API_URL = os.environ.get("API_URL", "https://smart-homes-system.onrender.com")

st.set_page_config(
    page_title="SmartHome Energy · Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="⚡",
)

# ── Global CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Reset & base ── */
html, body, [class*="css"] {
    font-family: 'Inter', system-ui, sans-serif !important;
}

/* ── Dark page background ── */
[data-testid="stAppViewContainer"],
section.main,
[data-testid="stMain"] {
    background: #0a0e1a !important;
    background-color: #0a0e1a !important;
}

.stApp { background: transparent !important; }

[data-testid="stHeader"] {
    background: rgba(10,14,26,0.85) !important;
    backdrop-filter: blur(16px);
    border-bottom: 1px solid rgba(255,255,255,0.06);
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 4rem;
    max-width: 1280px;
}

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #0f1729 0%, #111827 50%, #0a1628 100%);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 20px;
    padding: 2.5rem 2rem 2rem 2rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 40px rgba(99,102,241,0.12), 0 2px 8px rgba(0,0,0,0.4);
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 260px; height: 260px;
    background: radial-gradient(circle, rgba(99,102,241,0.18) 0%, transparent 70%);
    pointer-events: none;
}
.hero-badge {
    display: inline-block;
    font-size: 0.6875rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #a5b4fc;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.35);
    padding: 0.3rem 0.75rem;
    border-radius: 100px;
    margin-bottom: 1rem;
}
.hero h1 {
    font-size: clamp(1.75rem, 4vw, 2.4rem);
    font-weight: 800;
    letter-spacing: -0.03em;
    color: #f1f5f9;
    margin: 0 0 0.6rem 0;
    line-height: 1.15;
}
.hero h1 span { color: #818cf8; }
.hero p {
    font-size: 1rem;
    color: #94a3b8;
    margin: 0;
    max-width: 44rem;
    line-height: 1.6;
}
.hero-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #22c55e;
    box-shadow: 0 0 8px #22c55e;
    margin-right: 6px;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%,100% { opacity:1; transform:scale(1); }
    50% { opacity:0.6; transform:scale(1.3); }
}

/* ── Section labels ── */
.section-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #6366f1;
    margin: 2rem 0 0.75rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(to right, rgba(99,102,241,0.3), transparent);
}

/* ── KPI grid ── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-top: 1rem;
}
@media (max-width: 900px) { .kpi-grid { grid-template-columns: 1fr; } }

.kpi-card {
    background: linear-gradient(135deg, #111827, #0f172a);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.5rem 1.5rem 1.35rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    transition: transform 0.2s, box-shadow 0.2s;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #6366f1, #8b5cf6, #06b6d4);
}
.kpi-card:hover { transform: translateY(-3px); box-shadow: 0 8px 30px rgba(0,0,0,0.4); }
.kpi-card h4 {
    font-size: 0.75rem;
    font-weight: 600;
    color: #64748b;
    margin: 0 0 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}
.kpi-card .value-green { color: #34d399; font-size: 2rem; font-weight: 700; margin: 0; letter-spacing: -0.03em; }
.kpi-card .value-yellow { color: #fbbf24; font-size: 1.5rem; font-weight: 700; margin: 0; }
.kpi-card .value-red { color: #f87171; font-size: 1.5rem; font-weight: 700; margin: 0; }
.kpi-card .body-text { color: #94a3b8; font-size: 0.9rem; line-height: 1.6; margin: 0; white-space: pre-wrap; }

/* ── AI card ── */
.ai-card {
    background: linear-gradient(135deg, #0f172a, #111827);
    border: 1px solid rgba(139,92,246,0.25);
    border-radius: 16px;
    padding: 1.75rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 24px rgba(139,92,246,0.1);
    margin-top: 1rem;
}
.ai-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #8b5cf6, #6366f1, #06b6d4);
}
.ai-card h3 { margin: 0 0 0.9rem; font-size: 0.95rem; font-weight: 700; color: #c4b5fd; letter-spacing: -0.01em; }
.ai-card p { margin: 0; color: #cbd5e1; line-height: 1.7; font-size: 0.95rem; white-space: pre-wrap; }

/* ── Buttons ── */
div.stButton > button {
    width: 100%;
    height: 3rem;
    font-size: 0.9rem;
    font-weight: 700 !important;
    letter-spacing: 0.04em;
    border-radius: 12px;
    border: 1px solid rgba(99,102,241,0.5) !important;
    background: linear-gradient(135deg, #4f46e5 0%, #6366f1 50%, #7c3aed 100%) !important;
    color: #ffffff !important;
    box-shadow: 0 4px 15px rgba(99,102,241,0.35), inset 0 1px 0 rgba(255,255,255,0.15);
    transition: all 0.22s ease;
}
div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(99,102,241,0.5), inset 0 1px 0 rgba(255,255,255,0.2) !important;
    background: linear-gradient(135deg, #4338ca 0%, #4f46e5 50%, #6d28d9 100%) !important;
}
div.stButton > button:active { transform: translateY(0); }
div.stButton > button p,
div.stButton > button span { color: #ffffff !important; font-weight: 700 !important; }

/* ── Inputs / selects ── */
[data-baseweb="select"] > div {
    border-radius: 12px !important;
    border-color: rgba(255,255,255,0.12) !important;
    background-color: #111827 !important;
    color: #f1f5f9 !important;
}
[data-testid="stNumberInput"] input {
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    background: #111827 !important;
    color: #f1f5f9 !important;
}
[data-testid="stNumberInput"] button {
    border-color: rgba(255,255,255,0.1) !important;
    background: #1e293b !important;
    color: #94a3b8 !important;
}
[data-testid="stNumberInput"] label,
[data-testid="stSelectbox"] label,
[data-testid="stFileUploader"] label,
label[data-testid="stWidgetLabel"] p { color: #94a3b8 !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] section {
    padding: 1.5rem !important;
    background: #111827 !important;
    border: 1px dashed rgba(99,102,241,0.4) !important;
    border-radius: 14px !important;
}
[data-testid="stFileUploader"] section:hover {
    border-color: rgba(99,102,241,0.7) !important;
    background: rgba(99,102,241,0.05) !important;
}
[data-testid="stFileUploader"] [data-baseweb="button"],
[data-testid="stFileUploader"] button {
    color: #a5b4fc !important;
    background-color: rgba(99,102,241,0.1) !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
}
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] [data-testid="stMarkdownContainer"] p { color: #94a3b8 !important; }

/* ── Alerts ── */
div[data-testid="stNotification"], .stAlert { border-radius: 12px !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; }
[data-testid="stDataFrame"] thead th { background: #1e293b !important; color: #a5b4fc !important; }

/* ── Spinner ── */
[data-testid="stSpinner"] > div { border-top-color: #6366f1 !important; }

/* ── Caption / info text ── */
[data-testid="stCaptionContainer"] p { color: #64748b !important; }
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">⚡ Smart Homes Energy System</div>
    <h1>Energy <span>Forecast</span> Dashboard</h1>
    <p>
        <span class="hero-dot"></span>
        Real-time hourly load forecasting, AI-powered optimization, and smart appliance guidance
        for modern grid operations teams.
    </p>
</div>
""", unsafe_allow_html=True)

# ── SESSION STATE ─────────────────────────────────────────────────────────
if "energy_values" not in st.session_state:
    st.session_state.energy_values = [14000.0] * 24

for i in range(24):
    if f"input_{i}" not in st.session_state:
        st.session_state[f"input_{i}"] = st.session_state.energy_values[i]

# ── DATA INPUT ────────────────────────────────────────────────────────────
st.markdown('<p class="section-label">📂 Data Input</p>', unsafe_allow_html=True)

colA, colB = st.columns([1, 2])
with colA:
    if st.button("⚡ Load Sample Series", use_container_width=True):
        new_vals = list(np.linspace(14000, 15500, 24))
        st.session_state.energy_values = new_vals
        for i in range(24):
            st.session_state[f"input_{i}"] = new_vals[i]
        st.rerun()

with colB:
    file = st.file_uploader("Upload Hourly CSV", type=["csv"], label_visibility="visible")

if file:
    df_upload = pd.read_csv(file)
    st.success("✅ CSV loaded successfully.")
    numeric_cols = df_upload.select_dtypes(include=["number"]).columns
    if len(numeric_cols) == 0:
        st.error("⚠️ No numeric columns found in this file.")
    else:
        selected_column = st.selectbox("Select energy column", numeric_cols)
        new_values = df_upload[selected_column].dropna().tolist()[:24]
        if len(new_values) < 24:
            st.warning("⚠️ Please provide at least 24 numeric values.")
        else:
            st.session_state.energy_values = new_values
            for i in range(24):
                st.session_state[f"input_{i}"] = new_values[i]

# ── INPUT GRID ────────────────────────────────────────────────────────────
st.markdown('<p class="section-label">🕐 Hourly Energy Input (MW) — 24 Hours</p>', unsafe_allow_html=True)

values = []
cols = st.columns(6)
for i in range(24):
    with cols[i % 6]:
        val = st.number_input(
            f"H{i+1:02d}",
            value=float(st.session_state[f"input_{i}"]),
            key=f"input_{i}",
        )
        values.append(val)

st.session_state.energy_values = values

# ── CHART ─────────────────────────────────────────────────────────────────
st.markdown('<p class="section-label">📈 Load Profile</p>', unsafe_allow_html=True)

mpl.rcParams["font.family"] = ["DejaVu Sans", "sans-serif"]

fig, ax = plt.subplots(figsize=(12, 4), facecolor="#0f172a")
ax.set_facecolor("#111827")

hours = list(range(1, 25))
ax.fill_between(hours, values, alpha=0.18, color="#6366f1", zorder=1)
ax.fill_between(hours, values, alpha=0.06, color="#06b6d4", zorder=0)
ax.plot(hours, values, color="#818cf8", linewidth=2.5,
        marker="o", markersize=5, markerfacecolor="#1e293b",
        markeredgecolor="#6366f1", markeredgewidth=2, zorder=3)

ax.set_xlabel("Hour", color="#94a3b8", fontsize=10)
ax.set_ylabel("Energy (MW)", color="#94a3b8", fontsize=10)
ax.set_xticks(range(1, 25, 2))
ax.tick_params(axis="both", colors="#64748b", labelsize=9)
ax.grid(True, axis="y", linestyle="--", alpha=0.3, color="#334155", linewidth=0.8)
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_color("#1e293b")
    spine.set_linewidth(1)

avg_val = sum(values) / len(values)
ax.axhline(avg_val, color="#f59e0b", linewidth=1.2, linestyle="--", alpha=0.6, zorder=2)
ax.text(24.3, avg_val, f"avg\n{avg_val:.0f}", color="#f59e0b", fontsize=7.5, va="center")

fig.tight_layout(pad=1.5)
st.pyplot(fig, use_container_width=True)
plt.close(fig)

# ── APPLIANCE INPUTS ──────────────────────────────────────────────────────
st.markdown('<p class="section-label">🏠 Appliance Usage (kWh)</p>', unsafe_allow_html=True)
st.caption("Typical period totals for your home — used for AI optimization analysis.")

ap1, ap2 = st.columns(2)
with ap1:
    appliances = {
        "AC":              st.number_input("🌬️ AC (kWh)",              min_value=0.0, value=2.0,  step=0.1, format="%.2f", key="appl_ac"),
        "Refrigerator":    st.number_input("🧊 Refrigerator (kWh)",    min_value=0.0, value=1.0,  step=0.1, format="%.2f", key="appl_fridge"),
        "Washing Machine": st.number_input("🫧 Washing Machine (kWh)", min_value=0.0, value=0.5,  step=0.1, format="%.2f", key="appl_wm"),
        "Lights":          st.number_input("💡 Lights (kWh)",          min_value=0.0, value=0.8,  step=0.1, format="%.2f", key="appl_lights"),
    }
with ap2:
    appliances.update({
        "Fans":   st.number_input("🌀 Fans (kWh)",   min_value=0.0, value=0.6, step=0.1, format="%.2f", key="appl_fans"),
        "TV":     st.number_input("📺 TV (kWh)",     min_value=0.0, value=0.4, step=0.1, format="%.2f", key="appl_tv"),
        "Laptop": st.number_input("💻 Laptop (kWh)", min_value=0.0, value=0.3, step=0.1, format="%.2f", key="appl_laptop"),
        "Others": st.number_input("🔌 Others (kWh)", min_value=0.0, value=0.5, step=0.1, format="%.2f", key="appl_others"),
    })

# ── ACTIONS ───────────────────────────────────────────────────────────────
st.markdown('<p class="section-label">🔬 Analysis</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
prediction = status = suggestion = ai_suggestion = None

with col1:
    if st.button("🔮 Generate Forecast", use_container_width=True):
        with st.spinner("Running forecast model…"):
            try:
                res = requests.post(f"{API_URL}/predict", json={"input": values}, timeout=120)
                if res.status_code == 200:
                    data = res.json()
                    if "error" in data:
                        st.error(f"Backend error: {data['error']}")
                    else:
                        prediction = data.get("prediction_MW")
                        st.success(f"✅ Forecast complete: **{prediction:.2f} MW**")
                else:
                    st.error(f"Error {res.status_code}: {res.text}")
            except Exception as e:
                st.error(f"Connection failed: {e}")

with col2:
    if st.button("⚙️ Run Optimization", use_container_width=True):
        with st.spinner("Running AI optimization…"):
            try:
                res = requests.post(f"{API_URL}/optimize", json={"input": values, "appliances": appliances}, timeout=120)
                if res.status_code == 200:
                    data = res.json()
                    if "error" in data:
                        st.error(f"Backend error: {data['error']}")
                    else:
                        prediction    = data.get("prediction_MW")
                        status        = data.get("status")
                        suggestion    = data.get("suggestion")
                        ai_suggestion = data.get("ai_suggestion", "")
                        st.success(f"✅ Optimization complete — **{status}** at {prediction:.2f} MW")
                else:
                    st.error(f"Error {res.status_code}: {res.text}")
            except Exception as e:
                st.error(f"Connection failed: {e}")

# ── KPI CARDS ─────────────────────────────────────────────────────────────
if prediction is not None:
    def _status_class(s):
        if not s:
            return "value-yellow"
        sl = s.lower()
        if "low" in sl:
            return "value-green"
        if "moderate" in sl:
            return "value-yellow"
        return "value-red"

    st.markdown('<p class="section-label">📊 Results</p>', unsafe_allow_html=True)

    c1 = f'<div class="kpi-card"><h4>Forecast Output</h4><p class="value-green">{prediction:.2f} MW</p></div>'
    c2 = ""
    if status:
        sc = _status_class(status)
        c2 = f'<div class="kpi-card"><h4>Grid Utilization</h4><p class="{sc}">{html.escape(status)}</p></div>'
    c3 = ""
    if suggestion:
        esc = html.escape(str(suggestion))
        c3 = f'<div class="kpi-card"><h4>Recommendation</h4><p class="body-text">{esc}</p></div>'

    st.markdown(f'<div class="kpi-grid">{c1}{c2}{c3}</div>', unsafe_allow_html=True)

if ai_suggestion and str(ai_suggestion).strip():
    st.markdown('<p class="section-label">🤖 AI Smart Suggestions</p>', unsafe_allow_html=True)
    body_esc = html.escape(str(ai_suggestion)).replace("\n", "<br/>")
    st.markdown(f"""
<div class="ai-card">
    <h3>🧠 AI Energy Analysis</h3>
    <p>{body_esc}</p>
</div>
""", unsafe_allow_html=True)

# ── HISTORY ───────────────────────────────────────────────────────────────
st.markdown('<p class="section-label">🗂️ Forecast History</p>', unsafe_allow_html=True)

if st.button("🔄 Refresh History", use_container_width=True):
    with st.spinner("Fetching history…"):
        try:
            res = requests.get(f"{API_URL}/history", timeout=120)
            if res.status_code == 200:
                data = res.json()
                if isinstance(data, dict) and "error" in data:
                    st.error(f"Backend error: {data['error']}")
                elif len(data) == 0:
                    st.info("📭 No saved forecasts yet. Run a forecast to get started!")
                else:
                    st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
            else:
                st.error(f"Error {res.status_code}: {res.text}")
        except Exception as e:
            st.error(f"Connection failed: {e}")

# ── Footer ────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#334155; font-size:0.78rem; padding: 1rem 0;">
    SmartHome Energy Dashboard &nbsp;·&nbsp; Powered by LSTM &amp; AI Optimization
</div>
""", unsafe_allow_html=True)
