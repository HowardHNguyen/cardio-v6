import numpy as np
import pandas as pd
import json
import joblib
import streamlit as st
import datetime
import os
from pathlib import Path

st.markdown(
    """
    <style>
    /* Hide toolbar, footer, menu — but NOT the header itself,
       because the sidebar toggle arrow lives inside the header. */
    [data-testid="stToolbar"]       { visibility: hidden; height: 0%; position: fixed; }
    footer                          { visibility: hidden; height: 0%; }
    #MainMenu                       { visibility: hidden; }

    /* Hide the deploy button and other header chrome, but keep the header
       element visible so the sidebar arrow button remains accessible. */
    [data-testid="stHeader"]        { background: transparent; }
    [data-testid="stDecoration"]    { display: none; }

    /* ── Make sidebar arrow button large and easy to tap on mobile ── */
    [data-testid="collapsedControl"] {
        width: 48px !important;
        height: 48px !important;
        border-radius: 50% !important;
        background-color: #0f4c75 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.30) !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    [data-testid="collapsedControl"] svg {
        fill: white !important;
        width: 22px !important;
        height: 22px !important;
    }

    /* ── Mobile: stack 3-col form to 1 col ───────────────────────── */
    @media (max-width: 640px) {
        div[data-testid="column"] {
            min-width: 100% !important;
            flex: 1 1 100% !important;
        }
        button[data-baseweb="tab"] {
            font-size: 11px !important;
            padding: 6px 4px !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

try:
    import shap
    import matplotlib.pyplot as plt
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# =========================
# 1) PAGE CONFIG
# =========================
st.set_page_config(
    page_title="CVD Risk – Stacking GenAI v6.0 (Longitudinal)",
    page_icon="🫀",
    layout="wide"
)

# =========================
# 2) LOAD ARTIFACTS
# =========================
DEFAULT_THRESHOLD = 0.40

ARTIFACTS = {
    "scaler": Path("scaler_24.pkl"),
    "rf":     Path("rf_clin24.pkl"),
    "xgb":    Path("xgb_clin24.pkl"),
    "meta":   Path("stack_meta_clin24.pkl"),
    "features": Path("features_24.json"),
    "data":     Path("frmgham2.csv"),
}

@st.cache_resource
def load_artifacts():
    missing = [k for k, p in ARTIFACTS.items() if not p.exists()]
    if missing:
        st.error("Missing model files: " + ", ".join(missing))
        st.stop()
    scaler    = joblib.load(ARTIFACTS["scaler"])
    rf_model  = joblib.load(ARTIFACTS["rf"])
    xgb_model = joblib.load(ARTIFACTS["xgb"])
    meta_model= joblib.load(ARTIFACTS["meta"])
    with open(ARTIFACTS["features"]) as f:
        features_24 = json.load(f)
    if not isinstance(features_24, list) or len(features_24) != 24:
        st.error("features_24.json must be a JSON list of exactly 24 feature names.")
        st.stop()

    # Load a stratified background sample from training data for SHAP.
    # This forces XGBoost TreeExplainer to compute interventional SHAP values
    # in probability space rather than log-odds, fixing f(x) display.
    bg_data = None
    if ARTIFACTS["data"].exists():
        try:
            df_bg = pd.read_csv(ARTIFACTS["data"])
            # Keep only the 24 model features, drop rows with NaNs
            feat_cols = [c for c in features_24 if c in df_bg.columns]
            if len(feat_cols) == 24:
                df_bg = df_bg[feat_cols].dropna()
                # Use up to 100 representative background rows (shap recommendation)
                n_bg = min(100, len(df_bg))
                df_bg = df_bg.sample(n=n_bg, random_state=42)
                bg_data = scaler.transform(df_bg.values.astype(float))
        except Exception:
            bg_data = None

    return scaler, rf_model, xgb_model, meta_model, features_24, bg_data

scaler, rf_model, xgb_model, meta_model, FEATURES_24, BG_DATA = load_artifacts()

# =========================
# 3) PLAIN-LANGUAGE LABELS
# =========================
# Maps clinical abbreviation → friendly patient label → tooltip help text
FIELD_CONFIG = {
    "SEX":      ("Sex",                             "Biological sex assigned at birth"),
    "AGE":      ("Age (years)",                     "Your current age"),
    "educ":     ("Education level",                 "Highest level of education completed"),
    "CIGPDAY":  ("Cigarettes per day",              "How many cigarettes you smoke daily (enter 0 if non-smoker)"),
    "SYSBP":    ("Systolic BP (top number, mmHg)",  "The top number of your blood pressure reading (e.g. 130 in '130/80')"),
    "DIABP":    ("Diastolic BP (bottom number, mmHg)", "The bottom number of your blood pressure reading (e.g. 80 in '130/80')"),
    "BMI":      ("Body Mass Index (BMI, kg/m²)",    "A measure of body fat based on height and weight. Normal is 18.5–24.9"),
    "HEARTRTE": ("Resting heart rate (bpm)",        "Your heart beats per minute at rest"),
    "TOTCHOL":  ("Total cholesterol (mg/dL)",       "Total amount of cholesterol in your blood. Desirable is below 200"),
    "HDLC":     ("HDL 'good' cholesterol (mg/dL)",  "Higher is better. Protects against heart disease. Goal ≥ 40 men / ≥ 50 women"),
    "LDLC":     ("LDL 'bad' cholesterol (mg/dL)",   "Lower is better. High LDL raises heart disease risk. Goal < 100"),
    "GLUCOSE":  ("Fasting blood sugar (mg/dL)",     "Blood glucose after not eating for 8+ hours. Normal is 70–99"),
    "DIABETES": ("Diagnosed with diabetes?",        "Has a doctor told you that you have diabetes?"),
    "BPMEDS":   ("On blood pressure medication?",   "Are you currently taking any pills or medication for blood pressure?"),
    "PREVHYP":  ("Past high blood pressure (hypertension)?", "Have you ever been told you had high blood pressure?"),
    "PREVCHD":  ("Past coronary heart disease?",    "Has a doctor ever diagnosed you with coronary artery disease or blocked arteries?"),
    "PREVAP":   ("Past angina (chest pain)?",       "Have you ever been diagnosed with angina — chest pain or tightness due to reduced blood flow?"),
    "PREVMI":   ("Past heart attack?",              "Have you ever had a heart attack (myocardial infarction)?"),
    "PREVSTRK": ("Past stroke?",                    "Have you ever had a stroke?"),
    "HOSPMI":   ("Hospitalized for heart attack?",  "Were you ever hospitalized specifically for a heart attack?"),
    "ANGINA":   ("Currently experiencing angina?",  "Do you currently have chest pain or tightness (angina)?"),
    "MI_FCHD":  ("Family history of heart attack?", "Has a parent or sibling had a heart attack or coronary heart disease?"),
    "STROKE":   ("Current stroke diagnosis?",       "Have you been told you currently have stroke-related condition?"),
    "HYPERTEN": ("Currently diagnosed with hypertension?", "Has your doctor diagnosed you with hypertension (high blood pressure)?"),
}

# SHAP-friendly plain text for patient display (all 24 features)
SHAP_FRIENDLY = {
    "SEX":      "Male sex",
    "TOTCHOL":  "Total cholesterol",
    "AGE":      "Age",
    "SYSBP":    "Systolic blood pressure (top number)",
    "DIABP":    "Diastolic blood pressure",
    "CIGPDAY":  "Smoking (cigarettes per day)",
    "BMI":      "Body weight (BMI)",
    "DIABETES": "Diabetes diagnosis",
    "BPMEDS":   "Blood pressure medication",
    "HEARTRTE": "Heart rate",
    "GLUCOSE":  "Blood sugar levels",
    "educ":     "Education level",
    "PREVCHD":  "Prior heart disease",
    "PREVAP":   "Prior angina (chest pain)",
    "PREVMI":   "Prior heart attack",
    "PREVSTRK": "Prior stroke",
    "PREVHYP":  "History of high blood pressure",
    "HOSPMI":   "Hospitalized for heart attack",
    "HDLC":     "HDL good cholesterol",
    "LDLC":     "LDL bad cholesterol",
    "ANGINA":   "Current chest pain (angina)",
    "MI_FCHD":  "Family history of heart attack",
    "STROKE":   "Current stroke diagnosis",
    "HYPERTEN": "Current hypertension diagnosis",
}

# Binary (yes/no) features — used to convert 0/1 to No/Yes in patient waterfall
BINARY_FEATURES = {
    "DIABETES", "BPMEDS", "PREVCHD", "PREVAP", "PREVMI", "PREVSTRK",
    "PREVHYP", "HOSPMI", "ANGINA", "MI_FCHD", "STROKE", "HYPERTEN",
}
# Sex maps: 1=Male, 2=Female in this dataset encoding
SEX_MAP = {1.0: "Male", 2.0: "Female", 0.0: "Female"}


# ── Reference healthy values for clinical risk contribution chart ──────────
# For each feature, this is the "healthy baseline" used to compute
# how much each factor contributes to THIS patient's risk above baseline.
HEALTHY_REF = {
    "SEX":      None,   # keep patient's own sex (not modifiable)
    "TOTCHOL":  180.0,  # desirable total cholesterol
    "AGE":      None,   # keep patient's own age (not modifiable)
    "SYSBP":    115.0,  # optimal systolic BP
    "DIABP":    75.0,   # optimal diastolic BP
    "CIGPDAY":  0.0,    # non-smoker
    "BMI":      22.0,   # healthy BMI midpoint
    "DIABETES": 0.0,    # no diabetes
    "BPMEDS":   0.0,    # no BP meds needed
    "HEARTRTE": 65.0,   # healthy resting heart rate
    "GLUCOSE":  85.0,   # optimal fasting glucose
    "educ":     None,   # keep patient's own education
    "PREVCHD":  0.0,    # no prior heart disease
    "PREVAP":   0.0,
    "PREVMI":   0.0,
    "PREVSTRK": 0.0,
    "PREVHYP":  0.0,
    "HOSPMI":   0.0,
    "HDLC":     60.0,   # optimal HDL (higher is better)
    "LDLC":     70.0,   # optimal LDL
    "ANGINA":   0.0,
    "MI_FCHD":  0.0,
    "STROKE":   0.0,
    "HYPERTEN": 0.0,
}
# =========================
# 4) HELPERS
# =========================
def interpret_risk(prob: float):
    if prob < 0.05:  return "Low risk",          "🟢"
    if prob < 0.10:  return "Borderline risk",   "🟡"
    if prob < 0.20:  return "Intermediate risk", "🟠"
    return "High risk", "🔴"

def _as_int_yesno(v: str) -> int:
    return 1 if v == "Yes" else 0

def stacking_predict_proba_24(df_input: pd.DataFrame, threshold: float):
    X   = df_input.values.astype(float)
    Xs  = scaler.transform(X)
    p_rf  = rf_model.predict_proba(Xs)[:, 1]
    p_xgb = xgb_model.predict_proba(Xs)[:, 1]
    stack_in = np.column_stack([p_rf, p_xgb])
    p_final  = meta_model.predict_proba(stack_in)[:, 1]
    final_prob  = float(p_final[0])
    final_label = int(final_prob >= threshold)
    component_probs = {
        "RF (Clinical)":  float(p_rf[0]),
        "XGB (Clinical)": float(p_xgb[0]),
    }
    return final_prob, final_label, component_probs

def bie_scenarios_24(df_patient: pd.DataFrame, threshold: float, include_advanced: bool = True):
    base_prob, _, _ = stacking_predict_proba_24(df_patient, threshold=threshold)
    scenarios = [("Baseline", "Current profile", df_patient.copy())]
    if float(df_patient["CIGPDAY"].iloc[0]) > 0:
        d = df_patient.copy(); d["CIGPDAY"] = 0.0
        scenarios.append(("Quit smoking", "Cigarettes/day → 0", d))
    sysbp = float(df_patient["SYSBP"].iloc[0])
    d = df_patient.copy(); d["SYSBP"] = max(sysbp - 10.0, 90.0)
    scenarios.append(("Lower blood pressure by 10 mmHg", f"Top BP: {sysbp:.0f} → {d['SYSBP'].iloc[0]:.0f}", d))
    if include_advanced:
        bmi = float(df_patient["BMI"].iloc[0])
        d = df_patient.copy(); d["BMI"] = max(bmi - 2.0, 15.0)
        scenarios.append(("Lose weight (BMI down 2)", f"BMI: {bmi:.1f} → {d['BMI'].iloc[0]:.1f}", d))
        tc = float(df_patient["TOTCHOL"].iloc[0])
        d = df_patient.copy(); d["TOTCHOL"] = max(tc - 20.0, 100.0)
        scenarios.append(("Lower cholesterol by 20 mg/dL", f"Total Chol: {tc:.0f} → {d['TOTCHOL'].iloc[0]:.0f}", d))
    gl = float(df_patient["GLUCOSE"].iloc[0])
    d = df_patient.copy(); d["GLUCOSE"] = max(gl - 10.0, 60.0)
    scenarios.append(("Lower blood sugar by 10 mg/dL", f"Blood sugar: {gl:.0f} → {d['GLUCOSE'].iloc[0]:.0f}", d))
    rows, best = [], None
    for name, desc, dfx in scenarios:
        p, _, _ = stacking_predict_proba_24(dfx, threshold=threshold)
        abs_change = (p - base_prob) * 100.0
        rel_change = (p - base_prob) / base_prob * 100.0 if base_prob > 0 else 0.0
        rows.append({"Scenario": name, "Description": desc, "Risk (%)": p * 100.0,
                     "Change (pp)": abs_change, "Relative change (%)": rel_change})
        if name != "Baseline":
            drop = base_prob - p
            if best is None or drop > best["drop"]:
                best = {"name": name, "p": p, "drop": drop}
    return base_prob, pd.DataFrame(rows), best

def _get_shap_explainers():
    if not SHAP_AVAILABLE: return None, None
    if "rf_explainer" not in st.session_state:
        # RF: background data improves SHAP accuracy but isn't required for
        # probability output — RF TreeExplainer already works in prob space
        rf_kwargs = {"model_output": "probability"}
        if BG_DATA is not None:
            rf_kwargs["data"] = BG_DATA
        st.session_state["rf_explainer"] = shap.TreeExplainer(rf_model, **rf_kwargs)

        # XGB: passing background data is the ONLY reliable way to get SHAP
        # values in probability space for XGBoost across all library versions.
        # Without it, TreeExplainer uses the tree-path method which returns
        # log-odds values (causing f(x) = -1.517 instead of 0.18).
        if BG_DATA is not None:
            st.session_state["xgb_explainer"] = shap.TreeExplainer(
                xgb_model,
                data=BG_DATA,
                model_output="probability",
            )
        else:
            # No training data available — use log-odds with fallback conversion
            st.session_state["xgb_explainer"] = shap.TreeExplainer(xgb_model)
    return st.session_state["rf_explainer"], st.session_state["xgb_explainer"]


def _shap_legend(is_patient: bool = False):
    """
    Render a colour-coded legend + symbol guide below a SHAP waterfall plot.
    Call immediately after _shap_waterfall().
    """
    if is_patient:
        fx_label   = "**f(x)** — Your personal predicted risk score (shown at the top of the chart)"
        efx_label  = "**E[f(X)]** — The average risk score across all patients in the training data (the starting baseline)"
        red_pos    = "**Red bar (right of centre)** — This factor is *raising* your risk above the baseline"
        red_neg    = "**Red bar (left of centre, labelled in red)** — A smaller red bar pulling toward baseline"
        blue_pos   = "**Blue bar (right of centre, labelled in blue)** — A smaller blue bar pushing toward baseline"
        blue_neg   = "**Blue bar (left of centre)** — This factor is *lowering* your risk below the baseline"
        bar_width  = "**Bar width** — The wider the bar, the stronger the impact of that factor"
        value_note = "**Number on left (e.g. '1 = DIABETES')** — Your actual value for that factor"
        footer     = ("⚠️ Important: A factor labelled \"No\" can still show a red (risk-raising) bar. "
                      "This happens when the AI compares you to the average patient in its training data — "
                      "not to a perfect-health benchmark. For example, if most training patients with "
                      "your age and diabetes profile also had no family history, the model may still "
                      "associate that combination with above-average risk. "
                      "These are statistical patterns, not medical diagnoses. Always discuss with your doctor.")
    else:
        fx_label   = "**f(x)** — This model's predicted CVD probability for this patient"
        efx_label  = "**E[f(X)]** — Population baseline: average predicted probability across all training patients"
        red_pos    = "**Red bar →** — Feature pushes prediction *above* baseline (risk-increasing)"
        red_neg    = "**Red label, small bar** — Small positive contribution"
        blue_pos   = "**Blue label, small bar** — Small negative contribution"
        blue_neg   = "**Blue bar ←** — Feature pushes prediction *below* baseline (risk-reducing)"
        bar_width  = "**Bar magnitude** — Proportional to absolute SHAP value; wider = stronger influence"
        value_note = "**Left-side annotation (e.g. '110 = GLUCOSE')** — Patient's actual feature value"
        footer     = ("Note: SHAP values reflect the base model's learned associations, not causal treatment "
                      "effects. The stacked meta-learner combines RF and XGB outputs; individual waterfall "
                      "f(x) values may differ from the final stacked prediction shown above.")

    st.markdown(
        f"""
<div style="background:#f8f8f6;border:0.5px solid #d3d1c7;border-radius:10px;
            padding:14px 18px;margin-top:12px;font-size:13px;line-height:1.8">
<strong style="font-size:13px;color:#0f4c75">How to read this chart</strong><br><br>
{fx_label}<br>
{efx_label}<br><br>
<span style="color:#d85a30">{'■'}</span> {red_pos}<br>
<span style="color:#378add">{'■'}</span> {blue_neg}<br>
{bar_width}<br>
{value_note}<br><br>
<em style="color:#888780;font-size:12px">{footer}</em>
</div>
""",
        unsafe_allow_html=True,
    )


def _shap_waterfall(explainer, X_row_1xF, X_row_raw, feature_names,
                    title: str, max_display: int = 12,
                    model_prob: float = None):
    """
    Render a SHAP waterfall plot for a single patient row.

    Parameters
    ----------
    explainer    : shap.TreeExplainer
    X_row_1xF   : scaled feature array, shape (1, n_features)
    X_row_raw   : unscaled feature array, shape (1, n_features)
    feature_names: list[str]
    title        : plot title
    max_display  : features shown (sorted by |SHAP value|)
    model_prob   : actual predict_proba output for this row (used to
                   anchor f(x) so it always matches the model output)
    """
    if not SHAP_AVAILABLE or explainer is None:
        st.info("SHAP not available in this deployment.")
        return

    # ── 1. Compute raw SHAP values ──────────────────────────────────────────
    sv_raw = explainer.shap_values(X_row_1xF)

    if isinstance(sv_raw, list) and len(sv_raw) == 2:
        sv_raw = sv_raw[1]
    if hasattr(sv_raw, "values"):
        sv_raw = sv_raw.values
    sv_raw = np.array(sv_raw)
    if sv_raw.ndim == 3 and sv_raw.shape[-1] == 2:
        sv_raw = sv_raw[:, :, 1]
    if sv_raw.ndim != 2 or sv_raw.shape[0] != 1:
        st.warning(f"Unexpected SHAP shape: {sv_raw.shape}. Skipping waterfall.")
        return

    shap_vals_1d = np.ravel(sv_raw[0])
    raw_vals_1d  = np.ravel(X_row_raw[0])

    if len(feature_names) != len(shap_vals_1d):
        st.warning("Feature name / SHAP value length mismatch.")
        return

    # ── 2. Determine base value and normalise to probability space ───────────
    ev = explainer.expected_value
    if isinstance(ev, (list, np.ndarray)):
        base_val = float(ev[1]) if len(ev) > 1 else float(ev[0])
    else:
        base_val = float(ev)

    # Compute what f(x) would be using raw SHAP: base + sum(shap)
    raw_fx = base_val + float(np.sum(shap_vals_1d))

    # Detect log-odds space: f(x) outside [0,1] OR base outside [0,1]
    is_log_odds = not (0.0 <= base_val <= 1.0) or not (0.0 <= raw_fx <= 1.0)

    if is_log_odds:
        # Convert base to probability via sigmoid
        base_val = float(1.0 / (1.0 + np.exp(-base_val)))
        # Scale SHAP values: first-order delta method dP/d(logit) = P(1-P)
        scale = base_val * (1.0 - base_val)
        shap_vals_1d = shap_vals_1d * scale

    # ── 3. Anchor f(x) to the actual model predict_proba output ─────────────
    # If model_prob is provided, rescale SHAP values so that
    # base_val + sum(shap_vals) == model_prob exactly.
    # This ensures the waterfall f(x) label always matches the displayed risk %.
    if model_prob is not None:
        shap_sum = float(np.sum(shap_vals_1d))
        target_sum = float(model_prob) - base_val
        if abs(shap_sum) > 1e-10:
            shap_vals_1d = shap_vals_1d * (target_sum / shap_sum)

    # ── 4. Build shap.Explanation and plot ───────────────────────────────────
    explanation = shap.Explanation(
        values        = shap_vals_1d,
        base_values   = base_val,
        data          = raw_vals_1d,
        feature_names = feature_names,
    )

    # Trim max_display to exclude features whose |SHAP| rounds to 0.00
    # so bars labeled '+0' or '-0' never appear.
    sorted_abs = np.sort(np.abs(shap_vals_1d))[::-1]
    # Find how many features have |SHAP| >= 0.005 (shows as ≥ +0.01 with 2dp)
    meaningful = int(np.sum(sorted_abs >= 0.005))
    # Always show at least 5, at most max_display
    effective_max = max(5, min(max_display, meaningful))

    fig, ax = plt.subplots(figsize=(9, max(6, effective_max * 0.52)))
    shap.plots.waterfall(explanation, max_display=effective_max, show=False)
    fig = plt.gcf()
    fig.suptitle(title, fontsize=11, fontweight="bold",
                 color="#0f4c75", y=1.02)
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.12)
    st.pyplot(fig, clear_figure=True)

# =========================
# 5) LONGITUDINAL TRACKER (session-state backed)
# =========================
HISTORY_KEY = "cvd_history"

def history_load() -> list:
    return st.session_state.get(HISTORY_KEY, [])

def history_save(entry: dict):
    h = history_load()
    h.append(entry)
    st.session_state[HISTORY_KEY] = h

def history_clear():
    st.session_state[HISTORY_KEY] = []

def history_as_df() -> pd.DataFrame:
    h = history_load()
    if not h: return pd.DataFrame()
    return pd.DataFrame(h)

def history_export_csv() -> bytes:
    df = history_as_df()
    return df.to_csv(index=False).encode("utf-8")

# =========================
# 6) INPUT FORM (patient-friendly labels)
# =========================
def build_input_df_24(is_patient: bool):
    def label(field):
        friendly, tip = FIELD_CONFIG[field]
        return friendly if is_patient else f"{field} — {friendly}", tip

    lbl, tip = label("SEX")
    sex = st.selectbox(lbl, ["Male", "Female"], index=0, key="calc_sex", help=tip)
    col1, col2, col3 = st.columns(3)

    with col1:
        lbl, tip = label("AGE")
        age = st.number_input(lbl, 18, 100, 55, 1, key="calc_age", help=tip)
        educ_opts = ["1 – Some high school", "2 – High school graduate",
                     "3 – Some college", "4 – College graduate"]
        lbl_e = "Education level" if is_patient else "educ — Education level"
        educ = st.selectbox(lbl_e, educ_opts, index=2, key="calc_educ",
                            help=FIELD_CONFIG["educ"][1])
        lbl, tip = label("CIGPDAY")
        cigs = st.number_input(lbl, 0, 80, 0, 1, key="calc_cigs", help=tip)

    with col2:
        lbl, tip = label("SYSBP")
        sysbp = st.number_input(lbl, 80, 250, 130, 1, key="calc_sysbp", help=tip)
        lbl, tip = label("DIABP")
        diabp = st.number_input(lbl, 40, 160, 80, 1, key="calc_diabp", help=tip)
        lbl, tip = label("BMI")
        bmi = st.number_input(lbl, 15.0, 60.0, 27.0, 0.1, key="calc_bmi", help=tip)
        lbl, tip = label("HEARTRTE")
        heartrate = st.number_input(lbl, 30, 200, 70, 1, key="calc_hr", help=tip)

    with col3:
        lbl, tip = label("TOTCHOL")
        totchol = st.number_input(lbl, 80, 500, 200, 1, key="calc_totchol", help=tip)
        lbl, tip = label("HDLC")
        hdlc = st.number_input(lbl, 10, 150, 45, 1, key="calc_hdlc", help=tip)
        lbl, tip = label("LDLC")
        ldlc = st.number_input(lbl, 10, 300, 120, 1, key="calc_ldlc", help=tip)
        lbl, tip = label("GLUCOSE")
        glucose = st.number_input(lbl, 40, 400, 90, 1, key="calc_glucose", help=tip)

    st.markdown("---")
    st.markdown("#### Medical history & conditions" if is_patient else "#### Clinical conditions & history")
    if is_patient:
        st.caption("Please answer based on any formal diagnoses from a doctor.")

    col4, col5, col6 = st.columns(3)

    def yesno(field, col, key):
        lbl, tip = label(field)
        with col:
            return st.selectbox(lbl, ["No", "Yes"], index=0, key=key, help=tip)

    with col4:
        diabetes = yesno("DIABETES", col4, "calc_diabetes")
        bpmeds   = yesno("BPMEDS",   col4, "calc_bpmeds")
        prevhyp  = yesno("PREVHYP",  col4, "calc_prevhyp")

    with col5:
        prevchd  = yesno("PREVCHD",  col5, "calc_prevchd")
        prevap   = yesno("PREVAP",   col5, "calc_prevap")
        prevmi   = yesno("PREVMI",   col5, "calc_prevmi")
        hospmi   = yesno("HOSPMI",   col5, "calc_hospmi")

    with col6:
        prevstrk = yesno("PREVSTRK", col6, "calc_prevstrk")
        angina   = yesno("ANGINA",   col6, "calc_angina")
        mi_fchd  = yesno("MI_FCHD",  col6, "calc_mi_fchd")
        stroke   = yesno("STROKE",   col6, "calc_stroke")
        hyperten = yesno("HYPERTEN", col6, "calc_hyperten")

    row = {
        "SEX": 1 if sex == "Male" else 0,
        "TOTCHOL": float(totchol), "AGE": float(age),
        "SYSBP": float(sysbp),    "DIABP": float(diabp),
        "CIGPDAY": float(cigs),   "BMI": float(bmi),
        "DIABETES": _as_int_yesno(diabetes), "BPMEDS": _as_int_yesno(bpmeds),
        "HEARTRTE": float(heartrate),         "GLUCOSE": float(glucose),
        "educ": int(educ.split("–")[0].strip()),
        "PREVCHD":  _as_int_yesno(prevchd),  "PREVAP":   _as_int_yesno(prevap),
        "PREVMI":   _as_int_yesno(prevmi),   "PREVSTRK": _as_int_yesno(prevstrk),
        "PREVHYP":  _as_int_yesno(prevhyp),  "HOSPMI":   _as_int_yesno(hospmi),
        "HDLC": float(hdlc),   "LDLC": float(ldlc),
        "ANGINA":   _as_int_yesno(angina),   "MI_FCHD":  _as_int_yesno(mi_fchd),
        "STROKE":   _as_int_yesno(stroke),   "HYPERTEN": _as_int_yesno(hyperten),
    }
    try:
        row_ordered = {feat: row[feat] for feat in FEATURES_24}
    except KeyError as e:
        st.error(f"Feature mapping error: {e}"); st.stop()

    return pd.DataFrame([row_ordered])


# =========================
# 7) ESCALATION CARD
# =========================
def show_escalation_card(prob: float, category: str, is_patient: bool):
    """Show tiered escalation guidance based on risk level."""
    if prob >= 0.20:
        st.error(
            "🔴 **High Risk — Please speak with your doctor soon.**\n\n"
            "Your estimated 10-year risk is **high**. This does not mean a heart attack is certain, "
            "but it does mean that speaking with a cardiologist or your primary care physician is strongly recommended. "
            "Bring this report to your next appointment.\n\n"
            "**Suggested next steps:**\n"
            "- Schedule an appointment with your doctor or cardiologist\n"
            "- Ask about blood pressure and cholesterol management\n"
            "- Discuss whether medication or lifestyle changes are appropriate for you\n\n"
            "*This is not a diagnosis. Always consult a licensed healthcare professional.*"
        )
    elif prob >= 0.10:
        st.warning(
            "🟠 **Intermediate Risk — Talk to your doctor at your next visit.**\n\n"
            "Your risk is in the intermediate range. Discuss this result with your primary care physician. "
            "Lifestyle changes (diet, exercise, smoking cessation) can meaningfully reduce risk at this level.\n\n"
            "**Suggested next steps:**\n"
            "- Mention this risk score at your next checkup\n"
            "- Ask about blood pressure, cholesterol, and blood sugar targets\n"
            "- Consider a heart-healthy diet and regular physical activity\n\n"
            "*This is not a diagnosis. Always consult a licensed healthcare professional.*"
        )
    elif prob >= 0.05:
        st.info(
            "🟡 **Borderline Risk — Worth discussing with your doctor.**\n\n"
            "Your risk is borderline. This is a good time to focus on prevention — small lifestyle improvements "
            "can keep this from rising.\n\n"
            "**Suggested next steps:**\n"
            "- Continue routine checkups\n"
            "- Focus on diet, exercise, and avoiding smoking\n"
            "- Re-assess in 6–12 months"
        )
    else:
        st.success(
            "🟢 **Lower Risk — Keep up the healthy habits!**\n\n"
            "Your current profile suggests a lower 10-year heart disease risk. "
            "Maintain healthy habits and continue routine preventive care."
        )


# =========================
# 8) SIDEBAR
# =========================
with st.sidebar:
    st.markdown(
        """
        <h2 style='margin-bottom:0;'>🫀 CVD Stacking GenAI</h2>
        <p style='margin-top:4px;font-size:13px;'>
        <b>v6.0 – Longitudinal Risk Tracker</b><br>
        Framingham-based • 24-feature Clinical+History Model
        </p>
        """,
        unsafe_allow_html=True
    )
    st.markdown("---")

    user_mode = st.radio(
        "Display mode:",
        ["Patient Mode", "Clinician / Research Mode"],
        index=0,
        help="Patient Mode uses plain language. Clinician Mode shows full technical details."
    )
    IS_PATIENT_MODE = (user_mode == "Patient Mode")

    threshold = st.slider(
        "Alert threshold (probability of CVD)",
        0.10, 0.90, DEFAULT_THRESHOLD, 0.05,
        help="If predicted risk ≥ threshold, the model flags the patient as 'At Risk'.",
        key="sidebar_threshold"
    )

    if IS_PATIENT_MODE:
        show_components = False
        show_shap = False
    else:
        show_components = st.checkbox("Show component model probabilities (RF/XGB)", value=True, key="sidebar_comp")
        show_shap = st.checkbox("Show SHAP local explanation (RF/XGB)", value=False,
                                help="Requires shap + matplotlib in requirements.txt.", key="sidebar_shap")

    st.markdown("---")

    # Longitudinal quick-stats
    h = history_load()
    if h:
        df_h = pd.DataFrame(h)
        st.markdown(f"**📊 Saved assessments:** {len(h)}")
        latest = df_h.iloc[-1]
        first  = df_h.iloc[0]
        delta  = latest["risk_pct"] - first["risk_pct"]
        arrow  = "📉" if delta < 0 else ("📈" if delta > 0 else "➡️")
        st.markdown(f"**Latest risk:** {latest['risk_pct']:.1f}%")
        st.markdown(f"**Trend since first visit:** {arrow} {delta:+.1f} pp")

    st.markdown("---")
    st.markdown(
        """
        **Disclaimer**  
        This tool is for **research & demonstration** only and must not be used as
        a standalone diagnostic system.
        """
    )


# =========================
# 9) HERO HEADER
# =========================

# Mobile-only pill — hidden on desktop via CSS display:none + media override
st.markdown(
    """
    <div id="mob-hint" style="display:none;background:#fff3cd;border:1px solid #ffc107;
        border-radius:8px;padding:8px 14px;margin-bottom:8px;font-size:13px;color:#856404;">
        &#9776;&nbsp; Tap the <b>navy circle arrow</b> at top-left to open Settings
        &nbsp;(Display mode · Alert threshold · SHAP options)
    </div>
    <style>
        @media (max-width: 768px) { #mob-hint { display: block !important; } }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="background-color:#0f4c75;padding:18px;border-radius:8px;margin-bottom:16px;">
      <h1 style="color:white;margin-bottom:4px;">CVD Risk Prediction – Clinical Risk Stratification Model (v6.0)</h1>
      <p style="color:#e0f2f1;margin:0;font-size:14px;">
        10-Year Cardiovascular Risk Estimation • Longitudinal Tracking • Patient &amp; Clinician Modes
      </p>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================
# 10) TABS
# =========================
tab_calc, tab_trend, tab_bie, tab_coach, tab_model, tab_faq = st.tabs([
    "🧮 Risk Calculator",
    "📈 My Risk Over Time",
    "🔬 What-If Simulator (BIE)",
    "💬 Health Coach",
    "🧬 Model & Data",
    "❓ FAQ & Notes",
])


# ─────────────────────────────────────────
# TAB 1 – RISK CALCULATOR
# ─────────────────────────────────────────
with tab_calc:
    st.markdown(
        "Enter your health information below and click **Run Risk Prediction** "
        "to receive your estimated 10-year heart disease risk score."
        if IS_PATIENT_MODE else
        "Enter patient data (24-feature clinical+history model) to obtain a **10-year CVD risk estimate**."
    )

    if IS_PATIENT_MODE:
        st.info(
            "ℹ️ Hover over any field label for an explanation of what it means and how to find the value. "
            "All values come from a standard blood panel or checkup."
        )

    st.subheader("Your Health Profile" if IS_PATIENT_MODE else "Patient Profile, Clinical Risk Factors & Prior History")
    df_input = build_input_df_24(IS_PATIENT_MODE)

    # Optional visit label for history
    visit_label = st.text_input(
        "Visit label (optional)",
        placeholder="e.g. Jan 2025 checkup, After starting medication…",
        help="Add a short note to identify this assessment in your history.",
        key="calc_visit_label"
    )

    if not IS_PATIENT_MODE:
        with st.expander("View encoded feature vector (for experts)", expanded=False):
            st.dataframe(df_input.style.format(precision=2), use_container_width=True)

    run_btn = st.button("🫀 Run Risk Prediction", type="primary", key="btn_run_calc")

    if run_btn:
        with st.spinner("Analyzing your health data…"):
            final_prob, final_label, component_probs = stacking_predict_proba_24(df_input, threshold=threshold)

        # Save to session state for other tabs
        st.session_state["v6_last_input_df"]    = df_input
        st.session_state["v6_last_prob"]        = final_prob
        st.session_state["v6_last_label"]       = final_label
        st.session_state["v6_last_components"]  = component_probs
        st.session_state["v6_last_threshold"]   = threshold

        # Save to longitudinal history
        entry = {
            "date":       datetime.date.today().isoformat(),
            "label":      visit_label if visit_label else f"Assessment {len(history_load())+1}",
            "risk_pct":   round(final_prob * 100, 2),
            "risk_cat":   interpret_risk(final_prob)[0],
            "flagged":    bool(final_label),
            "sysbp":      float(df_input["SYSBP"].iloc[0]),
            "totchol":    float(df_input["TOTCHOL"].iloc[0]),
            "bmi":        float(df_input["BMI"].iloc[0]),
            "glucose":    float(df_input["GLUCOSE"].iloc[0]),
            "cigpday":    float(df_input["CIGPDAY"].iloc[0]),
            "threshold":  threshold,
        }
        history_save(entry)

        category, color = interpret_risk(final_prob)

        # ── Results ──
        st.markdown("---")
        st.markdown("### Your Result" if IS_PATIENT_MODE else "### Prediction Result")

        col_res1, col_res2 = st.columns([2, 1])
        with col_res1:
            if IS_PATIENT_MODE:
                st.metric("Your estimated 10-year heart disease risk", f"{final_prob*100:.1f}%")
                st.markdown(
                    f"This means roughly **{int(round(final_prob*100))} out of 100** people "
                    "with a similar health profile may develop heart disease over the next 10 years."
                )
                patient_cat = {
                    "Low risk":          "Lower risk — keep up the healthy habits.",
                    "Borderline risk":   "Moderate risk — worth discussing with your doctor.",
                    "Intermediate risk": "Higher risk — medical guidance is recommended.",
                    "High risk":         "High risk — please seek medical follow-up.",
                }
                st.markdown(f"**Risk level:** {color} **{category}** — {patient_cat.get(category,'')}")
            else:
                st.metric("Estimated 10-year CVD risk", f"{final_prob*100:.1f}%")
                st.markdown(f"**Risk category:** {color} **{category}**")
                st.markdown(
                    f"**Model decision at threshold {threshold:.2f}:** "
                    f"{'⚠️ At Risk (1)' if final_label == 1 else '✅ Not Flagged (0)'}"
                )

        with col_res2:
            guide_rows = [
                ("< 5%",  "🟢", "Low"),
                ("5–9%",  "🟡", "Borderline"),
                ("10–19%","🟠", "Intermediate"),
                ("≥ 20%", "🔴", "High"),
            ]
            st.markdown("**Risk level guide**" if IS_PATIENT_MODE else "**Interpretation guide**")
            for r, ic, lbl in guide_rows:
                st.markdown(f"- {ic} {r}: {lbl}")

        # Escalation guidance
        st.markdown("---")
        show_escalation_card(final_prob, category, IS_PATIENT_MODE)

        # Patient SHAP — waterfall (XGB only, cleaner for patients)
        if IS_PATIENT_MODE:
            st.markdown("---")
            st.markdown("### What's driving your score?")
            st.markdown(
                f"Your current 10-year heart disease risk is **{final_prob*100:.1f}%**. "
                "The chart below shows which of your health factors are contributing most to that risk, "
                "and — importantly — which ones you may be able to improve. "
                "Each bar shows how much that factor is estimated to be adding to your personal risk score."
            )

            # ── Clinical Risk Contribution Chart (BIE-based counterfactual) ──
            # For each feature, we ask: "what would the risk be if this factor
            # were at the healthy reference level?"  The difference = contribution.
            # This is intuitive for patients: every bar is positive and means
            # "this factor is adding X% to your risk."  No baseline confusion.
            contributions = []
            row_dict = df_input.iloc[0].to_dict()

            for feat in FEATURES_24:
                ref_val = HEALTHY_REF.get(feat)
                if ref_val is None:
                    continue   # skip non-modifiable (age, sex, education)
                patient_val = float(row_dict[feat])
                if abs(patient_val - ref_val) < 0.01:
                    continue   # already at healthy reference, no contribution

                # Build counterfactual row with this feature set to healthy ref
                cf_row = row_dict.copy()
                cf_row[feat] = ref_val
                cf_df = pd.DataFrame([{f: cf_row[f] for f in FEATURES_24}])
                cf_prob, _, _ = stacking_predict_proba_24(cf_df, threshold=threshold)

                contribution_pp = (final_prob - cf_prob) * 100.0
                if contribution_pp > 0.05:   # only show factors that meaningfully raise risk
                    label = SHAP_FRIENDLY.get(feat, feat)
                    if feat in BINARY_FEATURES:
                        val_str = "Yes" if patient_val >= 1 else "No"
                    elif patient_val == int(patient_val):
                        val_str = str(int(patient_val))
                    else:
                        val_str = f"{patient_val:.1f}"
                    contributions.append({
                        "feature":   feat,
                        "label":     label,
                        "value":     val_str,
                        "contrib_pp": contribution_pp,
                        "modifiable": feat not in {"PREVCHD","PREVMI","PREVSTRK",
                                                    "HOSPMI","MI_FCHD","PREVAP",
                                                    "PREVHYP","HYPERTEN","STROKE",
                                                    "ANGINA","PREVSTRK"},
                    })

            if contributions:
                contributions.sort(key=lambda x: x["contrib_pp"], reverse=True)
                top = contributions[:12]   # show top 12 contributors

                labels   = [f"{c['label']}: {c['value']}" for c in top]
                values   = [c["contrib_pp"] for c in top]
                colors   = ["#D85A30" if c["modifiable"] else "#888780" for c in top]

                fig, ax = plt.subplots(figsize=(9, max(5, len(top) * 0.52)))
                bars = ax.barh(range(len(top))[::-1], values, color=colors,
                               height=0.6, edgecolor="none")

                # Value labels on bars
                for i, (bar, val) in enumerate(zip(bars, values)):
                    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                            f"+{val:.1f} pp", va="center", ha="left",
                            fontsize=10, color="#444441", fontweight="500")

                ax.set_yticks(range(len(top))[::-1])
                ax.set_yticklabels(labels, fontsize=10)
                ax.set_xlabel("Estimated contribution to your risk (percentage points)", fontsize=10)
                ax.set_title(f"Your personal risk factors — contributing to your {final_prob*100:.1f}% score",
                             fontsize=11, fontweight="bold", color="#0f4c75", pad=12)
                ax.axvline(0, color="#d3d1c7", linewidth=0.8)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["left"].set_visible(False)
                ax.set_xlim(0, max(values) * 1.28)
                plt.tight_layout()
                st.pyplot(fig, clear_figure=True)

                # Legend
                st.markdown(
                    """
<div style="background:#f8f8f6;border:0.5px solid #d3d1c7;border-radius:10px;
            padding:14px 18px;margin-top:10px;font-size:13px;line-height:1.9">
<strong style="font-size:13px;color:#0f4c75">How to read this chart</strong><br><br>
Each bar shows how much that health factor is estimated to be adding to your personal risk score,
measured in <em>percentage points (pp)</em>.<br><br>
<span style="display:inline-block;width:12px;height:12px;background:#D85A30;
      border-radius:2px;vertical-align:middle;margin-right:6px"></span>
<strong>Orange bar</strong> — a factor you may be able to improve (e.g. blood pressure, cholesterol, smoking, weight)<br>
<span style="display:inline-block;width:12px;height:12px;background:#888780;
      border-radius:2px;vertical-align:middle;margin-right:6px"></span>
<strong>Grey bar</strong> — a factor from your medical history (less directly modifiable, but still important to discuss with your doctor)<br><br>
<strong>Example:</strong> a bar showing <em>+4.5 pp</em> for "Diabetes: Yes" means the model estimates
that diabetes is adding approximately 4.5 percentage points to your 10-year risk.<br><br>
Only factors currently above their healthy reference level are shown.
Factors already at a healthy level (e.g. you are a non-smoker) are not displayed
because they are not contributing extra risk.<br><br>
<em style="color:#888780;font-size:12px">These are model-based estimates from observational data — not a medical diagnosis.
Always discuss your results with your doctor before making any health decisions.</em>
</div>
""",
                    unsafe_allow_html=True,
                )
            else:
                st.success(
                    "All of your measurable risk factors are at or near healthy reference levels. "
                    "Your risk score is driven primarily by factors outside this chart. "
                    "Continue your current healthy habits and discuss with your doctor."
                )

        # Clinician components + SHAP
        if show_components:
            st.markdown("### Component Model Contributions")
            comp_df = pd.DataFrame(
                {"Model": list(component_probs.keys()),
                 "Predicted CVD risk (%)": [p * 100 for p in component_probs.values()]}
            )
            st.bar_chart(comp_df.set_index("Model"))

        if show_shap:
            with st.expander("Local SHAP explanation — base model drivers", expanded=False):
                st.markdown(
                    "SHAP (SHapley Additive exPlanations) shows **why** each base model "
                    "scored this patient the way it did — which features pushed the predicted "
                    "risk up or down, and by how much. Two waterfall charts are shown: one for "
                    "the Random Forest (RF) base model and one for XGBoost (XGB). The final "
                    f"stacked prediction of **{final_prob*100:.1f}%** is produced by a "
                    "Logistic Regression meta-learner that combines both base model outputs."
                )
                if not SHAP_AVAILABLE:
                    st.info("Add shap + matplotlib to requirements.txt to enable.")
                else:
                    rf_exp, xgb_exp = _get_shap_explainers()
                    X_raw    = df_input.values.astype(float)
                    X_scaled = scaler.transform(X_raw)

                    # Get each base model's individual predict_proba to anchor f(x)
                    _p_rf  = float(rf_model.predict_proba(X_scaled)[:, 1][0])
                    _p_xgb = float(xgb_model.predict_proba(X_scaled)[:, 1][0])

                    # ── RF waterfall ──
                    st.markdown("#### Random Forest (RF) — SHAP waterfall")
                    st.markdown(
                        f"The RF base model predicts a **{_p_rf*100:.1f}% CVD risk** for this patient. "
                        f"The population baseline (average across all training patients) is shown as "
                        f"**E[f(X)]** on the x-axis. Each feature bar shows how much that feature "
                        f"shifted the prediction away from this baseline — red bars increase risk, "
                        f"blue bars decrease it. Features are ranked top-to-bottom by impact magnitude."
                    )
                    _shap_waterfall(rf_exp, X_scaled, X_raw, FEATURES_24,
                                    "RF: Local SHAP waterfall", max_display=24,
                                    model_prob=_p_rf)
                    _shap_legend(is_patient=False)

                    st.markdown("---")

                    # ── XGB waterfall ──
                    st.markdown("#### XGBoost (XGB) — SHAP waterfall")
                    st.markdown(
                        f"The XGB base model predicts a **{_p_xgb*100:.1f}% CVD risk** for this patient. "
                        f"The meta-learner then combines the RF ({_p_rf*100:.1f}%) and XGB "
                        f"({_p_xgb*100:.1f}%) outputs to produce the final stacked prediction of "
                        f"**{final_prob*100:.1f}%**. Differences between RF and XGB waterfalls "
                        f"reflect each model's unique learned associations — both are informative."
                    )
                    _shap_waterfall(xgb_exp, X_scaled, X_raw, FEATURES_24,
                                    "XGB: Local SHAP waterfall", max_display=24,
                                    model_prob=_p_xgb)
                    _shap_legend(is_patient=False)

        st.info("💡 Open the **My Risk Over Time** tab to track changes across visits.")

    else:
        st.info("Fill in your health information above and click **Run Risk Prediction**.")


# ─────────────────────────────────────────
# TAB 2 – LONGITUDINAL TREND TRACKER
# ─────────────────────────────────────────
with tab_trend:
    if IS_PATIENT_MODE:
        st.subheader("My Heart Disease Risk Over Time")
        st.markdown(
            "Each time you run the Risk Calculator, your score is saved here automatically. "
            "Use this tab to track how your risk changes as you make lifestyle improvements or start new treatments."
        )
    else:
        st.subheader("Longitudinal Risk Tracker")
        st.markdown("Tracks CVD risk score and key clinical markers across multiple assessments in this session.")

    h = history_load()

    if not h:
        st.info(
            "No history yet. Run a prediction in the **Risk Calculator** tab first — "
            "each assessment is saved automatically to this timeline."
        )
    else:
        df_h = history_as_df()

        # ── Summary metrics ──
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.metric("Total assessments", len(df_h))
        with col_b:
            st.metric("Latest risk", f"{df_h['risk_pct'].iloc[-1]:.1f}%")
        with col_c:
            st.metric("Lowest risk ever", f"{df_h['risk_pct'].min():.1f}%")
        with col_d:
            delta = df_h["risk_pct"].iloc[-1] - df_h["risk_pct"].iloc[0]
            st.metric("Change since first visit", f"{delta:+.1f} pp",
                      delta_color="inverse")  # negative = good = green

        st.markdown("---")

        # ── Risk trend chart ──
        st.markdown("#### 10-Year CVD Risk Trend")
        chart_df = df_h[["label", "risk_pct"]].set_index("label")
        st.line_chart(chart_df, use_container_width=True)

        # ── Key clinical markers ──
        st.markdown("#### Key Clinical Marker History")
        markers_available = [c for c in ["sysbp", "totchol", "bmi", "glucose", "cigpday"] if c in df_h.columns]
        marker_labels = {
            "sysbp":   "Systolic BP (mmHg)" if IS_PATIENT_MODE else "SYSBP",
            "totchol": "Total Cholesterol (mg/dL)" if IS_PATIENT_MODE else "TOTCHOL",
            "bmi":     "BMI (kg/m²)",
            "glucose": "Blood Sugar (mg/dL)" if IS_PATIENT_MODE else "GLUCOSE",
            "cigpday": "Cigarettes/day" if IS_PATIENT_MODE else "CIGPDAY",
        }
        if markers_available:
            marker_choice = st.multiselect(
                "Show markers:",
                options=markers_available,
                default=markers_available[:3],
                format_func=lambda x: marker_labels.get(x, x)
            )
            if marker_choice:
                m_df = df_h[["label"] + marker_choice].set_index("label")
                m_df.columns = [marker_labels.get(c, c) for c in m_df.columns]
                st.line_chart(m_df, use_container_width=True)

        # ── Full history table ──
        st.markdown("#### Full Assessment History")
        display_cols = {
            "date": "Date", "label": "Visit label", "risk_pct": "10-yr risk (%)",
            "risk_cat": "Category", "sysbp": "Top BP", "totchol": "Cholesterol",
            "bmi": "BMI", "glucose": "Blood sugar", "cigpday": "Cigs/day"
        }
        show_cols = [c for c in display_cols if c in df_h.columns]
        view_df = df_h[show_cols].rename(columns=display_cols)
        st.dataframe(view_df, use_container_width=True)

        # ── Export + Clear ──
        col_exp, col_clr = st.columns([3, 1])
        with col_exp:
            csv_bytes = history_export_csv()
            st.download_button(
                "⬇️ Download my history as CSV",
                data=csv_bytes,
                file_name=f"cvd_risk_history_{datetime.date.today()}.csv",
                mime="text/csv",
            )
        with col_clr:
            if st.button("🗑️ Clear history", key="btn_clear_history"):
                history_clear()
                st.rerun()

        # ── Trend interpretation ──
        if len(df_h) >= 2:
            delta = df_h["risk_pct"].iloc[-1] - df_h["risk_pct"].iloc[0]
            st.markdown("---")
            if delta < -1.0:
                st.success(
                    f"📉 **Your risk has decreased by {abs(delta):.1f} percentage points** since your first assessment. "
                    "This is a positive trend — keep it up!"
                )
            elif delta > 1.0:
                st.warning(
                    f"📈 **Your risk has increased by {delta:.1f} percentage points** since your first assessment. "
                    "Consider discussing this trend with your doctor."
                )
            else:
                st.info("➡️ Your risk has remained stable across assessments.")

        st.caption(
            "⚠️ History is stored in your current browser session only. "
            "Download your history as CSV to save it permanently. "
            "Clearing your browser session or refreshing will reset the history."
        )


# ─────────────────────────────────────────
# TAB 3 – BIE (What-If Simulator)
# ─────────────────────────────────────────
with tab_bie:
    if IS_PATIENT_MODE:
        st.subheader("What-If Simulator")
        st.markdown(
            "Some 'what-if' changes may appear counterintuitive. This occurs because the model reflects patterns in historical data, including treatment effects and clinical complexity. "
            "These results do not imply that lowering blood pressure or cholesterol is harmful. It is not a guarantee and not medical advice. Always consult your doctor before making health changes."            
        )
        st.caption(
            "Use the sliders below to explore how specific lifestyle changes might affect your estimated risk. "
            "Each slider lets you adjust one health factor — your other values stay the same. "
            "The chart updates to show your new estimated risk after each change.")
    else:
        st.subheader("Behavioral Impact Engine (BIE)")
        st.markdown(
            "Interactive counterfactual simulator. Adjust individual clinical parameters to observe "
            "their modelled impact on the stacked CVD risk prediction. "
            "Results reflect learned associations in the Framingham training data — not causal treatment effects. "
            "In treated high-risk populations, directional changes may appear counterintuitive due to confounding."
        )

    if "v6_last_input_df" not in st.session_state:
        st.warning("Please run a prediction in **Risk Calculator** first.")
    else:
        df_patient    = st.session_state["v6_last_input_df"]
        used_threshold= st.session_state.get("v6_last_threshold", threshold)
        base_prob, _, _= stacking_predict_proba_24(df_patient, threshold=used_threshold)
        base_cat, base_color = interpret_risk(base_prob)

        # ── Baseline summary ──────────────────────────────────────────────────
        col_base1, col_base2, col_base3 = st.columns(3)
        with col_base1:
            st.metric("Baseline risk", f"{base_prob*100:.1f}%", help="Your current predicted 10-year CVD risk")
        with col_base2:
            st.metric("Risk category", f"{base_color} {base_cat}")
        with col_base3:
            st.metric("Alert threshold", f"{used_threshold*100:.0f}%",
                      delta="Flagged" if base_prob >= used_threshold else "Not flagged",
                      delta_color="inverse" if base_prob >= used_threshold else "normal")

        st.markdown("---")

        # ── Interactive sliders ───────────────────────────────────────────────
        if IS_PATIENT_MODE:
            st.markdown("#### Adjust your health factors")
            st.markdown("Move any slider to see how that change might affect your risk score.")
        else:
            st.markdown("#### Adjust clinical parameters")

        # Get current values
        cur = df_patient.iloc[0].to_dict()

        col_s1, col_s2 = st.columns(2)

        with col_s1:
            new_cigs = st.slider(
                "Cigarettes per day" if IS_PATIENT_MODE else "CIGPDAY",
                min_value=0, max_value=60,
                value=int(cur["CIGPDAY"]),
                step=1,
                help="Set to 0 to simulate quitting smoking",
                key="bie_cigs"
            )
            new_sysbp = st.slider(
                "Systolic blood pressure (top number)" if IS_PATIENT_MODE else "SYSBP (mmHg)",
                min_value=90, max_value=200,
                value=int(cur["SYSBP"]),
                step=1,
                help="Normal is below 120 mmHg",
                key="bie_sysbp"
            )
            new_bmi = st.slider(
                "Body weight (BMI)" if IS_PATIENT_MODE else "BMI (kg/m²)",
                min_value=15.0, max_value=50.0,
                value=float(round(cur["BMI"], 1)),
                step=0.5,
                help="Healthy range is 18.5–24.9",
                key="bie_bmi"
            )
            new_totchol = st.slider(
                "Total cholesterol (mg/dL)" if IS_PATIENT_MODE else "TOTCHOL (mg/dL)",
                min_value=100, max_value=400,
                value=int(cur["TOTCHOL"]),
                step=5,
                help="Desirable is below 200 mg/dL",
                key="bie_totchol"
            )

        with col_s2:
            new_glucose = st.slider(
                "Fasting blood sugar (mg/dL)" if IS_PATIENT_MODE else "GLUCOSE (mg/dL)",
                min_value=60, max_value=300,
                value=int(cur["GLUCOSE"]),
                step=1,
                help="Normal fasting glucose is 70–99 mg/dL",
                key="bie_glucose"
            )
            new_hdlc = st.slider(
                "HDL 'good' cholesterol (mg/dL)" if IS_PATIENT_MODE else "HDLC (mg/dL)",
                min_value=20, max_value=120,
                value=int(cur["HDLC"]),
                step=1,
                help="Higher is better. Goal ≥ 60 mg/dL is protective",
                key="bie_hdlc"
            )
            new_ldlc = st.slider(
                "LDL 'bad' cholesterol (mg/dL)" if IS_PATIENT_MODE else "LDLC (mg/dL)",
                min_value=40, max_value=250,
                value=int(cur["LDLC"]),
                step=5,
                help="Lower is better. Optimal is below 100 mg/dL",
                key="bie_ldlc"
            )
            new_hr = st.slider(
                "Resting heart rate (bpm)" if IS_PATIENT_MODE else "HEARTRTE (bpm)",
                min_value=40, max_value=140,
                value=int(cur["HEARTRTE"]),
                step=1,
                help="Normal resting heart rate is 60–100 bpm",
                key="bie_hr"
            )

        # ── Live prediction with slider values ────────────────────────────────
        df_modified = df_patient.copy()
        df_modified["CIGPDAY"]  = float(new_cigs)
        df_modified["SYSBP"]    = float(new_sysbp)
        df_modified["BMI"]      = float(new_bmi)
        df_modified["TOTCHOL"]  = float(new_totchol)
        df_modified["GLUCOSE"]  = float(new_glucose)
        df_modified["HDLC"]     = float(new_hdlc)
        df_modified["LDLC"]     = float(new_ldlc)
        df_modified["HEARTRTE"] = float(new_hr)

        new_prob, _, _ = stacking_predict_proba_24(df_modified, threshold=used_threshold)
        new_cat, new_color = interpret_risk(new_prob)
        delta_pp  = (new_prob - base_prob) * 100.0
        delta_rel = (new_prob - base_prob) / base_prob * 100.0 if base_prob > 0 else 0.0

        st.markdown("---")

        # ── Live result display ───────────────────────────────────────────────
        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        with col_r1:
            st.metric("Adjusted risk", f"{new_prob*100:.1f}%",
                      delta=f"{delta_pp:+.1f} pp",
                      delta_color="inverse")
        with col_r2:
            st.metric("New category", f"{new_color} {new_cat}")
        with col_r3:
            st.metric("Absolute change", f"{delta_pp:+.1f} pp",
                      delta_color="inverse")
        with col_r4:
            st.metric("Relative change", f"{delta_rel:+.1f}%",
                      delta_color="inverse")

        # ── Comparison bar chart ──────────────────────────────────────────────
        if abs(delta_pp) >= 0.1:
            fig_bie, ax_bie = plt.subplots(figsize=(7, 2.2))
            bars_data  = [base_prob * 100, new_prob * 100]
            bar_labels = ["Current profile", "With your changes"]
            bar_colors = ["#888780", "#D85A30" if new_prob > base_prob else "#1D9E75"]
            b = ax_bie.barh(bar_labels, bars_data, color=bar_colors,
                            height=0.45, edgecolor="none")
            for bar, val in zip(b, bars_data):
                ax_bie.text(bar.get_width() + 0.3,
                            bar.get_y() + bar.get_height() / 2,
                            f"{val:.1f}%", va="center", fontsize=11,
                            fontweight="500", color="#444441")
            ax_bie.set_xlim(0, max(bars_data) * 1.22)
            ax_bie.axvline(used_threshold * 100, color="#E24B4A",
                           linewidth=1, linestyle="--", alpha=0.7)
            ax_bie.text(used_threshold * 100 + 0.2, 1.42,
                        f"Alert threshold ({used_threshold*100:.0f}%)",
                        fontsize=8, color="#A32D2D", va="center")
            ax_bie.set_xlabel("10-year CVD risk (%)", fontsize=10)
            ax_bie.set_title(
                "Before vs after your adjustments" if IS_PATIENT_MODE
                else "Counterfactual risk comparison",
                fontsize=10, fontweight="bold", color="#0f4c75"
            )
            ax_bie.spines["top"].set_visible(False)
            ax_bie.spines["right"].set_visible(False)
            ax_bie.spines["left"].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig_bie, clear_figure=True)

            if IS_PATIENT_MODE:
                if delta_pp < -0.5:
                    st.success(
                        f"These changes could lower your estimated risk by **{abs(delta_pp):.1f} percentage points** "
                        f"— from {base_prob*100:.1f}% down to {new_prob*100:.1f}%. "
                        "Discuss these targets with your doctor to make a personalised plan."
                    )
                elif delta_pp > 0.5:
                    st.warning(
                        f"These slider values show a **higher** risk ({new_prob*100:.1f}%). "
                        "Try adjusting toward healthier values — reducing blood pressure, "
                        "cholesterol, smoking, or weight."
                    )
                else:
                    st.info("These changes have a minimal effect on the model's estimate for this profile.")

        # ── Scenario sweep table ──────────────────────────────────────────────
        st.markdown("---")
        if IS_PATIENT_MODE:
            st.markdown("#### Individual impact of each change")
            st.markdown(
                "The table below tests each change **one at a time** from your current baseline, "
                "so you can see which single change would help the most."
            )
        else:
            st.markdown("#### Individual counterfactual scenarios (single-feature perturbation)")

        # Build comprehensive scenario list from current patient values
        scenarios_custom = []
        if float(cur["CIGPDAY"]) > 0:
            d = df_patient.copy(); d["CIGPDAY"] = 0.0
            scenarios_custom.append(("Quit smoking completely", f"Cigs/day: {int(cur['CIGPDAY'])} → 0", d))
        if float(cur["SYSBP"]) > 120:
            d = df_patient.copy(); d["SYSBP"] = max(float(cur["SYSBP"]) - 10, 90.0)
            scenarios_custom.append(("Lower blood pressure by 10 mmHg",
                                     f"Sys BP: {int(cur['SYSBP'])} → {int(max(float(cur['SYSBP'])-10,90))}", d))
            d2 = df_patient.copy(); d2["SYSBP"] = max(float(cur["SYSBP"]) - 20, 90.0)
            scenarios_custom.append(("Lower blood pressure by 20 mmHg",
                                     f"Sys BP: {int(cur['SYSBP'])} → {int(max(float(cur['SYSBP'])-20,90))}", d2))
        if float(cur["BMI"]) > 24.9:
            d = df_patient.copy(); d["BMI"] = max(float(cur["BMI"]) - 2.0, 18.5)
            scenarios_custom.append(("Lose weight (BMI − 2)",
                                     f"BMI: {cur['BMI']:.1f} → {max(float(cur['BMI'])-2,18.5):.1f}", d))
            d2 = df_patient.copy(); d2["BMI"] = max(float(cur["BMI"]) - 5.0, 18.5)
            scenarios_custom.append(("Lose more weight (BMI − 5)",
                                     f"BMI: {cur['BMI']:.1f} → {max(float(cur['BMI'])-5,18.5):.1f}", d2))
        if float(cur["TOTCHOL"]) > 180:
            d = df_patient.copy(); d["TOTCHOL"] = max(float(cur["TOTCHOL"]) - 20, 100.0)
            scenarios_custom.append(("Lower total cholesterol by 20 mg/dL",
                                     f"Chol: {int(cur['TOTCHOL'])} → {int(max(float(cur['TOTCHOL'])-20,100))}", d))
        if float(cur["LDLC"]) > 100:
            d = df_patient.copy(); d["LDLC"] = max(float(cur["LDLC"]) - 30, 40.0)
            scenarios_custom.append(("Lower LDL cholesterol by 30 mg/dL",
                                     f"LDL: {int(cur['LDLC'])} → {int(max(float(cur['LDLC'])-30,40))}", d))
        if float(cur["HDLC"]) < 60:
            d = df_patient.copy(); d["HDLC"] = min(float(cur["HDLC"]) + 10, 120.0)
            scenarios_custom.append(("Raise HDL good cholesterol by 10",
                                     f"HDL: {int(cur['HDLC'])} → {int(min(float(cur['HDLC'])+10,120))}", d))
        if float(cur["GLUCOSE"]) > 99:
            d = df_patient.copy(); d["GLUCOSE"] = max(float(cur["GLUCOSE"]) - 15, 70.0)
            scenarios_custom.append(("Lower blood sugar by 15 mg/dL",
                                     f"Glucose: {int(cur['GLUCOSE'])} → {int(max(float(cur['GLUCOSE'])-15,70))}", d))
        if float(cur["HEARTRTE"]) > 75:
            d = df_patient.copy(); d["HEARTRTE"] = max(float(cur["HEARTRTE"]) - 10, 55.0)
            scenarios_custom.append(("Lower resting heart rate by 10 bpm",
                                     f"HR: {int(cur['HEARTRTE'])} → {int(max(float(cur['HEARTRTE'])-10,55))}", d))
        # Combined best-case scenario
        d_all = df_patient.copy()
        if float(cur["CIGPDAY"]) > 0:        d_all["CIGPDAY"]  = 0.0
        if float(cur["SYSBP"]) > 120:        d_all["SYSBP"]    = max(float(cur["SYSBP"]) - 15, 90.0)
        if float(cur["BMI"]) > 24.9:         d_all["BMI"]      = max(float(cur["BMI"]) - 3.0, 18.5)
        if float(cur["TOTCHOL"]) > 180:      d_all["TOTCHOL"]  = max(float(cur["TOTCHOL"]) - 20, 100.0)
        if float(cur["GLUCOSE"]) > 99:       d_all["GLUCOSE"]  = max(float(cur["GLUCOSE"]) - 10, 70.0)
        scenarios_custom.append(("Combined lifestyle improvements", "All modifiable factors improved", d_all))

        if scenarios_custom:
            sweep_rows = []
            for name, desc, dfx in scenarios_custom:
                p, _, _ = stacking_predict_proba_24(dfx, threshold=used_threshold)
                drop_pp  = (base_prob - p) * 100.0
                drop_rel = (base_prob - p) / base_prob * 100.0 if base_prob > 0 else 0.0
                new_c, new_ic = interpret_risk(p)
                sweep_rows.append({
                    "Scenario":          name,
                    "Description":       desc,
                    "New risk (%)":      round(p * 100, 1),
                    "Risk reduction (pp)": round(drop_pp, 1),
                    "Reduction (%)":     round(drop_rel, 1),
                    "New category":      f"{new_ic} {new_c}",
                })
            sweep_df = pd.DataFrame(sweep_rows).sort_values("Risk reduction (pp)", ascending=False)

            # Highlight best row
            def highlight_best(row):
                if row.name == sweep_df.index[0]:
                    return ["background-color: #EAF3DE"] * len(row)
                return [""] * len(row)

            fmt_sweep = {
                "New risk (%)":         "{:.1f}",
                "Risk reduction (pp)":  "{:+.1f}",
                "Reduction (%)":        "{:+.1f}",
            }
            st.dataframe(
                sweep_df.style.format(fmt_sweep).apply(highlight_best, axis=1),
                use_container_width=True, hide_index=True
            )

            # Scenario impact bar chart
            top_scenarios = sweep_df[sweep_df["Risk reduction (pp)"] > 0].head(8)
            if not top_scenarios.empty:
                fig_sc, ax_sc = plt.subplots(figsize=(8, max(3, len(top_scenarios) * 0.55)))
                sc_labels = [s[:45] + "…" if len(s) > 45 else s
                             for s in top_scenarios["Scenario"].tolist()]
                sc_vals   = top_scenarios["Risk reduction (pp)"].tolist()
                sc_colors = ["#1D9E75" if v > 0 else "#E24B4A" for v in sc_vals]
                ax_sc.barh(range(len(sc_labels))[::-1], sc_vals,
                           color=sc_colors, height=0.55, edgecolor="none")
                for i, (val, lbl) in enumerate(zip(sc_vals, sc_labels)):
                    ax_sc.text(val + 0.05, len(sc_labels) - 1 - i,
                               f"−{val:.1f} pp", va="center", fontsize=9.5,
                               color="#27500A", fontweight="500")
                ax_sc.set_yticks(range(len(sc_labels))[::-1])
                ax_sc.set_yticklabels(sc_labels, fontsize=9.5)
                ax_sc.set_xlabel("Risk reduction (percentage points)", fontsize=10)
                ax_sc.set_title(
                    "Estimated risk reduction per scenario" if IS_PATIENT_MODE
                    else "Single-feature counterfactual impact (pp reduction)",
                    fontsize=10, fontweight="bold", color="#0f4c75"
                )
                ax_sc.axvline(0, color="#d3d1c7", linewidth=0.8)
                ax_sc.spines["top"].set_visible(False)
                ax_sc.spines["right"].set_visible(False)
                ax_sc.spines["left"].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig_sc, clear_figure=True)

            # Best scenario callout
            best_row = sweep_df.iloc[0]
            if best_row["Risk reduction (pp)"] > 0:
                if IS_PATIENT_MODE:
                    _bname = best_row["Scenario"]
                    _bnew  = best_row["New risk (%)"]
                    _bdrop = best_row["Risk reduction (pp)"]
                    st.success(
                        f"**Most impactful single change: {_bname}**\n\n"
                        f"This could lower your estimated risk from **{base_prob*100:.1f}%** "
                        f"to **{_bnew:.1f}%** "
                        f"(a reduction of **{_bdrop:.1f} percentage points**).\n\n"
                        "Discuss this with your doctor to create a personalised action plan."
                    )
                else:
                    _bname = best_row["Scenario"]
                    _bnew  = best_row["New risk (%)"]
                    _bdrop = best_row["Risk reduction (pp)"]
                    _brel  = best_row["Reduction (%)"]
                    st.info(
                        f"**Highest-impact lever:** {_bname} — "
                        f"predicted drop from {base_prob*100:.1f}% to {_bnew:.1f}% "
                        f"(−{_bdrop:.1f} pp, −{_brel:.1f}% relative). "
                        "Highlighted in green in the table above."
                    )

        # ── Evidence-based recommendations ───────────────────────────────────
        st.markdown("---")
        st.markdown("#### Evidence-based recommendations" if IS_PATIENT_MODE
                    else "#### Clinical evidence summary")
        col_ev1, col_ev2 = st.columns(2)
        with col_ev1:
            st.markdown(
                "**Quit smoking** — The single highest-impact modifiable risk factor. Risk reduction begins within weeks and continues for years.\n\n"
                "**Lower blood pressure** — Each 10 mmHg reduction in systolic BP reduces CVD risk by ~10-20%. Lifestyle and medications both help.\n\n"
                "**Control blood sugar** — Keeping fasting glucose in the normal range significantly reduces long-term cardiovascular risk."
            )
        with col_ev2:
            st.markdown(
                "**Improve cholesterol profile** — Lowering LDL and raising HDL both reduce cardiovascular risk. Statins, diet, and exercise all help.\n\n"
                "**Healthy weight** — Even modest weight loss (5-10% of body weight) improves blood pressure, glucose, and cholesterol simultaneously.\n\n"
                "**Physical activity** — 150 minutes/week of moderate aerobic activity reduces CVD risk by 20-30%. It also improves blood pressure and mental health."
            )
        st.caption(
            "Sources: ACC/AHA Cardiovascular Risk Guidelines, Framingham Heart Study, "
            "WHO Global Action Plan for NCDs. These recommendations are general — "
            "always discuss your personal plan with a licensed clinician."
        )


# ─────────────────────────────────────────
# TAB 4 – AI HEALTH COACH (Claude API)
# ─────────────────────────────────────────
with tab_coach:
    if IS_PATIENT_MODE:
        st.subheader("💬 Your Heart Health Assistant")
        st.markdown(
            "Ask any question about your heart health, your risk score, what your lab values mean, "
            "or what lifestyle changes might help. This assistant uses AI to provide evidence-based information."
        )
    else:
        st.subheader("💬 Clinical AI Assistant")
        st.markdown(
            "Ask clinical questions about CVD risk factors, interpretation of this model's output, "
            "guidelines, or patient communication strategies."
        )

    st.info(
        "⚠️ This assistant provides **general health information only**. "
        "It is not a substitute for medical advice from a licensed clinician. "
        "Always consult your doctor for diagnosis and treatment decisions."
    )

    # Initialise chat history
    if "coach_messages" not in st.session_state:
        st.session_state["coach_messages"] = []

    # Build patient context string if a prediction has been run
    patient_ctx = ""
    if "v6_last_prob" in st.session_state:
        prob = st.session_state["v6_last_prob"]
        cat, _ = interpret_risk(prob)
        patient_ctx = (
            f"\n\nThe user's most recent CVD risk assessment: "
            f"10-year risk = {prob*100:.1f}% ({cat}). "
            f"They are using a Framingham-based stacked ML model (RF + XGB + Logistic Regression meta-learner). "
            f"Please reference this context when relevant."
        )

    # System prompt
    SYSTEM_PROMPT = (
        "You are a helpful, empathetic, and accurate heart health assistant. "
        "You provide evidence-based information about cardiovascular disease risk, "
        "heart-healthy lifestyle changes, common lab values (cholesterol, blood pressure, glucose, BMI), "
        "and general cardiology concepts. "
        "You ALWAYS remind users that your answers are educational and not a substitute for medical advice. "
        "You NEVER diagnose, prescribe, or recommend specific medications. "
        "You speak clearly and accessibly, avoiding jargon unless the user is clearly a clinician. "
        "If a user seems to be in distress or describes symptoms of a heart attack or stroke, "
        "you immediately advise them to call emergency services (911)."
        + patient_ctx
    )

    # Render chat history
    for msg in st.session_state["coach_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Suggested starter questions
    if not st.session_state["coach_messages"]:
        st.markdown("**Try asking:**")
        starter_cols = st.columns(2)
        starters = [
            "What does my cholesterol number mean?",
            "How can I lower my blood pressure naturally?",
            "What foods are good for heart health?",
            "What is the difference between HDL and LDL?",
            "How much exercise is recommended for heart health?",
            "What does my risk score actually mean?",
        ]
        for i, q in enumerate(starters):
            with starter_cols[i % 2]:
                if st.button(q, key=f"starter_{i}"):
                    st.session_state["coach_messages"].append({"role": "user", "content": q})
                    st.rerun()

    # Chat input
    if user_input := st.chat_input("Ask your heart health question…"):
        st.session_state["coach_messages"].append({"role": "user", "content": user_input})
        st.rerun()

    # Generate response if last message is from user
    msgs = st.session_state["coach_messages"]
    if msgs and msgs[-1]["role"] == "user":
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    import urllib.request

                    # Retrieve API key from Streamlit secrets
                    # Add to Streamlit Cloud: Settings → Secrets → ANTHROPIC_API_KEY = "sk-ant-..."
                    try:
                        _api_key = st.secrets["ANTHROPIC_API_KEY"]
                    except Exception:
                        # Fallback: try common alternative secret names
                        try:
                            _api_key = st.secrets["anthropic_api_key"]
                        except Exception:
                            _api_key = None

                    if not _api_key:
                        reply = (
                            "The Health Coach is not configured yet. "
                            "Please add your Anthropic API key to Streamlit secrets:\n\n"
                            "1. Go to your Streamlit Cloud dashboard\n"
                            "2. Click **Settings** → **Secrets**\n"
                            "3. Add: `ANTHROPIC_API_KEY = \"sk-ant-your-key-here\"`\n\n"
                            "Once added, redeploy the app and the Health Coach will be active."
                        )
                    else:
                        api_payload = json.dumps({
                            "model": "claude-sonnet-4-20250514",
                            "max_tokens": 1000,
                            "system": SYSTEM_PROMPT,
                            "messages": [
                                {"role": m["role"], "content": m["content"]}
                                for m in msgs
                            ]
                        }).encode("utf-8")
                        req = urllib.request.Request(
                            "https://api.anthropic.com/v1/messages",
                            data=api_payload,
                            headers={
                                "Content-Type": "application/json",
                                "x-api-key": _api_key,
                                "anthropic-version": "2023-06-01",
                            },
                            method="POST"
                        )
                        with urllib.request.urlopen(req, timeout=30) as resp:
                            result = json.loads(resp.read().decode("utf-8"))
                        reply = result["content"][0]["text"]
                except Exception as e:
                    reply = (
                        "I'm sorry, I couldn't connect to the AI assistant right now. "
                        "Please try again shortly, or consult a healthcare professional for your question.\n\n"
                        f"*(Technical detail: {str(e)[:120]})*"
                    )
                st.markdown(reply)
        st.session_state["coach_messages"].append({"role": "assistant", "content": reply})

    # Clear chat
    if msgs:
        if st.button("🗑️ Clear conversation", key="btn_clear_chat"):
            st.session_state["coach_messages"] = []
            st.rerun()


# ─────────────────────────────────────────
# TAB 5 – MODEL & DATA (unchanged from v5)
# ─────────────────────────────────────────
with tab_model:
    st.subheader("Model & Data")
    st.markdown(
        """
        **MODEL & DATA OVERVIEW**

        This application implements a stacked machine-learning model to estimate 10-year cardiovascular disease (CVD) risk,
        using an expanded clinical and prior-history feature set.

        **Data Source**
        - Primary source: Framingham Heart Study–based dataset
        - Target variable: 10-year cardiovascular disease (CVD) outcome
        - Prediction task: Binary classification → probability of a CVD event within 10 years

        **Feature Set (24 features — Clinical + History)**

        *Demographics:* AGE, SEX, educ (education level)

        *Blood Pressure:* SYSBP, DIABP, HYPERTEN, PREVHYP

        *Lipids:* TOTCHOL, HDLC, LDLC

        *Metabolic:* BMI, GLUCOSE, DIABETES

        *Lifestyle:* CIGPDAY (cigarettes/day)

        *Cardiac / Vital Signs:* HEARTRTE (heart rate), BPMEDS (BP medication)

        *Prior Cardiovascular History:* ANGINA, MI_FCHD, PREVAP, PREVCHD, PREVMI, HOSPMI, STROKE, PREVSTRK

        **Model Architecture (Stacking Ensemble)**
        - Base learners: RandomForestClassifier + XGBoostClassifier
        - Meta-learner: Logistic Regression on [p_RF, p_XGB]
        - Output: Probability of 10-year CVD event

        **Interpretability**
        - SHAP local explanations (per-patient feature contributions)
        - Behavioral Impact Engine (BIE) — counterfactual what-if scenarios

        **v6.0 additions**
        - Longitudinal risk tracker (session-based)
        - Plain-language patient mode with field tooltips
        - Tiered escalation guidance (Low → High risk)
        - AI Health Coach tab (Claude API)
        - Visit labels and CSV export
        """
    )
    st.markdown("**Artifacts:** scaler_24.pkl, rf_clin24.pkl, xgb_clin24.pkl, stack_meta_clin24.pkl, features_24.json")


# ─────────────────────────────────────────
# TAB 6 – FAQ & NOTES
# ─────────────────────────────────────────
with tab_faq:
    st.subheader("FAQ & Notes")
    st.markdown(
        """
        **Why can some changes look counterintuitive (e.g., BMI/Cholesterol)?**
        The model learns statistical patterns from observational data. Some relationships appear reversed
        due to confounding or treatment effects (e.g., sicker patients receiving therapy).
        The BIE is a model-based what-if, not a causal treatment estimate.

        **This is not medical advice.**
        Always consult a licensed clinician for diagnosis and treatment decisions.

        **What does "10-year CVD risk" mean?**
        The estimated probability that someone with your health profile may experience a cardiovascular event
        (heart attack, stroke, or coronary disease) within the next 10 years.
        It is a population-based estimate, not a guarantee for any individual.

        **How should I interpret the percentage?**
        A predicted risk of 15% means that among 100 people with similar profiles,
        approximately 15 may experience a CVD event within 10 years.

        **Why is my history only saved for this session?**
        The longitudinal tracker stores data in your browser session to protect your privacy.
        Use the CSV download button to keep a permanent record.

        **Is this a clinical decision support tool?**
        No. This app is for research and educational purposes only.
        It must not be used as a standalone diagnostic or treatment decision system.

        **How is the AI Health Coach powered?**
        The Health Coach uses the Claude AI model (Anthropic) to answer heart health questions.
        All responses are educational. The coach cannot diagnose or prescribe.

        **Important Limitations**
        - The model reflects patterns from the Framingham training population
        - Performance may vary across populations and healthcare settings
        - Predictions depend on feature accuracy and population similarity
        - Outputs must not replace professional medical judgment

        **Version history**
        - v3.0: Prevention-focused, minimal history
        - v4.0: Added limited clinical history
        - v5.0: Expanded prior CVD history + RF/XGB stacking ensemble
        - v6.0: Longitudinal tracking, plain-language patient mode, escalation guidance, AI Health Coach
        """
    )


# =========================
# FOOTER
# =========================
st.markdown(
    """
    <hr style="margin-top:32px;margin-bottom:8px;">
    <div style="text-align:center;font-size:12px;color:gray;">
      Stacking Generative AI CVD Risk Model v6.0 • 24 features (clinical+history) • Research &amp; Demonstration Only<br>
      This application does not provide medical advice, diagnosis, or treatment.<br>
      © 2025 Howard Nguyen, PhD. For demonstration only — not for clinical decision-making.
    </div>
    """,
    unsafe_allow_html=True
)
