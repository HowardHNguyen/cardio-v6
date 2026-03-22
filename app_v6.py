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
    [data-testid="stToolbar"] { visibility: hidden; height: 0%; position: fixed; }
    footer { visibility: hidden; height: 0%; }
    #MainMenu { visibility: hidden; }
    header { visibility: hidden; height: 0%; }
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
    return scaler, rf_model, xgb_model, meta_model, features_24

scaler, rf_model, xgb_model, meta_model, FEATURES_24 = load_artifacts()

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

# SHAP-friendly plain text for patient display
SHAP_FRIENDLY = {
    "SYSBP":    "High systolic blood pressure (top number)",
    "DIABP":    "High diastolic blood pressure (bottom number)",
    "CIGPDAY":  "Smoking (cigarettes per day)",
    "AGE":      "Age",
    "SEX":      "Male sex",
    "GLUCOSE":  "Blood sugar levels",
    "DIABETES": "Diabetes",
    "BPMEDS":   "Blood pressure medication use",
    "PREVMI":   "Prior heart attack",
    "PREVCHD":  "Prior heart disease",
    "PREVSTRK": "Prior stroke",
    "PREVHYP":  "History of hypertension",
    "TOTCHOL":  "Total cholesterol",
    "BMI":      "Body weight (BMI)",
    "HDLC":     "HDL good cholesterol",
    "LDLC":     "LDL bad cholesterol",
    "HEARTRTE": "Heart rate",
    "ANGINA":   "Chest pain (angina)",
    "HOSPMI":   "Hospitalization for heart attack",
    "MI_FCHD":  "Family history of heart attack",
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
        # RF outputs probability natively
        st.session_state["rf_explainer"] = shap.TreeExplainer(
            rf_model, model_output="probability"
        )
        # XGB internal output is log-odds by default; force probability so
        # f(x) and E[f(X)] render as probabilities matching the stacking model
        try:
            st.session_state["xgb_explainer"] = shap.TreeExplainer(
                xgb_model, model_output="probability"
            )
        except Exception:
            # Fallback: some XGB versions do not support probability output
            st.session_state["xgb_explainer"] = shap.TreeExplainer(xgb_model)
    return st.session_state["rf_explainer"], st.session_state["xgb_explainer"]

def _shap_waterfall(explainer, X_row_1xF, X_row_raw, feature_names,
                    title: str, max_display: int = 12):
    """
    Render a SHAP waterfall plot for a single patient row.
    Shows baseline expected value, each feature's directional push
    (red = raises risk, blue = lowers risk), and the final prediction.

    Parameters
    ----------
    explainer    : shap.TreeExplainer fitted to the model
    X_row_1xF   : scaled feature array, shape (1, n_features)
    X_row_raw   : unscaled feature array, shape (1, n_features) — used for
                  feature value annotations on the y-axis
    feature_names: list of feature name strings (length == n_features)
    title        : string shown above the plot
    max_display  : how many features to show (sorted by |SHAP value|)
    """
    if not SHAP_AVAILABLE or explainer is None:
        st.info("SHAP not available in this deployment.")
        return

    # ── 1. Compute raw SHAP values ──────────────────────────────────────────
    sv_raw = explainer.shap_values(X_row_1xF)

    # Normalise to class-1 probability (binary classifiers return a list)
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

    shap_vals_1d = np.ravel(sv_raw[0])          # shape (n_features,)
    raw_vals_1d  = np.ravel(X_row_raw[0])        # shape (n_features,) — actual patient values

    if len(feature_names) != len(shap_vals_1d):
        st.warning("Feature name / SHAP value length mismatch. Skipping waterfall.")
        return

    # ── 2. Extract base value safely (RF list vs XGB scalar/array) ─────────
    ev = explainer.expected_value
    if isinstance(ev, (list, np.ndarray)):
        base_val = float(ev[1]) if len(ev) > 1 else float(ev[0])
    else:
        base_val = float(ev)

    # If XGB returned log-odds (outside [0,1]), convert to probability space
    if not (0.0 <= base_val <= 1.0):
        base_val = float(1.0 / (1.0 + np.exp(-base_val)))
        p0 = base_val * (1.0 - base_val)   # d(sigmoid)/d(log-odds) at base
        shap_vals_1d = shap_vals_1d * p0    # approximate probability-space values

    explanation = shap.Explanation(
        values        = shap_vals_1d,
        base_values   = base_val,
        data          = raw_vals_1d,
        feature_names = feature_names,
    )

    # ── 3. Plot ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, max(5, max_display * 0.45)))
    shap.plots.waterfall(explanation, max_display=max_display, show=False)
    fig = plt.gcf()
    fig.suptitle(title, fontsize=11, fontweight="bold",
                 color="#0f4c75", y=1.02)
    plt.tight_layout()
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
            st.caption(
                "The chart below shows which of your health values pushed your risk "
                "score up (red) or down (blue) compared to the average patient. "
                "These are model-based associations — not diagnoses."
            )
            if SHAP_AVAILABLE:
                _, xgb_exp = _get_shap_explainers()
                X_raw_pt    = df_input.values.astype(float)
                X_scaled_pt = scaler.transform(X_raw_pt)

                # Use friendly names for patient-facing waterfall
                friendly_names = [
                    SHAP_FRIENDLY.get(f, f) for f in FEATURES_24
                ]
                _shap_waterfall(
                    xgb_exp, X_scaled_pt, X_raw_pt,
                    friendly_names,
                    "Your personal risk drivers (XGBoost model)",
                    max_display=10,
                )
                st.caption(
                    "E[f(X)] = average risk across all patients in the training dataset. "
                    "f(x) = your individual predicted risk. "
                    "Each bar shows how much one factor moved your score."
                )
            else:
                st.info("Install `shap` and `matplotlib` to enable the waterfall explanation.")

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
                st.caption("SHAP shows which features pushed RF and XGB predictions up or down for this patient.")
                if not SHAP_AVAILABLE:
                    st.info("Add shap + matplotlib to requirements.txt to enable.")
                else:
                    rf_exp, xgb_exp = _get_shap_explainers()
                    X_raw    = df_input.values.astype(float)
                    X_scaled = scaler.transform(X_raw)
                    st.markdown("**Random Forest — SHAP waterfall**")
                    st.caption(
                        "Red bars push the predicted risk higher than the baseline. "
                        "Blue bars push it lower. Bar width = magnitude of impact. "
                        "E[f(X)] is the model's average prediction across all training patients."
                    )
                    _shap_waterfall(rf_exp, X_scaled, X_raw, FEATURES_24,
                                    "RF: Local SHAP waterfall", max_display=12)
                    st.markdown("---")
                    st.markdown("**XGBoost — SHAP waterfall**")
                    st.caption(
                        "Same interpretation as above for the XGBoost base learner."
                    )
                    _shap_waterfall(xgb_exp, X_scaled, X_raw, FEATURES_24,
                                    "XGB: Local SHAP waterfall", max_display=12)

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
    st.subheader("What-If Simulator" if IS_PATIENT_MODE else "Behavioral Impact Engine (BIE)")

    if IS_PATIENT_MODE:
        st.markdown(
            "This tool shows how changes in your health — like quitting smoking, lowering your blood pressure, "
            "or losing weight — might change your estimated risk score. "
            "**Note:** These are model estimates, not guarantees. Always consult your doctor before making changes."
        )
    else:
        st.markdown(
            "The **BIE** performs local counterfactual re-scoring. In high-risk treated populations, "
            "lowering measured BP or cholesterol may move the profile toward clusters associated with advanced disease. "
            "These effects reflect learned associations, not causal treatment effects."
        )

    st.markdown("---")

    if "v6_last_input_df" not in st.session_state:
        st.warning("Please run a prediction in **Risk Calculator** first.")
    else:
        df_patient = st.session_state["v6_last_input_df"]
        used_threshold = st.session_state.get("v6_last_threshold", threshold)

        if st.button("▶️ Run What-If Analysis", key="btn_run_bie"):
            base_prob, scenario_df, best = bie_scenarios_24(
                df_patient, threshold=used_threshold, include_advanced=not IS_PATIENT_MODE
            )
            category, color = interpret_risk(base_prob)

            st.markdown(f"**Baseline risk:** {base_prob*100:.1f}%  {color} {category}")
            st.markdown("#### Scenarios")

            fmt = {"Risk (%)": "{:.2f}", "Change (pp)": "{:+.2f}", "Relative change (%)": "{:+.1f}"}
            st.dataframe(scenario_df.style.format(fmt), use_container_width=True)

            if best:
                if IS_PATIENT_MODE:
                    st.success(
                        f"💡 **Most impactful change for you: {best['name']}**\n\n"
                        f"This change could lower your estimated risk from **{base_prob*100:.1f}%** "
                        f"to **{best['p']*100:.1f}%**. Discuss this with your doctor."
                    )
                else:
                    abs_pp = best["drop"] * 100.0
                    rel_pct = (best["drop"] / base_prob * 100.0) if base_prob > 0 else 0.0
                    st.markdown(
                        f"**Most impactful lever:** {best['name']} → "
                        f"{best['p']*100:.2f}% (−{abs_pp:.2f} pp, −{rel_pct:.1f}% relative)"
                    )

            st.markdown("#### Evidence-based recommendations")
            st.markdown(
                """
                **Smoking** — If you smoke, quitting is the single highest-impact change you can make.  
                **Blood pressure** — If elevated, lifestyle changes and clinician-guided management can help.  
                **Metabolic** — Managing blood sugar and weight reduces long-term cardiovascular risk.  
                **Lifestyle** — Regular physical activity, a heart-healthy diet, quality sleep, and stress management all contribute.
                """
            )
        else:
            st.info("Click **Run What-If Analysis** to see how lifestyle changes could affect your risk.")


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
