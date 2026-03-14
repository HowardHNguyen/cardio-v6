import numpy as np
import pandas as pd
import json
import joblib
import streamlit as st

import streamlit as st

st.markdown(
    """
    <style>
    /* Hide top-right Streamlit menu (Share, Fork, etc.) */
    [data-testid="stToolbar"] {
        visibility: hidden;
        height: 0%;
        position: fixed;
    }

    /* Hide bottom-right "Hosted with Streamlit" badge */
    footer {
        visibility: hidden;
        height: 0%;
    }

    /* Hide hamburger menu (⋮) */
    #MainMenu {
        visibility: hidden;
    }

    /* Optional: remove extra padding caused by toolbar */
    header {
        visibility: hidden;
        height: 0%;
    }
    </style>
    """,
    unsafe_allow_html=True
)

from pathlib import Path

# Optional SHAP imports (app still runs if not installed)
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
    page_title="CVD Risk – Stacking GenAI v5.0 (24 Features Clinical+History)",
    page_icon="🫀",
    layout="wide"
)

# =========================
# 2) LOAD ARTIFACTS (24-feature clinical+history)
# =========================
DEFAULT_THRESHOLD = 0.40

ARTIFACTS = {
    "scaler": Path("scaler_24.pkl"),
    "rf": Path("rf_clin24.pkl"),
    "xgb": Path("xgb_clin24.pkl"),
    "meta": Path("stack_meta_clin24.pkl"),
    "features": Path("features_24.json"),
}

@st.cache_resource
def load_artifacts():
    missing = [k for k, p in ARTIFACTS.items() if not p.exists()]
    if missing:
        st.error(
            "Missing required model files:\n\n"
            + "\n".join([f"- {k}: {ARTIFACTS[k]}" for k in missing])
            + "\n\nUpload these into the same repo folder as app.py."
        )
        st.stop()

    scaler = joblib.load(ARTIFACTS["scaler"])
    rf_model = joblib.load(ARTIFACTS["rf"])
    xgb_model = joblib.load(ARTIFACTS["xgb"])
    meta_model = joblib.load(ARTIFACTS["meta"])

    with open(ARTIFACTS["features"], "r") as f:
        features_24 = json.load(f)

    # Safety checks
    if not isinstance(features_24, list) or len(features_24) != 24:
        st.error("features_24.json must be a JSON list of exactly 24 feature names.")
        st.stop()

    return scaler, rf_model, xgb_model, meta_model, features_24

scaler, rf_model, xgb_model, meta_model, FEATURES_24 = load_artifacts()


# =========================
# 3) HELPERS
# =========================
def interpret_risk(prob: float):
    if prob < 0.05:
        return "Low risk", "🟢"
    if prob < 0.10:
        return "Borderline risk", "🟡"
    if prob < 0.20:
        return "Intermediate risk", "🟠"
    return "High risk", "🔴"

def _as_int_yesno(v: str) -> int:
    return 1 if v == "Yes" else 0

def build_input_df_24():
    """
    Render UI for 24 features and return a single-row DataFrame.
    IMPORTANT: final ordering MUST match FEATURES_24 (training order).
    """
    st.subheader("Patient Profile, Clinical Risk Factors & Prior History (24 Features)")

    col1, col2, col3 = st.columns(3)

    with col1:
        sex = st.selectbox("Sex", ["Male", "Female"], index=0, key="calc_sex")
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=55, step=1, key="calc_age")
        educ = st.selectbox("Education (1–4)", ["1 – Some HS", "2 – HS Grad", "3 – Some College", "4 – College Grad"], index=2, key="calc_educ")
        cigs = st.number_input("Cigarettes per day", min_value=0, max_value=80, value=0, step=1, key="calc_cigs")

    with col2:
        sysbp = st.number_input("SYSBP (mmHg)", min_value=80, max_value=250, value=130, step=1, key="calc_sysbp")
        diabp = st.number_input("DIABP (mmHg)", min_value=40, max_value=160, value=80, step=1, key="calc_diabp")
        bmi = st.number_input("BMI (kg/m²)", min_value=15.0, max_value=60.0, value=27.0, step=0.1, key="calc_bmi")
        heartrate = st.number_input("Heart Rate (bpm)", min_value=30, max_value=200, value=70, step=1, key="calc_hr")

    with col3:
        totchol = st.number_input("Total Cholesterol (mg/dL)", min_value=80, max_value=500, value=200, step=1, key="calc_totchol")
        hdlc = st.number_input("HDL-C (mg/dL)", min_value=10, max_value=150, value=45, step=1, key="calc_hdlc")
        ldlc = st.number_input("LDL-C (mg/dL)", min_value=10, max_value=300, value=120, step=1, key="calc_ldlc")
        glucose = st.number_input("Glucose (mg/dL)", min_value=40, max_value=400, value=90, step=1, key="calc_glucose")

    st.markdown("#### Clinical conditions & history")
    col4, col5, col6 = st.columns(3)

    with col4:
        diabetes = st.selectbox("Diabetes", ["No", "Yes"], index=0, key="calc_diabetes")
        bpmeds = st.selectbox("BP meds", ["No", "Yes"], index=0, key="calc_bpmeds")
        prevhyp = st.selectbox("PREVHYP (prior HTN)", ["No", "Yes"], index=0, key="calc_prevhyp")

    with col5:
        prevchd = st.selectbox("PREVCHD", ["No", "Yes"], index=0, key="calc_prevchd")
        prevap = st.selectbox("PREVAP", ["No", "Yes"], index=0, key="calc_prevap")
        prevmi = st.selectbox("PREVMI", ["No", "Yes"], index=0, key="calc_prevmi")
        hospmi = st.selectbox("HOSPMI", ["No", "Yes"], index=0, key="calc_hospmi")

    with col6:
        prevstrk = st.selectbox("PREVSTRK", ["No", "Yes"], index=0, key="calc_prevstrk")
        angina = st.selectbox("ANGINA", ["No", "Yes"], index=0, key="calc_angina")
        mi_fchd = st.selectbox("MI_FCHD", ["No", "Yes"], index=0, key="calc_mi_fchd")
        stroke = st.selectbox("STROKE", ["No", "Yes"], index=0, key="calc_stroke")
        hyperten = st.selectbox("HYPERTEN", ["No", "Yes"], index=0, key="calc_hyperten")

    # Map to numeric row
    row = {
        "SEX": 1 if sex == "Male" else 0,
        "TOTCHOL": float(totchol),
        "AGE": float(age),
        "SYSBP": float(sysbp),
        "DIABP": float(diabp),
        "CIGPDAY": float(cigs),
        "BMI": float(bmi),
        "DIABETES": _as_int_yesno(diabetes),
        "BPMEDS": _as_int_yesno(bpmeds),
        "HEARTRTE": float(heartrate),
        "GLUCOSE": float(glucose),
        "educ": int(educ.split("–")[0].strip()),

        "PREVCHD": _as_int_yesno(prevchd),
        "PREVAP": _as_int_yesno(prevap),
        "PREVMI": _as_int_yesno(prevmi),
        "PREVSTRK": _as_int_yesno(prevstrk),
        "PREVHYP": _as_int_yesno(prevhyp),
        "HOSPMI": _as_int_yesno(hospmi),

        "HDLC": float(hdlc),
        "LDLC": float(ldlc),
        "ANGINA": _as_int_yesno(angina),
        "MI_FCHD": _as_int_yesno(mi_fchd),
        "STROKE": _as_int_yesno(stroke),
        "HYPERTEN": _as_int_yesno(hyperten),
    }

    # Ensure training order
    try:
        row_ordered = {feat: row[feat] for feat in FEATURES_24}
    except KeyError as e:
        st.error(f"features_24.json expects feature missing from UI mapping: {e}")
        st.stop()

    return pd.DataFrame([row_ordered])

def stacking_predict_proba_24(df_input: pd.DataFrame, threshold: float):
    """
    Correct stacking (same pattern as v4):
    - scale 24 features
    - p_rf, p_xgb from base models
    - meta LR uses ONLY [p_rf, p_xgb] (2 columns)
    """
    X = df_input.values.astype(float)
    Xs = scaler.transform(X)

    p_rf = rf_model.predict_proba(Xs)[:, 1]
    p_xgb = xgb_model.predict_proba(Xs)[:, 1]

    stack_in = np.column_stack([p_rf, p_xgb])
    p_final = meta_model.predict_proba(stack_in)[:, 1]

    final_prob = float(p_final[0])
    final_label = int(final_prob >= threshold)

    component_probs = {
        "RF (Clinical)": float(p_rf[0]),
        "XGB (Clinical)": float(p_xgb[0]),
    }
    return final_prob, final_label, component_probs

def bie_scenarios_24(df_patient: pd.DataFrame, threshold: float, include_advanced: bool = True):
    """
    BIE for v5 clinical+history model:
    - baseline
    - no smoking
    - lower SBP by 10
    - (advanced) BMI -2
    - (advanced) TOTCHOL -20
    - GLUCOSE -10
    """
    base_prob, _, _ = stacking_predict_proba_24(df_patient, threshold=threshold)

    scenarios = []
    scenarios.append(("Baseline", "Current profile", df_patient.copy()))

    if float(df_patient["CIGPDAY"].iloc[0]) > 0:
        d = df_patient.copy()
        d["CIGPDAY"] = 0.0
        scenarios.append(("No smoking", "Set cigarettes/day → 0", d))

    sysbp = float(df_patient["SYSBP"].iloc[0])
    d = df_patient.copy()
    d["SYSBP"] = max(sysbp - 10.0, 90.0)
    scenarios.append(("Lower SBP by 10 mmHg", f"SYSBP: {sysbp:.0f} → {d['SYSBP'].iloc[0]:.0f}", d))

    if include_advanced:
        bmi = float(df_patient["BMI"].iloc[0])
        d = df_patient.copy()
        d["BMI"] = max(bmi - 2.0, 15.0)
        scenarios.append(("Reduce BMI by 2 kg/m²", f"BMI: {bmi:.1f} → {d['BMI'].iloc[0]:.1f}", d))

        tc = float(df_patient["TOTCHOL"].iloc[0])
        d = df_patient.copy()
        d["TOTCHOL"] = max(tc - 20.0, 100.0)
        scenarios.append(("Lower Total Chol by 20 mg/dL", f"TOTCHOL: {tc:.0f} → {d['TOTCHOL'].iloc[0]:.0f}", d))

    gl = float(df_patient["GLUCOSE"].iloc[0])
    d = df_patient.copy()
    d["GLUCOSE"] = max(gl - 10.0, 60.0)
    scenarios.append(("Lower Glucose by 10 mg/dL", f"GLUCOSE: {gl:.0f} → {d['GLUCOSE'].iloc[0]:.0f}", d))

    rows = []
    best = None

    for name, desc, dfx in scenarios:
        p, _, _ = stacking_predict_proba_24(dfx, threshold=threshold)
        abs_change = (p - base_prob) * 100.0
        rel_change = (p - base_prob) / base_prob * 100.0 if base_prob > 0 else 0.0

        rows.append({
            "Scenario": name,
            "Description": desc,
            "Risk (%)": p * 100.0,
            "Δ abs (pp)": abs_change,
            "Δ rel (%)": rel_change,
        })

        if name != "Baseline":
            drop = base_prob - p
            if best is None or drop > best["drop"]:
                best = {"name": name, "p": p, "drop": drop}

    return base_prob, pd.DataFrame(rows), best

def _get_shap_explainers():
    """
    Build SHAP explainers once per app session.
    Avoids Streamlit cache hashing errors on unhashable model objects.
    """
    if not SHAP_AVAILABLE:
        return None, None

    if "rf_explainer" not in st.session_state or "xgb_explainer" not in st.session_state:
        st.session_state["rf_explainer"] = shap.TreeExplainer(rf_model)
        st.session_state["xgb_explainer"] = shap.TreeExplainer(xgb_model)

    return st.session_state["rf_explainer"], st.session_state["xgb_explainer"]


def _shap_local_bar(explainer, X_row_1xF, feature_names, title: str, max_display=12):
    """
    Render a simple local SHAP bar plot (Streamlit-friendly).
    X_row_1xF must be shape (1, F).
    """
    if not SHAP_AVAILABLE or explainer is None:
        st.info("SHAP is not available in this deployment. Add `shap` and `matplotlib` to requirements.txt.")
        return

    sv = explainer.shap_values(X_row_1xF)

    # Normalize to class 1 (binary)
    if isinstance(sv, list) and len(sv) == 2:
        sv = sv[1]

    # Try to convert common SHAP formats to a 2D array [1, F]
    if hasattr(sv, "values"):
        sv = sv.values
    sv = np.array(sv)

    # Some explainers return (1, F, 2) or (1, 2, F) — handle defensively
    if sv.ndim == 3 and sv.shape[-1] == 2:
        sv = sv[:, :, 1]  # class 1

    if sv.ndim != 2 or sv.shape[0] != 1:
        st.warning(f"Unexpected SHAP shape: {sv.shape}. Skipping SHAP plot.")
        return

    vals = np.ravel(sv[0])

    if len(feature_names) != len(vals):
        st.warning(
            f"Feature name count ({len(feature_names)}) != SHAP feature count ({len(vals)}). "
            "Skipping SHAP plot."
        )
        return

    idx = np.argsort(np.abs(vals))[::-1][:max_display]
    names = [feature_names[int(i)] for i in idx]
    impacts = vals[idx]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(range(len(idx))[::-1], impacts)
    ax.set_yticks(range(len(idx))[::-1])
    ax.set_yticklabels(names)
    ax.set_title(title)
    ax.set_xlabel("SHAP impact on model output (class=1)")
    ax.axvline(0, linewidth=1)
    plt.tight_layout()
    st.pyplot(fig)


# =========================
# 4) SIDEBAR
# =========================
with st.sidebar:
    st.markdown(
        """
        <h2 style='margin-bottom:0;'>🫀 CVD Stacking GenAI</h2>
        <p style='margin-top:4px;font-size:13px;'>
        <b>v5.0 – 24 Features (Clinical+History)</b><br>
        Framingham-based • Expanded clinical history (PREV* / HOSPMI)
        </p>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    # =========================
    # USER MODE (PATIENT vs CLINICIAN)
    # =========================
    user_mode = st.radio(
        "Display mode:",
        ["Patient Mode", "Clinician / Research Mode"],
        index=0,
        help="Patient Mode uses plain language and simplified explanations. Clinician Mode shows full technical details."
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
    else:
        show_components = st.checkbox(
            "Show component model probabilities (RF/XGB)",
            value=True,
            key="sidebar_show_components"
        )

    if IS_PATIENT_MODE:
        show_shap = False
    else:
        show_shap = st.checkbox(
            "Show SHAP local explanation (RF/XGB)",
            value=False,
            help="Requires `shap` + `matplotlib` in requirements.txt. Adds interpretability plots for the current patient.",
            key="sidebar_show_shap"
        )

    st.markdown("---")
    st.markdown(
        """
        **Disclaimer**  
        This tool is for **research & education** only and must not be used as
        a standalone diagnostic system.
        """
    )


# =========================
# 5) HERO HEADER
# =========================
st.markdown(
    """
    <div style="background-color:#0f4c75;padding:18px;border-radius:8px;margin-bottom:16px;">
      <h1 style="color:white;margin-bottom:4px;">CVD Risk Prediction – Clinical Risk Stratification Model (v5.0)</h1>
      <p style="color:#e0f2f1;margin:0;font-size:14px;">
        10-Year Cardiovascular Risk Estimation with Expanded Clinical History (PREV* / HOSPMI)
      </p>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================
# 6) TABS
# =========================
tab_calc, tab_bie, tab_model, tab_faq = st.tabs(
    ["🧮 Risk Calculator", "🧠 Behavioral Impact Engine (BIE)", "🧬 Model & Data", "❓ FAQ & Notes"]
)

# -------------------------
# TAB 1 – RISK CALCULATOR
# -------------------------
with tab_calc:
    st.markdown(
        "Use this tab to enter patient information and obtain a **10-year CVD risk estimate** (24-feature clinical+history model)."
    )

    df_input = build_input_df_24()

    if not IS_PATIENT_MODE:
        with st.expander("View encoded feature vector (for experts)", expanded=False):
            st.dataframe(df_input.style.format(precision=2), use_container_width=True)

    run_btn = st.button("Run CVD Risk Prediction", type="primary", key="btn_run_calc")

    if run_btn:
        with st.spinner("Running clinical stacking model..."):
            final_prob, final_label, component_probs = stacking_predict_proba_24(df_input, threshold=threshold)

        st.session_state["v5_last_input_df"] = df_input
        st.session_state["v5_last_prob"] = final_prob
        st.session_state["v5_last_label"] = final_label
        st.session_state["v5_last_components"] = component_probs
        st.session_state["v5_last_threshold"] = threshold

        category, color = interpret_risk(final_prob)

        st.markdown("### Prediction Result")
        col_res1, col_res2 = st.columns([2, 1])

        with col_res1:
            if IS_PATIENT_MODE:
                st.metric("Your estimated 10-year heart disease risk", f"{final_prob*100:.1f} %")
                st.markdown(
                    f"This means about **{int(round(final_prob*100))} out of 100** people with similar profiles "
                    "may develop heart disease over the next 10 years."
                )

                patient_cat = {
                    "Low risk": "Lower risk – continue healthy habits.",
                    "Borderline risk": "Moderate risk – worth paying attention to.",
                    "Intermediate risk": "Higher risk – medical guidance is recommended.",
                    "High risk": "High risk – medical follow-up is strongly recommended."
                }
                st.markdown(f"**Risk level:** {color} **{category}** — {patient_cat.get(category,'')}")
            else:
                st.metric("Estimated 10-year CVD risk", f"{final_prob*100:.1f} %")
                st.markdown(f"**Risk category:** {color} **{category}**")
                st.markdown(
                    f"**Model decision at threshold {threshold:.2f}:** "
                    f"{'⚠️ At Risk (1)' if final_label == 1 else '✅ Not Flagged (0)'}"
                )

        with col_res2:
            if IS_PATIENT_MODE:
                st.markdown(
                    """
                    **How to read the risk level**  
                    - <5%: Lower risk  
                    - 5–9%: Moderate (Borderline)  
                    - 10–19%: Higher risk  
                    - ≥20%: High risk  
                    """
                )
            else:
                st.markdown(
                    """
                    **Interpretation guide**  
                    - <5%: Low risk  
                    - 5–9%: Borderline  
                    - 10–19%: Intermediate  
                    - ≥20%: High risk  
                    """
                )

        # Patient-friendly "Why" summary (SHAP text only)
        if IS_PATIENT_MODE:
            st.markdown("### Why the model rated your risk this way")
            st.caption(
                "The model weighs many factors together. Some may slightly lower the score, "
                "but they cannot cancel out major risk factors."
            )
            if not SHAP_AVAILABLE:
                st.info("Explanation is simplified here. (Optional) Add `shap` + `matplotlib` to show detailed explanations.")
            else:
                _, xgb_explainer = _get_shap_explainers()

                X_raw = df_input.values.astype(float)
                X_scaled = scaler.transform(X_raw)

                sv = xgb_explainer.shap_values(X_scaled)
                if isinstance(sv, list):
                    sv = sv[1] if len(sv) >= 2 else sv[0]
                if hasattr(sv, "values"):
                    sv = sv.values
                sv = np.array(sv)
                if sv.ndim == 3 and sv.shape[-1] >= 2:
                    sv = sv[:, :, 1]
                if sv.ndim == 2 and sv.shape[0] == 1:
                    vals = np.ravel(sv[0])

                    FRIENDLY = {
                        "SYSBP": "High systolic blood pressure (top number)",
                        "DIABP": "High diastolic blood pressure (bottom number)",
                        "CIGPDAY": "Smoking (cigarettes per day)",
                        "AGE": "Age",
                        "SEX": "Male sex",
                        "GLUCOSE": "Blood sugar",
                        "DIABETES": "Diabetes",
                        "BPMEDS": "Blood pressure medication use",
                        "PREVMI": "Prior heart attack",
                        "PREVCHD": "Prior heart disease",
                        "PREVSTRK": "Prior stroke",
                        "PREVHYP": "History of hypertension",
                        "TOTCHOL": "Total cholesterol",
                        "BMI": "Body weight (BMI)",
                    }

                    # Sort positive and negative separately (patient-friendly)
                    pos_idx = np.argsort(vals)[::-1]      # descending
                    neg_idx = np.argsort(vals)            # ascending

                    inc = []
                    dec = []

                    for i in pos_idx:
                        if vals[int(i)] > 0:
                            inc.append(FRIENDLY.get(FEATURES_24[int(i)], FEATURES_24[int(i)]))
                        if len(inc) == 3:
                            break

                    for i in neg_idx:
                        if vals[int(i)] < 0:
                            dec.append(FRIENDLY.get(FEATURES_24[int(i)], FEATURES_24[int(i)]))
                        if len(dec) == 3:
                            break

                    if inc:
                        st.markdown("**What increases your risk most:**")
                        for t in inc[:3]:
                            st.markdown(f"- {t}")
                    if dec:
                        st.markdown("**What helps lower your risk:**")
                        for t in dec[:3]:
                            st.markdown(f"- {t}")

                st.caption("These factors explain how the *model* scored your profile (association-based), not guaranteed outcomes or treatment effects.")

        if show_components:
            st.markdown("### Component Model Contributions")
            comp_df = pd.DataFrame(
                {"Model": list(component_probs.keys()),
                 "Predicted CVD risk (%)": [p * 100 for p in component_probs.values()]}
            )
            st.bar_chart(comp_df.set_index("Model"))

        # SHAP local explanations (optional - clinician)
        if show_shap:
            with st.expander("Local explanation (SHAP) — why the base models scored this way", expanded=False):
                st.caption(
                    "SHAP highlights which features most pushed the RF and XGB predictions up or down for this patient. "
                    "This is an association-based explanation from the trained model (not a treatment-effect estimate)."
                )

                if not SHAP_AVAILABLE:
                    st.info("SHAP is not available here. Add `shap` and `matplotlib` to requirements.txt, then redeploy.")
                else:
                    rf_explainer, xgb_explainer = _get_shap_explainers()

                    X_raw = df_input.values.astype(float)
                    X_scaled = scaler.transform(X_raw)

                    st.markdown("**Random Forest (RF) — Local SHAP (Top drivers)**")
                    _shap_local_bar(rf_explainer, X_scaled, FEATURES_24, "RF: Top local drivers", max_display=12)

                    st.markdown("**XGBoost (XGB) — Local SHAP (Top drivers)**")
                    _shap_local_bar(xgb_explainer, X_scaled, FEATURES_24, "XGB: Top local drivers", max_display=12)

        st.info("Next: open **Behavioral Impact Engine (BIE)** to see patient-specific what-if scenarios.")
    else:
        st.info("Fill in the patient information and click **Run CVD Risk Prediction**.")


# -------------------------
# TAB 2 – BIE
# -------------------------
with tab_bie:
    st.subheader("What-If Simulator (Research-Based)" if IS_PATIENT_MODE else "Behavioral Impact Engine (BIE)")

    if IS_PATIENT_MODE:
        st.markdown(
            """
            Some "what-if" changes may appear counterintuitive. This occurs because the model reflects patterns in historical data, including treatment effects and clinical complexity. These results do not imply that lowering blood pressure or cholesterol is harmful.
            It is **not a guarantee** and **not medical advice**.
            """
        )
    else:
        st.markdown(
            """
            The **Behavioral Impact Engine (BIE)** performs local counterfactual re-scoring on an observational model. In high-risk, treated populations, lowering measured BP or cholesterol may move the profile toward clusters associated with advanced disease or secondary prevention, resulting in higher predicted risk. These effects reflect learned associations, not causal treatment effects.
            """
        )

    st.markdown("---")

    if "v5_last_input_df" not in st.session_state:
        st.warning("Please run a prediction in **Risk Calculator** first.")
    else:
        df_patient = st.session_state["v5_last_input_df"]
        used_threshold = st.session_state.get("v5_last_threshold", threshold)

        run_bie = st.button("Run BIE Analysis", key="btn_run_bie")

        if run_bie:
            base_prob, scenario_df, best = bie_scenarios_24(
                df_patient,
                threshold=used_threshold,
                include_advanced=not IS_PATIENT_MODE
            )
            category, color = interpret_risk(base_prob)

            st.markdown("### Patient-Specific Summary")
            st.markdown(f"Baseline risk (same model): **{base_prob*100:.1f}%**  ({color} {category})")

            st.markdown("### Scenario Table (model-based what-if analysis)")

            if IS_PATIENT_MODE:
                scenario_df = scenario_df.rename(columns={
                    "Δ abs (pp)": "Change in risk (points)",
                    "Δ rel (%)": "Relative change (%)"
                })

            # Patient-friendly formatting
            fmt = {}
            if "Risk (%)" in scenario_df.columns:
                fmt["Risk (%)"] = "{:.2f}"
            if "Δ abs (pp)" in scenario_df.columns:
                fmt["Δ abs (pp)"] = "{:+.2f}"
            if "Δ rel (%)" in scenario_df.columns:
                fmt["Δ rel (%)"] = "{:+.1f}"
            if "Change in risk (points)" in scenario_df.columns:
                fmt["Change in risk (points)"] = "{:+.2f}"
            if "Relative change (%)" in scenario_df.columns:
                fmt["Relative change (%)"] = "{:+.1f}"

            st.dataframe(
                scenario_df.style.format(fmt),
                use_container_width=True
            )

            if IS_PATIENT_MODE:
                st.markdown("### Most helpful next step (based on this model)")
                if best is None:
                    st.write("No scenario produced a measurable change (rare).")
                else:
                    st.success(
                        f"**{best['name']}**\n\nEstimated change: **{base_prob*100:.2f}% → {best['p']*100:.2f}%**"
                    )
                    st.caption("This is a model-based what-if estimate (not a guarantee).")
            else:
                st.markdown("### Most impactful lever (for this profile)")
                if best is None:
                    st.write("No scenario produced a measurable change (rare).")
                else:
                    abs_pp = best["drop"] * 100.0
                    rel_pct = (best["drop"] / base_prob * 100.0) if base_prob > 0 else 0.0
                    st.write(
                        f"**{best['name']}** → estimated risk becomes **{best['p']*100:.2f}%** "
                        f"(**-{abs_pp:.2f} pp absolute**, **-{rel_pct:.1f}% relative**)."
                    )

            st.markdown("### Evidence-based recommendations (starter set)")
            st.markdown(
                """
                **Smoking**
                - If the patient smokes, discuss smoking cessation options and supports.

                **Blood Pressure**
                - If SBP/DBP are elevated, discuss lifestyle and clinician-guided management.

                **Metabolic**
                - Encourage clinician-guided support for diabetes risk factors and glucose control.

                **Lifestyle**
                - Discuss nutrition, activity, sleep, and stress reduction (individualized).
                """
            )

        else:
            st.info("Click **Run BIE Analysis** to generate patient-specific what-if scenarios.")


# -------------------------
# TAB 3 – MODEL & DATA
# -------------------------
with tab_model:
    st.subheader("Model & Data")
    st.markdown(
        """
        **MODEL & DATA OVERVIEW**  

        This application implements a stacked machine-learning model to estimate 10-year cardiovascular disease (CVD) risk, using an expanded clinical and prior-history feature set.

        The model is designed for research and educational use, supporting both clinical interpretation and patient understanding of cardiovascular risk factors.

        **Data Source**  
        - Primary source: Framingham Heart Study–based dataset
        - Target variable: 10-year cardiovascular disease (CVD) outcome
        - Prediction task: Binary classification → probability of a CVD event within 10 years

        The Framingham dataset is widely used in cardiovascular risk modeling and provides long-term follow-up with standardized clinical variables.

        **Feature Set (24 features — Clinical + History)**  
        This v5.0 model extends v4.0 by incorporating expanded prior cardiovascular history, while preserving core preventive risk factors.

        **Demographics**  
        - AGE – Age (years)
        - SEX – Biological sex
        - educ – Education level (proxy for socioeconomic factors)

        **Blood Pressure**  
        - SYSBP – Systolic blood pressure (mmHg)
        - DIABP – Diastolic blood pressure (mmHg)
        - HYPERTEN – Hypertension diagnosis
        - PREVHYP – Prior hypertension history

        **Lipids**  
        - TOTCHOL – Total cholesterol
        - HDLC – HDL cholesterol
        - LDLC – LDL cholesterol

        **Metabolic**  
        - MI – Body mass index
        - GLUCOSE – Fasting glucose
        - DIABETES – Diabetes diagnosis

        **Lifestyle**  
        - CIGPDAY – Cigarettes per day

        **Cardiac / Vital Signs**  
        - HEARTRTE – Heart rate

        **Treatment**  
        - BPMEDS – Blood pressure medication use

        **Symptoms & Prior Cardiovascular History**  
        - ANGINA – Angina / chest pain
        - MI_FCHD – Family history of myocardial infarction
        - PREVAP - Prevalent Angina Pectoris at exam
        - PREVCHD – Prior coronary heart disease
        - PREVMI – Prior myocardial infarction
        - HOSPMI – Hospitalized myocardial infarction
        - STROKE – Stroke indicator
        - PREVSTRK – Prior stroke history

        ⚠️ Important: Some history variables may appear protective or harmful depending on whether they are marked Yes or No for a given individual. Interpretations are association-based, not causal.
        
        **MODEL ARCHITECTURE (Stacking Ensemble)**

        **Base Learners**  
        - RandomForestClassifier (clinical patterns & interactions)
        - XGBoostClassifier (nonlinear boosting dynamics)

        **Each base model outputs:**  
        - p_RF (CVD=1∣x)
        - p_XGB (CVD=1∣x)

        **Meta-Learner (Stacking)**  
        - Logistic Regression
        - Input:
                z=[p_RF,p_XGB]
        - Output:
                p_stack (CVD=1∣x)

        The meta-model learns how to optimally combine the strengths of the base learners.

        **Model Output**  
        - Primary output: Probability of a 10-year CVD event
        - Displayed as: Percentage (%) with risk category
        - Risk categories (educational guidance):
          - <5% → Lower risk
          - 5–9% → Borderline risk
          - 10–19% → Higher risk
          - ≥20% → High risk

        **Interpretability Components**  
        **SHAP (Local Explanations)** 
        - Explains why the RF and XGB models scored a given profile higher or lower
        - Feature contributions are local and patient-specific
        - SHAP values reflect model associations, not treatment effects

        **Behavioral Impact Engine (BIE)**  
        - Provides counterfactual “what-if” scenarios
        - Shows how small changes (e.g., smoking, BP, cholesterol) would alter risk
        - Uses the same trained model to ensure internal consistency`

        **HANDLING OF MISSING VALUES (NaN Imputation)** 
        
        Missing values were handled using feature-appropriate imputation strategies. Continuous variables were imputed using training-set medians, while categorical variables were imputed using the most frequent category. All imputation parameters were learned exclusively from the training data to prevent information leakage. These transparent methods were chosen to balance clinical realism, reproducibility, and interpretability.
        """
    )
    st.markdown("**Artifacts loaded from repo root:** scaler_24.pkl, rf_clin24.pkl, xgb_clin24.pkl, stack_meta_clin24.pkl, features_24.json")


# -------------------------
# TAB 4 – FAQ & NOTES
# -------------------------
with tab_faq:
    st.subheader("FAQ & Notes")
    st.markdown(
        """
        **Why can some changes look counterintuitive (e.g., BMI/Cholesterol)?**  
        This model is trained to learn statistical patterns from the dataset. Some relationships can appear reversed
        due to confounding, treatment effects (e.g., sicker patients receiving therapy), sampling artifacts, or feature interactions.
        BIE is a *model-based what-if*, not a causal treatment estimate.

        **This is not medical advice.**  
        Always consult a licensed clinician for diagnosis and treatment decisions.

        **Is this a clinical decision support tool?**  
        No. This application is intended for research and educational purposes only. It must not be used as a standalone diagnostic or treatment decision system.

        **What does “10-year CVD risk” mean?**  
        A 10-year CVD risk represents the estimated probability that an individual will experience a cardiovascular event within the next 10 years, based on their current risk profile and patterns learned from the training data.
        It is a population-based estimate, not a prediction of certainty for any single person.

        **How should I interpret the percentage?**  
        A predicted risk of 15% means that, among 100 people with similar profiles, approximately 15 may experience a CVD event within 10 years.
        This does not guarantee that an event will or will not occur for an individual.

        **What events are included under “CVD”?**  
        Depending on the dataset definition, CVD may include:
        - Myocardial infarction (heart attack)
        - Stroke
        - Coronary heart disease and related cardiovascular outcomes
        The model predicts the same outcome label used during training.

        **Why does the model sometimes show both “risk-increasing” and “risk-lowering” factors?**  
        The model evaluates all features together.
        Some factors may slightly lower the score relative to the highest-risk patients, but they cannot cancel out major risk drivers such as diabetes, severe hypertension, smoking, or prior cardiovascular disease.

        **Why do some history variables appear protective?**  
        If a severe event indicator is marked “No”, the model interprets that as lower risk relative to patients who have experienced that event.
        This does not mean the factor is inherently protective — only that it is less risky than the alternative.

        **How is this risk typically used in practice?**  
        In clinical settings, 10-year CVD risk estimates are commonly used to:
        - Support lifestyle counseling
        - Guide blood pressure and lipid management discussions
        - Enable shared decision-making between patients and clinicians
        This app demonstrates how such estimates may be generated using machine-learning methods.

        **Important Limitations**  
        - The model reflects patterns learned from the training population
        - Performance may vary across populations, healthcare settings, and data availability
        - Predictions depend on feature accuracy, calibration, and population similarity
        - Outputs must not replace professional medical judgment

        **🧭 Versioning Note**  
        - v3.0: Prevention-focused, minimal history, non-leakage setup
        - v4.0: Added limited clinical history
        - v5.0: Expanded prior CVD history + stacking ensemble (RF + XGB)

        Future versions (e.g., v5.1) may explore synthetic data augmentation (GANs) for robustness and subgroup balance.
        """
    )

# =========================
# 7) FOOTER
# =========================
st.markdown(
    """
    <hr style="margin-top:32px;margin-bottom:8px;">
    <div style="text-align:center;font-size:12px;color:gray;">
      Stacking Generative AI CVD Risk Model v5.0 • 24 features (clinical+history) • Research & Demonstration Only<br>
      This application does not provide medical advice, diagnosis, or treatment.<br>
      © 2025 Howard Nguyen, PhD. For demonstration only — not for clinical decision-making.
    </div>
    """,
    unsafe_allow_html=True
)