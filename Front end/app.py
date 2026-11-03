from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


ROOT = Path(__file__).parent.parent
MODEL_PATH = ROOT / "outputs_final" / "models" / "maternal_risk_final_model.joblib"
METADATA_PATH = ROOT / "outputs_final" / "models" / "model_metadata.json"

RISK_LABELS = {
    0: "Minimum Risk",
    1: "Medium Risk",
    2: "High Risk",
    "0": "Minimum Risk",
    "1": "Medium Risk",
    "2": "High Risk",
    "low risk": "Minimum Risk",
    "mid risk": "Medium Risk",
    "high risk": "High Risk",
}
RISK_COLORS = {
    0: "#3bb273",
    1: "#e9a23b",
    2: "#e45756",
    "0": "#3bb273",
    "1": "#e9a23b",
    "2": "#e45756",
    "low risk": "#3bb273",
    "mid risk": "#e9a23b",
    "high risk": "#e45756",
}


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model artifact not found at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def build_features(age, systolic_bp, diastolic_bp, blood_sugar, body_temp, heart_rate):
    pulse_pressure = systolic_bp - diastolic_bp
    mean_arterial_pressure = diastolic_bp + pulse_pressure / 3
    systolic_diastolic_ratio = systolic_bp / diastolic_bp
    return pd.DataFrame(
        [
            {
                "Age": age,
                "SystolicBP": systolic_bp,
                "DiastolicBP": diastolic_bp,
                "BS": blood_sugar,
                "BodyTemp": body_temp,
                "HeartRate": heart_rate,
                "PulsePressure": pulse_pressure,
                "MeanArterialPressure": mean_arterial_pressure,
                "SystolicDiastolicRatio": systolic_diastolic_ratio,
                "AgeGroup_adult": int(age >= 18),
                "AgeGroup_advanced_maternal_age": int(age >= 35),
            }
        ]
    )


def risk_key(label):
    value = getattr(label, "item", lambda: label)()
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, int):
        return value
    return str(value).strip().lower()


def risk_badge(label):
    key = risk_key(label)
    color = RISK_COLORS.get(key, "#2a9d8f")
    title = RISK_LABELS.get(key, str(label).title())
    st.markdown(
        f"""
        <div class="risk-result" style="border-color:{color};">
            <div class="risk-kicker">MODEL ASSESSMENT</div>
            <div class="risk-title" style="color:{color};">{title}</div>
            <div class="risk-copy">Use this result as a screening signal alongside qualified clinical judgment.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(
    page_title="Materna | Maternal Risk Screening",
    page_icon="♡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');
    :root { --ink:#17324d; --muted:#60758a; --teal:#168b87; --pale:#eef8f7; }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: linear-gradient(135deg,#fbfdfd 0%,#f2f8f8 55%,#eef5fb 100%); color:var(--ink); }
    h1, h2, h3 { font-family:'Playfair Display', serif; color:var(--ink); }
    .hero { padding: 2rem 0 1.1rem; }
    .eyebrow { color:var(--teal); font-weight:700; letter-spacing:.13em; font-size:.75rem; }
    .hero h1 { font-size:clamp(2.4rem,5vw,4.7rem); line-height:1.04; margin:.35rem 0 .8rem; }
    .hero p { color:var(--muted); font-size:1.08rem; max-width:650px; line-height:1.65; }
    .glass-card { background:rgba(255,255,255,.82); border:1px solid rgba(22,139,135,.13); border-radius:20px; padding:1.25rem 1.35rem; box-shadow:0 12px 35px rgba(23,50,77,.06); }
    .stat-number { font-size:1.8rem; font-weight:700; color:var(--teal); }
    .stat-label { color:var(--muted); font-size:.83rem; }
    .risk-result { border:2px solid; border-radius:18px; padding:1.3rem 1.4rem; background:#fff; margin:1rem 0 1.2rem; }
    .risk-kicker { color:var(--muted); font-size:.72rem; font-weight:700; letter-spacing:.12em; }
    .risk-title { font-family:'Playfair Display',serif; font-size:2.1rem; font-weight:700; margin:.25rem 0; }
    .risk-copy { color:var(--muted); font-size:.9rem; }
    .care-note { color:#244b62; background:#e9f5f7; border:1px solid #b8dfe3; padding:.72rem 1rem; border-radius:12px; font-size:.86rem; line-height:1.5; }
    .stNumberInput input { color:#ffffff !important; background-color:#17324d !important; -webkit-text-fill-color:#ffffff !important; }
    .stNumberInput input:focus { border-color:#74d4cf !important; box-shadow:0 0 0 .15rem rgba(116,212,207,.2) !important; }
    section[data-testid="stSidebar"] { background:#17324d; }
    section[data-testid="stSidebar"] * { color:#f5fbfb !important; }
    .stButton > button { border-radius:10px; font-weight:700; padding:.65rem 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

try:
    model = load_model()
except Exception as exc:
    st.error("The prediction model could not be loaded.")
    st.code(str(exc))
    st.stop()

with st.sidebar:
    st.markdown("## Materna")
    st.caption("Evidence-informed maternal risk screening")
    st.markdown("---")
    st.markdown("### Navigation")
    page = st.radio("Go to", ["Risk assessment", "Model card", "About"], label_visibility="collapsed")
    st.markdown("---")
    st.caption("Designed for use with healthcare professionals")

if page == "Risk assessment":
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">MATERNAL HEALTH INTELLIGENCE</div>
          <h1>Maternal risk<br>assessment.</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="care-note"><b>Use with a healthcare professional.</b> Review the assessment alongside clinical history, examination, and local care protocols.</div>',
        unsafe_allow_html=True,
    )
    st.write("")

    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Patient observations")
        st.caption("Use the most recent reliable clinical measurements.")
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age (years)", min_value=10.0, max_value=70.0, value=25.0, step=1.0)
            systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=70.0, max_value=200.0, value=120.0, step=1.0)
            blood_sugar = st.number_input("Blood sugar (mmol/L)", min_value=2.0, max_value=30.0, value=7.0, step=0.1, format="%.1f")
        with c2:
            diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=40.0, max_value=130.0, value=80.0, step=1.0)
            body_temp = st.number_input("Body temperature (°F)", min_value=95.0, max_value=108.0, value=98.0, step=0.1, format="%.1f")
            heart_rate = st.number_input("Heart rate (bpm)", min_value=40.0, max_value=180.0, value=70.0, step=1.0)
        submit = st.button("Assess maternal risk", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Assessment")
        st.caption("Your result will appear here after assessment.")
        if submit:
            if diastolic_bp >= systolic_bp:
                st.error("Systolic blood pressure should be higher than diastolic blood pressure.")
            else:
                features = build_features(age, systolic_bp, diastolic_bp, blood_sugar, body_temp, heart_rate)
                prediction = model.predict(features)[0]
                probabilities = model.predict_proba(features)[0]
                predicted_label = risk_key(prediction)
                risk_badge(predicted_label)
                st.markdown("**Model confidence distribution**")
                for label, probability in zip(model.classes_, probabilities):
                    label_key = risk_key(label)
                    st.progress(float(probability), text=f"{RISK_LABELS.get(label_key, str(label).title())}  ·  {probability:.1%}")
                with st.expander("Derived clinical features"):
                    st.dataframe(
                        features[["PulsePressure", "MeanArterialPressure", "SystolicDiastolicRatio"]].rename(
                            columns={
                                "PulsePressure": "Pulse pressure",
                                "MeanArterialPressure": "Mean arterial pressure",
                                "SystolicDiastolicRatio": "Systolic/diastolic ratio",
                            }
                        ),
                        hide_index=True,
                        use_container_width=True,
                    )
        else:
            st.info("Complete the observation form and select **Assess maternal risk**.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    s1, s2, s3, s4 = st.columns(4)
    for column, number, label in [
        (s1, "1,014", "observations analyzed"),
        (s2, "3", "risk categories"),
        (s3, "84.7%", "test accuracy"),
        (s4, "0.953", "ROC-AUC"),
    ]:
        with column:
            st.markdown(f'<div class="glass-card"><div class="stat-number">{number}</div><div class="stat-label">{label}</div></div>', unsafe_allow_html=True)

elif page == "Model card":
    st.markdown('<div class="hero"><div class="eyebrow">TRANSPARENCY</div><h1>Model card</h1><p>Understand what powers the assessment and where it should be used with care.</p></div>', unsafe_allow_html=True)
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Final selected model")
        st.markdown("### Random Forest")
        st.write("Selected after benchmarking seven classification algorithms. The model uses clinical measurements plus engineered hemodynamic features.")
        st.metric("Macro F1", "0.852")
        st.metric("Balanced accuracy", "0.857")
        st.metric("ROC-AUC (OvR)", "0.953")
        st.markdown("</div>", unsafe_allow_html=True)
    with b:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Input signals")
        st.write("Age, systolic and diastolic blood pressure, blood sugar, body temperature, and heart rate.")
        st.write("Derived automatically: pulse pressure, mean arterial pressure, systolic/diastolic ratio, adult indicator, and advanced maternal age indicator.")
        st.markdown("</div>", unsafe_allow_html=True)
    st.write("")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Performance snapshot")
    chart = ROOT / "outputs_final" / "figures" / "06_holdout_model_comparison.png"
    if chart.exists():
        st.image(str(chart), use_container_width=True)
    st.caption("Holdout performance from the completed project experiment. Metrics describe the test set and do not guarantee performance for an individual patient.")
    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown('<div class="hero"><div class="eyebrow">ABOUT THE PROJECT</div><h1>Maternal health<br>risk assessment.</h1></div>', unsafe_allow_html=True)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Developer")
    st.markdown("**Yabets Desalegn**")
    st.write("Data science and machine learning project developer.")
    st.markdown("[GitHub profile](https://github.com/Yabets-art)  ·  [Project repository](https://github.com/Yabets-art/Maternal_Health_Risk_Yabets)")
    st.subheader("Clinical use")
    st.write("Use this assessment with a healthcare professional, together with clinical history, examination, and local care protocols.")
    st.subheader("Risk categories")
    st.write("The model classifies observations as Low Risk, Mid Risk, or High Risk. These labels represent the training dataset’s categories and are not a replacement for a clinician’s assessment.")
    st.markdown("</div>", unsafe_allow_html=True)
