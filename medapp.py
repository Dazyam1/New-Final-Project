import streamlit as st
import joblib
import numpy as np

# Page config
st.set_page_config(page_title="MedPredict AI", layout="wide", page_icon="🧬")

# Minimal safe CSS
st.markdown("""
<style>
    * { font-family: 'Inter', sans-serif; }
    .hero { background: linear-gradient(135deg, #667eea, #764ba2); padding: 2rem; color: white; text-align: center; }
    .hero h1 { font-size: 3rem; margin-bottom: 0.5rem; }
    .form-section { background: white; padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
</style>
""", unsafe_allow_html=True)

# Load models using joblib
@st.cache_resource
def load_models():
    models = {}

    try:
        models["hepatitis"] = joblib.load("models/hepatitis_model.pkl")
    except Exception as e:
        models["hepatitis"] = None
        st.error(f"⚠️ Error loading Hepatitis model: {e}")

    try:
        models["hiv"] = joblib.load("models/hiv_model.pkl")
        models["vectorizer"] = joblib.load("models/vectorizer.pkl")
    except Exception as e:
        models["hiv"] = models["vectorizer"] = None
        st.error(f"⚠️ Error loading HIV model/vectorizer: {e}")

    try:
        models["tb"] = joblib.load("models/tb_predictor_model.pkl")
    except Exception as e:
        models["tb"] = None
        st.error(f"⚠️ Error loading TB model: {e}")

    return models

# Load all models once
models = load_models()

# Sidebar
with st.sidebar:
    st.title("🩺 MedPredict AI")
    choice = st.radio("Choose Analysis", ["Hepatitis", "HIV", "Tuberculosis"])
    st.markdown("---")
    st.info("All models are trained and locally deployed.")

# Hero section
st.markdown("""
<div class="hero">
    <h1>MedPredict AI</h1>
    <p>Advanced ML-powered diagnosis for Hepatitis, HIV, and TB</p>
</div>
""", unsafe_allow_html=True)

# Mapping
def map_bool(val): return {'False': 0, 'True': 1, 'Unknown': -1}[val]
def map_sex(val): return 0 if val == "male" else 1

# Hepatitis Model
if choice == "Hepatitis":
    st.header("🫀 Hepatitis Prognosis")
    if models["hepatitis"] is None:
        st.error("⚠️ Hepatitis model not found.")
    else:
        with st.form("hep_form"):
            col1, col2 = st.columns(2)
            with col1:
                age = st.slider("Age", 7, 78, 40)
                sex = st.selectbox("Sex", ["male", "female"])
                steroid = st.selectbox("Steroid", ["False", "True", "Unknown"])
                antivirals = st.selectbox("Antivirals", ["False", "True"])
                fatigue = st.selectbox("Fatigue", ["False", "True", "Unknown"])
                anorexia = st.selectbox("Anorexia", ["False", "True", "Unknown"])
                spiders = st.selectbox("Spiders", ["False", "True", "Unknown"])
                ascites = st.selectbox("Ascites", ["False", "True", "Unknown"])
            with col2:
                liver_big = st.selectbox("Liver Enlarged", ["False", "True", "Unknown"])
                spleen_palpable = st.selectbox("Spleen Palpable", ["False", "True", "Unknown"])
                varices = st.selectbox("Varices", ["False", "True", "Unknown"])
                histology = st.selectbox("Histology", ["False", "True"])
                bilirubin = st.number_input("Bilirubin (mg/dL)", 0.3, 8.0, 1.0)
                alk_phos = st.number_input("Alk Phosphatase", 26, 295, 85)
                sgot = st.number_input("SGOT", 14, 648, 25)
                albumin = st.number_input("Albumin", 2.1, 6.4, 4.0)
                protime = st.number_input("Prothrombin Time", 0, 100, 85)

            submit = st.form_submit_button("Predict")

            if submit:
                features = [
                    age, map_sex(sex), map_bool(steroid), map_bool(antivirals),
                    map_bool(fatigue), map_bool(anorexia), map_bool(liver_big),
                    map_bool(spleen_palpable), map_bool(spiders), map_bool(ascites),
                    map_bool(varices), map_bool(histology), bilirubin, alk_phos,
                    sgot, albumin, protime
                ]
                pred = models["hepatitis"].predict([features])[0]
                if pred == 1:
                    st.success("✅ Favorable Prognosis")
                else:
                    st.error("⚠️ Concerning Prognosis")

# HIV Model
elif choice == "HIV":
    st.header("🧪 HIV Risk Assessment")
    if models["hiv"] is None or models["vectorizer"] is None:
        st.error("⚠️ Model or vectorizer not found.")
    else:
        symptoms = [
            "Fever", "Night Sweats", "Fatigue", "Weight Loss", "Diarrhea", "Skin Lesions",
            "Oral Candidiasis", "Lymph Nodes", "Neuro Symptoms", "Opportunistic Infections"
        ]
        selected = st.multiselect("Select Present Symptoms", symptoms)

        if st.button("Assess Risk"):
            if not selected:
                st.warning("⚠️ Please select symptoms")
            else:
                vec = models["vectorizer"].transform([", ".join(selected)])
                pred = models["hiv"].predict(vec)[0]
                prob = models["hiv"].predict_proba(vec)[0][pred]

                if pred == 1:
                    st.error(f"🚨 High Risk (Confidence: {prob:.2%})")
                else:
                    st.success(f"✅ Low Risk (Confidence: {prob:.2%})")

# TB Model
elif choice == "Tuberculosis":
    st.header("🫁 Tuberculosis Screening")
    if models["tb"] is None:
        st.error("⚠️ TB model not found.")
    else:
        tb_symptoms = [
            "Fever", "Cough", "Night Sweats", "Weight Loss", "Chest Pain", 
            "Hemoptysis", "Fatigue", "Lymphadenopathy"
        ]
        tb_data = []
        for sym in tb_symptoms:
            val = st.selectbox(f"{sym}?", ["Absent", "Present"], key=sym)
            tb_data.append(1 if val == "Present" else 0)

        if st.button("Run TB Screening"):
            pred = models["tb"].predict([tb_data])[0]
            prob = models["tb"].predict_proba([tb_data])[0][pred]
            if pred == 1:
                st.error(f"🚨 High TB Risk (Confidence: {prob:.2%})")
            else:
                st.success(f"✅ Low TB Risk (Confidence: {prob:.2%})")

# Footer
st.markdown("---")
st.markdown("⚠️ *This is an educational prototype. Always consult a medical professional for real-world diagnosis.*")
