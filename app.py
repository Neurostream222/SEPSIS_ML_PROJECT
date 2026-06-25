import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Set up a professional, clean page layout
st.set_page_config(page_title="Sepsis Early Warning System", layout="centered")

st.title("🏥 Clinical Sepsis Early-Warning Interface")
st.markdown("---")
st.subheader("Patient Bedside Metrics Input")

# 2. Load the trained AI model we exported from the notebook
@st.cache_resource
def load_clinical_model():
    model = joblib.load('vitals_sepsis_model.pkl')
    features = joblib.load('model_features.pkl')
    return model, features

try:
    model, model_features = load_clinical_model()
except Exception as e:
    st.error("Could not locate the saved model files. Please run Cell 10 in your notebook first.")
    st.stop()

# 3. Create interactive sliders and numeric entry fields for the clinician
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Patient Age", min_value=1, max_value=120, value=65)
    bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=60.0, value=25.6)
    gcs_total = st.slider("Glasgow Coma Scale (Mental Status)", min_value=3, max_value=15, value=15)
    qsofa = st.slider("Current qSOFA Score", min_value=0, max_value=3, value=1)

with col2:
    hr_mean = st.slider("Mean Heart Rate (Past Hour)", min_value=40, max_value=200, value=85)
    hr_max = st.slider("Max Heart Rate (Past Hour)", min_value=40, max_value=200, value=95)
    hr_delta_3h = st.slider("3-Hour Heart Rate Delta (Rate of Change)", min_value=-50, max_value=50, value=10)
    qsofa_delta_3h = st.slider("3-Hour qSOFA Delta", min_value=-3, max_value=3, value=0)

st.markdown("---")

# 4. Process predictions when the clinician clicks the assessment button
if st.button("🧬 Run Predictive Diagnostic Assessment", type="primary"):
    
    # Structure the inputs into a dictionary mapping exactly to our model's requirements
    input_data = {
        'hr_mean': hr_mean, 'hr_max': hr_max, 'hr_delta_3h': hr_delta_3h,
        'qsofa': qsofa, 'qsofa_delta_3h': qsofa_delta_3h, 'gcs_total': gcs_total,
        'bmi': bmi, 'age': age
    }
    
    # Convert to DataFrame matching the exact structural column sequence
    input_df = pd.DataFrame([input_data])[model_features]
    
    # Calculate probability scores instead of a rigid binary 0 or 1
    probabilities = model.predict_proba(input_df)[0]
    sepsis_probability = probabilities[1] * 100
    
    # 5. Display high-visibility triage alerts based on systemic risk thresholds
    st.subheader("AI Diagnostic Risk Analysis Summary:")
    
    if sepsis_probability < 30:
        st.success(f"🟢 Low Sepsis Risk Profile: {sepsis_probability:.1f}%")
        st.info("Recommendation: Maintain standard baseline monitoring intervals.")
    elif 30 <= sepsis_probability < 70:
        st.warning(f"🟡 Elevated Clinical Risk Profile: {sepsis_probability:.1f}%")
        st.info("Recommendation: Initiate frequent vital charting and consider blood culture panels.")
    else:
        st.error(f"🔴 CRITICAL Sepsis Risk Profile: {sepsis_probability:.1f}%")
        st.info("Recommendation: Urgent clinical evaluation required. Evaluate systemic inflammatory response protocols.")