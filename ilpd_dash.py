import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="ILPD Liver Patient Prediction",
    page_icon="🏥",
    layout="wide"
)

# Title and description
st.title("🏥 Indian Liver Patient Dataset - Prediction Dashboard")
st.markdown("---")
st.write("Dashboard ini digunakan untuk memprediksi apakah seseorang memiliki penyakit liver berdasarkan data medis.")

# Sidebar untuk input
st.sidebar.header("📝 Input Data Pasien")

# Input fields berdasarkan kolom dataset ILPD
def user_input_features():
    age = st.sidebar.slider('Age (Umur)', 4, 90, 32)
    gender = st.sidebar.selectbox('Gender (Jenis Kelamin)', ['Male', 'Female'])
    tb = st.sidebar.number_input('TB (Total Bilirubin)', min_value=0.0, max_value=75.0, value=1.0, step=0.1)
    db = st.sidebar.number_input('DB (Direct Bilirubin)', min_value=0.0, max_value=20.0, value=0.3, step=0.1)
    alkphos = st.sidebar.number_input('Alkphos (Alkaline Phosphotase)', min_value=0, max_value=2500, value=200, step=1)
    sgpt = st.sidebar.number_input('SGPT (Alamine Aminotransferase)', min_value=0, max_value=2500, value=25, step=1)
    sgot = st.sidebar.number_input('SGOT (Aspartate Aminotransferase)', min_value=0, max_value=5000, value=30, step=1)
    tp = st.sidebar.number_input('TP (Total Proteins)', min_value=0.0, max_value=15.0, value=7.0, step=0.1)
    alb = st.sidebar.number_input('ALB (Albumin)', min_value=0.0, max_value=10.0, value=4.0, step=0.1)
    ag_ratio = st.sidebar.number_input('A/G Ratio (Albumin/Globulin Ratio)', min_value=0.0, max_value=5.0, value=1.0, step=0.1)
    
    # Convert gender to numeric (Male=1, Female=0)
    gender_numeric = 1 if gender == 'Male' else 0
    
    data = {
        'Age': age,
        'Gender': gender_numeric,
        'TB': tb,
        'DB': db,
        'Alkphos': alkphos,
        'Sgpt': sgpt,
        'Sgot': sgot,
        'TP': tp,
        'ALB': alb,
        'A/G Ratio': ag_ratio
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

# Get user input
input_df = user_input_features()

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Data Input Pasien")
    st.write("Data yang akan digunakan untuk prediksi:")
    
    # Display input in a nice format
    display_df = input_df.copy()
    display_df['Gender'] = display_df['Gender'].map({1: 'Male', 0: 'Female'})
    st.dataframe(display_df, use_container_width=True)
    
    # Model information
    st.subheader("🤖 Informasi Model")
    st.info("""
    Model yang digunakan telah dilatih menggunakan dataset ILPD (Indian Liver Patient Dataset) 
    dari UCI Machine Learning Repository. Model akan memprediksi apakah pasien memiliki 
    penyakit liver (Selector = 1) atau tidak (Selector = 0).
    """)

with col2:
    st.subheader("🎯 Prediksi")
    
    # Prediction button
    if st.button('🔮 Predict', type="primary", use_container_width=True):
        try:
            # Load model (ganti dengan path model Anda)
            model_path = 'ilpd_dtc_model.joblib'
            
            # Coba load model dan scaler
            try:
                model = joblib.load(model_path)
                scaler = StandardScaler()
                
                with st.spinner('Melakukan prediksi...'):
                    # Scale the input
                    input_scaled = scaler.fit_transform(input_df[['Age', 'Gender', 'TB', 'DB', 'Alkphos', 'Sgpt', 'Sgot', 'TP', 'ALB', 'A/G Ratio']])
                    st.write(input_scaled)
                    
                    # Make prediction
                    prediction = model.predict(input_scaled)
                    prediction_proba = model.predict_proba(input_scaled)
                    
                    # Display results
                    if prediction[0] == 1:
                        st.error("⚠️ **PREDIKSI: POSITIF LIVER DISEASE**")
                        st.write("Pasien diprediksi **MEMILIKI** penyakit liver.")
                    else:
                        st.success("✅ **PREDIKSI: NEGATIF LIVER DISEASE**")
                        st.write("Pasien diprediksi **TIDAK MEMILIKI** penyakit liver.")
                    
                    # Show probability
                    st.write("**Probabilitas:**")
                    prob_healthy = prediction_proba[0][0] * 100
                    prob_disease = prediction_proba[0][1] * 100
                    
                    st.write(f"- Tidak memiliki liver disease: {prob_healthy:.1f}%")
                    st.write(f"- Memiliki liver disease: {prob_disease:.1f}%")
                    
                    # Progress bars for probability
                    st.progress(prob_healthy/100, text=f"Sehat: {prob_healthy:.1f}%")
                    st.progress(prob_disease/100, text=f"Sakit: {prob_disease:.1f}%")
                    
            except FileNotFoundError:
                st.error("❌ **Model tidak ditemukan!**")
                st.write("""
                **Langkah-langkah untuk menggunakan dashboard ini:**
                
                1. Latih model menggunakan dataset ILPD
                2. Simpan model dengan joblib:
                   ```python
                   import joblib
                   joblib.dump(model, 'liver_patient_model.pkl')
                   joblib.dump(scaler, 'scaler.pkl')
                   ```
                3. Pastikan file model berada di direktori yang sama dengan script ini
                4. Jalankan ulang dashboard
                """)
                
        except Exception as e:
            st.error(f"❌ Terjadi error: {str(e)}")
    
    # Information about the dataset
    st.subheader("📋 Informasi Dataset")
    st.write("""
    **Dataset Features:**
    - **Age**: Umur pasien
    - **Gender**: Jenis kelamin (Male/Female)
    - **TB**: Total Bilirubin
    - **DB**: Direct Bilirubin  
    - **Alkphos**: Alkaline Phosphotase
    - **Sgpt**: Alamine Aminotransferase
    - **Sgot**: Aspartate Aminotransferase
    - **TP**: Total Proteins
    - **ALB**: Albumin
    - **A/G Ratio**: Albumin/Globulin Ratio
    
    **Target**: Selector (1=Liver Disease, 0=No Disease)
    """)

# Footer
st.markdown("---")
st.markdown("**Note:** Dashboard ini hanya untuk tujuan edukasi dan demonstrasi. Konsultasikan dengan tenaga medis profesional untuk diagnosis yang akurat.")

# Instructions for running
st.sidebar.markdown("---")
st.sidebar.markdown("**💡 Cara Menjalankan:**")
st.sidebar.markdown("""
1. Pastikan model telah disimpan
2. Letakkan file di direktori yang sama
3. Jalankan: `streamlit run dashboard.py`
""")