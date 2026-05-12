import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="MeghaAI",
    page_icon="☁️",
    layout="centered" # Responsive untuk mobile & desktop
)

# =========================================================
# CUSTOM CSS (Modern & Responsive)
# =========================================================
# Menggunakan CSS kustom untuk memberikan efek premium
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        text-align: center;
        margin-top: -2rem;
        margin-bottom: 2rem;
    }
    
    .title {
        font-size: 3rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #38BDF8, #2563EB);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #64748B;
        font-weight: 400;
    }
    
    .info-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(150, 150, 150, 0.2);
        padding: 1.5rem;
        border-radius: 16px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }
    
    .prediction-title {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #94A3B8;
        margin-bottom: 0.5rem;
    }
    
    .prediction-value {
        font-size: 2rem;
        font-weight: 800;
        color: #38BDF8;
        margin-bottom: 1rem;
    }
    
    .weather-info {
        font-size: 1rem;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# CACHE MODEL
# =========================================================
@st.cache_resource
def load_keras_model():
    return load_model("model_awan.keras")

try:
    model = load_keras_model()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# =========================================================
# CLASS NAMES & WEATHER INFO
# =========================================================
class_names = [
    'cirriform', 'clear_sky', 'cumulonimbus', 'cumulus', 
    'high_cumuliform', 'stratiform', 'stratocumulus'
]

weather_info = {
    'cirriform': 'Cuaca cenderung cerah, namun dapat menandakan perubahan cuaca.',
    'clear_sky': 'Langit cerah dan kondisi cuaca stabil.',
    'cumulonimbus': 'Berpotensi terjadi hujan lebat, petir, atau badai.',
    'cumulus': 'Cuaca umumnya cerah berawan.',
    'high_cumuliform': 'Kemungkinan terjadi perubahan cuaca ringan.',
    'stratiform': 'Langit mendung dan berpotensi hujan ringan.',
    'stratocumulus': 'Cuaca berawan dengan kemungkinan gerimis.'
}

# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div class="main-header">
    <div class="title">MeghaAI ☁️</div>
    <div class="subtitle">Intelligent Cloud & Weather Analysis System</div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# INPUT SECTION (TABS)
# =========================================================
tab1, tab2 = st.tabs(["📸 Ambil dari Kamera", "📁 Upload File"])

image_source = None

with tab1:
    camera_image = st.camera_input("Ambil foto awan secara langsung")
    if camera_image:
        image_source = camera_image

with tab2:
    uploaded_file = st.file_uploader("Atau pilih gambar awan dari perangkat Anda", type=['jpg', 'jpeg', 'png'])
    if uploaded_file:
        image_source = uploaded_file

# =========================================================
# PROCESSING & RESULTS
# =========================================================
if image_source is not None:
    st.markdown("---")
    
    # Preprocess
    image = Image.open(image_source).convert('RGB')
    
    # Layout dua kolom: Kiri untuk gambar, Kanan untuk hasil
    # Secara otomatis akan bertumpuk (stack) jika dibuka di mobile
    col1, col2 = st.columns([1, 1.2], gap="medium")
    
    with col1:
        st.image(image, caption="Gambar Input", use_container_width=True)
        
    with col2:
        img = image.resize((224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        with st.spinner("Menganalisis jenis awan..."):
            prediction = model.predict(img_array)
            predicted_class = class_names[np.argmax(prediction)]
            confidence = np.max(prediction) * 100
            weather_prediction = weather_info[predicted_class]
            
        st.markdown(f"""
        <div class="info-card">
            <div class="prediction-title">Hasil Prediksi</div>
            <div class="prediction-value">{predicted_class.replace('_', ' ').title()}</div>
            <div class="weather-info">
                <strong>🌦️ Analisis Cuaca:</strong><br>
                {weather_prediction}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Menampilkan indikator Confidence
        st.metric(label="Tingkat Kepercayaan Model", value=f"{confidence:.2f}%")
        st.progress(float(confidence) / 100)

    # =========================================================
    # CHART (HIDDEN IN EXPANDER)
    # =========================================================
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📊 Lihat Detail Probabilitas Semua Jenis Awan"):
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(class_names)))
        
        labels = [name.replace('_', ' ').title() for name in class_names]
        
        bars = ax.bar(labels, prediction[0], color=colors)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        ax.set_ylabel("Probabilitas", fontsize=10)
        ax.set_xlabel("Jenis Awan", fontsize=10)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        st.pyplot(fig)

# =========================================================
# FOOTER
# =========================================================
st.markdown("""
<div style="text-align: center; margin-top: 50px; margin-bottom: 20px; color: #94A3B8; font-size: 0.85rem;">
    MeghaAI © 2026 • Intelligent Cloud & Weather Analysis
</div>
""", unsafe_allow_html=True)