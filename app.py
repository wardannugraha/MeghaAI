import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt

# =========================================================
# KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(
    page_title="Cloud Weather Analysis AI",
    page_icon="☁️",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}

.main {
    background: linear-gradient(to bottom right, #e0f2fe, #f8fafc);
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.title {
    text-align: center;
    font-size: 52px;
    font-weight: 800;
    color: #0F172A;
    margin-bottom: 10px;
}

.subtitle {
    text-align: center;
    font-size: 20px;
    color: #475569;
    margin-bottom: 40px;
}

.card {
    background: white;
    padding: 25px;
    border-radius: 22px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.result-title {
    font-size: 28px;
    font-weight: bold;
    color: #0F172A;
}

.prediction {
    font-size: 32px;
    font-weight: bold;
    color: #2563EB;
}

.weather-box {
    background-color: #EFF6FF;
    padding: 18px;
    border-radius: 15px;
    border-left: 6px solid #2563EB;
    margin-top: 15px;
}

.footer {
    text-align: center;
    color: gray;
    margin-top: 50px;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
# =========================================================
model = load_model("model_awan.keras")

# =========================================================
# NAMA KELAS
# =========================================================
class_names = [
    'cirriform',
    'clear_sky',
    'cumulonimbus',
    'cumulus',
    'high_cumuliform',
    'stratiform',
    'stratocumulus'
]

# =========================================================
# INTERPRETASI CUACA
# =========================================================
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
<div class="title">
☁️ Cloud Weather Analysis AI
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="subtitle">
Sistem Analisis Cuaca Berbasis Deep Learning Menggunakan CNN
</div>
""", unsafe_allow_html=True)

# =========================================================
# LAYOUT
# =========================================================
col1, col2 = st.columns([1, 1])

# =========================================================
# UPLOAD FILE
# =========================================================
with col1:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("📤 Upload Gambar Awan")

    uploaded_file = st.file_uploader(
        "Pilih gambar awan",
        type=['jpg', 'jpeg', 'png']
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert('RGB')

        st.image(
            image,
            caption='Gambar yang diupload',
            use_container_width=True
        )

    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# PREDIKSI
# =========================================================
with col2:

    if uploaded_file is not None:

        # Resize image
        img = image.resize((224, 224))

        # Convert image
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Prediction
        prediction = model.predict(img_array)

        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        weather_prediction = weather_info[predicted_class]

        # =========================================================
        # HASIL
        # =========================================================
        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.markdown("""
        <div class="result-title">
        🔍 Hasil Analisis
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="prediction">
        {predicted_class}
        </div>
        """, unsafe_allow_html=True)

        st.progress(float(confidence) / 100)

        st.write(f"Confidence Score: **{confidence:.2f}%**")

        # =========================================================
        # ANALISIS CUACA
        # =========================================================
        st.markdown(f"""
        <div class="weather-box">
        🌦️ <b>Analisis Cuaca:</b><br><br>
        {weather_prediction}
        </div>
        """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# VISUALISASI PROBABILITAS
# =========================================================
if uploaded_file is not None:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("📊 Probabilitas Setiap Jenis Awan")

    fig, ax = plt.subplots(figsize=(10, 5))

    bars = ax.bar(class_names, prediction[0])

    ax.set_ylabel("Probability")
    ax.set_xlabel("Cloud Type")

    plt.xticks(rotation=15)

    st.pyplot(fig)

    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# FOOTER
# =========================================================
st.markdown("""
<div class="footer">
Developed using Streamlit, TensorFlow, CNN, and Deep Learning
</div>
""", unsafe_allow_html=True)