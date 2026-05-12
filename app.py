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
    page_title="Cloud Weather Analysis AI",
    page_icon="☁️",
    layout="centered"
)

# =========================================================
# DARK / LIGHT MODE
# =========================================================
theme = st.toggle("🌙 Dark Mode")

if theme:
    bg_color = "#0F172A"
    card_color = "#1E293B"
    text_color = "#F8FAFC"
    subtext_color = "#CBD5E1"
    accent_color = "#38BDF8"
    border_color = "#334155"
else:
    bg_color = "#F1F5F9"
    card_color = "#FFFFFF"
    text_color = "#0F172A"
    subtext_color = "#475569"
    accent_color = "#2563EB"
    border_color = "#E2E8F0"

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown(f"""
<style>

.stApp {{
    background-color: {bg_color};
}}

html, body, [class*="css"] {{
    font-family: 'Segoe UI', sans-serif;
}}

.main-container {{
    max-width: 750px;
    margin: auto;
}}

.main-card {{
    background-color: {card_color};
    padding: 30px;
    border-radius: 24px;
    border: 1px solid {border_color};
    box-shadow: 0 8px 24px rgba(0,0,0,0.08);
    margin-top: 20px;
}}

.title {{
    text-align: center;
    font-size: 44px;
    font-weight: 800;
    color: {text_color};
    margin-bottom: 10px;
}}

.subtitle {{
    text-align: center;
    font-size: 18px;
    color: {subtext_color};
    margin-bottom: 35px;
}}

.result-title {{
    color: {text_color};
    font-size: 28px;
    font-weight: bold;
}}

.prediction {{
    color: {accent_color};
    font-size: 32px;
    font-weight: bold;
    margin-top: 10px;
}}

.analysis {{
    color: {text_color};
    font-size: 17px;
    line-height: 1.7;
    margin-top: 20px;
}}

.footer {{
    text-align: center;
    color: {subtext_color};
    margin-top: 50px;
    margin-bottom: 20px;
    font-size: 14px;
}}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
# =========================================================
model = load_model("model_awan.keras")

# =========================================================
# CLASS NAMES
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
# WEATHER ANALYSIS
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
st.markdown('<div class="main-container">', unsafe_allow_html=True)

st.markdown(f"""
<div class="title">
☁️ Cloud Weather Analysis AI
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="subtitle">
Sistem Analisis Cuaca Berbasis Deep Learning Menggunakan CNN
</div>
""", unsafe_allow_html=True)

# =========================================================
# MAIN CARD
# =========================================================
st.markdown('<div class="main-card">', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload gambar awan",
    type=['jpg', 'jpeg', 'png']
)

if uploaded_file is not None:

    # =========================================================
    # READ IMAGE
    # =========================================================
    image = Image.open(uploaded_file).convert('RGB')

    st.image(
        image,
        caption='Gambar yang diupload',
        use_container_width=True
    )

    # =========================================================
    # PREPROCESSING
    # =========================================================
    img = image.resize((224, 224))

    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # =========================================================
    # PREDICTION
    # =========================================================
    prediction = model.predict(img_array)

    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    weather_prediction = weather_info[predicted_class]

    # =========================================================
    # RESULT
    # =========================================================
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
    # WEATHER ANALYSIS
    # =========================================================
    st.markdown(f"""
    <div class="analysis">
    🌦️ <b>Analisis Cuaca:</b><br><br>
    {weather_prediction}
    </div>
    """, unsafe_allow_html=True)

    # =========================================================
    # CHART
    # =========================================================
    st.markdown("### 📊 Probabilitas Setiap Jenis Awan")

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.bar(class_names, prediction[0])

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
Developed using Streamlit, TensorFlow, and CNN
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)