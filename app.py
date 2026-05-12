import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="Cloud Classification AI",
    page_icon="☁️",
    layout="centered"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f7fa;
    }

    .title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: #1E3A5F;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #4B5563;
        margin-bottom: 30px;
    }

    .result-box {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# LOAD MODEL
# =========================
model = load_model('model_awan.keras')

# =========================
# NAMA KELAS
# =========================
class_names = [
    'cirriform',
    'clear_sky',
    'cumulonimbus',
    'cumulus',
    'high_cumuliform',
    'stratiform',
    'stratocumulus'
]

# =========================
# HEADER
# =========================
st.markdown('<div class="title">☁️ Cloud Classification AI</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="subtitle">'
    'Sistem klasifikasi jenis awan berbasis Deep Learning menggunakan CNN'
    '</div>',
    unsafe_allow_html=True
)

# =========================
# UPLOAD FILE
# =========================
uploaded_file = st.file_uploader(
    "Upload gambar awan",
    type=['jpg', 'jpeg', 'png']
)

# =========================
# PREDIKSI
# =========================
if uploaded_file is not None:

    # Membaca gambar
    image = Image.open(uploaded_file).convert('RGB')

    # Tampilkan gambar
    st.image(image, caption='Gambar yang diupload', use_container_width=True)

    # Resize gambar
    img = image.resize((224, 224))

    # Convert ke array
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Prediksi
    prediction = model.predict(img_array)

    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # =========================
    # HASIL PREDIKSI
    # =========================
    st.markdown('<div class="result-box">', unsafe_allow_html=True)

    st.subheader("Hasil Klasifikasi")

    st.success(f"Jenis awan terdeteksi: {predicted_class}")

    st.info(f"Confidence Score: {confidence:.2f}%")

    st.markdown('</div>', unsafe_allow_html=True)

    # =========================
    # VISUALISASI PROBABILITAS
    # =========================
    st.subheader("Probabilitas Setiap Kelas")

    fig, ax = plt.subplots(figsize=(8,5))

    ax.bar(class_names, prediction[0])

    plt.xticks(rotation=20)
    plt.ylabel('Probability')
    plt.xlabel('Cloud Type')

    st.pyplot(fig)

# =========================
# FOOTER
# =========================
st.markdown('---')
st.caption('Developed using TensorFlow, CNN, and Streamlit')
