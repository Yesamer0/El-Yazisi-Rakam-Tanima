import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import cv2
import numpy as np

# Sayfa ayarları
st.set_page_config(page_title="AI Rakam Tanıyıcı", layout="centered")

# 1. Modeli Yükle
@st.cache_resource
def model_yukle():
    return tf.keras.models.load_model('el_yazisi_modeli.h5')

model = model_yukle()

# Ekranı sıfırlamak için bir sayaç tutuyoruz
if 'canvas_key' not in st.session_state:
    st.session_state.canvas_key = 0

st.title("🖌️ Akıllı Rakam Tanıma v2")
st.write("Siyah alana rakamı çizin ve 'Tahmin Et' butonuna basın.")

# 2. Yan Menü - Sadece Fırça Kalsın
st.sidebar.header("Kontrol Paneli")
firca_kalinligi = st.sidebar.slider("Fırça Kalınlığı: ", 10, 50, 25)

# 3. Çizim Alanı ve Butonlar
col1, col2 = st.columns([1, 1])

with col1:
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=firca_kalinligi,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=300,
        width=300,
        drawing_mode="freedraw",
        display_toolbar=False, # Alttaki araç çubuğu kapalı
        key=f"canvas_{st.session_state.canvas_key}", # Sıfırlama için dinamik key
    )

with col2:
    # TAHMİN ET BUTONU
    if st.button('Tahmin Et 🚀', use_container_width=True):
        if canvas_result.image_data is not None:
            img = canvas_result.image_data.astype(np.uint8)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_resized = cv2.resize(img_gray, (28, 28))
            img_normalized = img_resized / 255.0
            img_reshaped = img_normalized.reshape(1, 28, 28)

            prediction = model.predict(img_reshaped)
            sonuc = np.argmax(prediction)
            olasilik = np.max(prediction)

            st.success(f"Tahmin: {sonuc}")
            st.metric("Güven Oranı", f"%{olasilik*100:.1f}")
        else:
            st.error("Lütfen önce bir rakam çizin!")
    
    # EKRANI SIFIRLA BUTONU
    if st.button('Ekranı Sıfırla 🧹', use_container_width=True):
        st.session_state.canvas_key += 1 # Anahtarı değiştirerek canvas'ı zorla sıfırlıyoruz
        st.rerun()

st.divider()