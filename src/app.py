import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import keras

st.set_page_config(
    page_title="Smart Waste Classifier - UAS KA",
    page_icon="♻️",
    layout="centered",
)

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

WASTE_INFO = {
    "cardboard": {
        "nama_id": "Kardus / Karton (Cardboard)",
        "kategori_tong": "Tempat Sampah Anorganik / Kertas (Biru)",
        "status": "Dapat Didaur Ulang (Layak Bank Sampah)",
        "panduan": "Lipat atau pipihkan kardus agar hemat ruang. Pastikan tidak basah atau terkena minyak makanan.",
        "warna": "#1E88E5",
    },
    "glass": {
        "nama_id": "Kaca / Beling (Glass)",
        "kategori_tong": "Tempat Sampah Anorganik (Kuning)",
        "status": "Dapat Didaur Ulang (Perlu Penanganan Hati-hati)",
        "panduan": "Bilas sisa cairan di dalam botol/toples kaca. Jika pecah, bungkus dengan koran/kardus tebal sebelum dibuang demi keamanan petugas.",
        "warna": "#FDD835",
    },
    "metal": {
        "nama_id": "Logam / Kaleng (Metal)",
        "kategori_tong": "Tempat Sampah Anorganik (Kuning)",
        "status": "Bernilai Tinggi di Bank Sampah",
        "panduan": "Kosongkan isi kaleng, bilas singkat, dan pipihkan jika memungkinkan sebelum dimasukkan ke wadah daur ulang.",
        "warna": "#78909C",
    },
    "paper": {
        "nama_id": "Kertas (Paper)",
        "kategori_tong": "Tempat Sampah Anorganik / Kertas (Biru)",
        "status": "Dapat Didaur Ulang",
        "panduan": "Pisahkan kertas kering (HVS, koran, majalah) dari kertas tisu atau kertas pembungkus makanan yang berminyak.",
        "warna": "#1E88E5",
    },
    "plastic": {
        "nama_id": "Plastik (Plastic)",
        "kategori_tong": "Tempat Sampah Anorganik (Kuning)",
        "status": "Dapat Didaur Ulang (Layak Bank Sampah)",
        "panduan": "Kosongkan botol/gelas plastik, lepas label jika mudah, dan remukkan untuk menghemat volume sampah.",
        "warna": "#FB8C00",
    },
    "trash": {
        "nama_id": "Sampah Residu / Campuran (Trash)",
        "kategori_tong": "Tempat Sampah Residu (Abu-abu / Hitam)",
        "status": "Tidak Dapat Didaur Ulang (Buang ke TPA)",
        "panduan": "Masukkan ke tempat sampah residu tertutup. Hindari mencampur sampah residu dengan sampah daur ulang yang bersih.",
        "warna": "#546E7A",
    },
}


@st.cache_resource
def load_trained_model(model_path):
    return keras.models.load_model(model_path)


def preprocess_image(image_bytes):
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_AREA)
    # Input model menerima rentang [0, 255] float32 karena preprocess_input sudah ada di dalam layer model
    input_tensor = np.expand_dims(img_resized.astype(np.float32), axis=0)
    return img_rgb, input_tensor


# --- SIDEBAR ---
st.sidebar.title("Pengaturan Model")
available_models = {}
if os.path.exists("models/baseline_model.keras"):
    available_models["Baseline MobileNetV2 (Akurasi 81.46%)"] = "models/baseline_model.keras"
if os.path.exists("models/improved_model.keras"):
    available_models["Improved MobileNetV2 (Akurasi 79.63%)"] = "models/improved_model.keras"

if not available_models:
    st.error("File model tidak ditemukan di folder `models/`. Jalankan training terlebih dahulu.")
    st.stop()

selected_label = st.sidebar.selectbox("Pilih Model AI:", list(available_models.keys()))
model_path = available_models[selected_label]
confidence_threshold = st.sidebar.slider("Batas Minimum Keyakinan (%)", 40, 90, 60, 5)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Panduan Foto Terbaik:**\n"
    "- Pastikan hanya ada **1 objek sampah utama** di tengah foto.\n"
    "- Gunakan pencahayaan yang cukup terang.\n"
    "- Hindari latar belakang yang terlalu ramai."
)

# --- HALAMAN UTAMA ---
st.title("♻️ Smart Waste Classifier")
st.caption("Sistem Klasifikasi Sampah Rumah Tangga Berbasis Deep Learning (MobileNetV2) — Proyek UAS Kecerdasan Artifisial 2026")

tab_upload, tab_camera = st.tabs(["📁 Unggah Foto Sampah", "📷 Ambil Foto dari Kamera"])

image_data = None
with tab_upload:
    uploaded_file = st.file_uploader("Pilih gambar sampah (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_data = uploaded_file.read()

with tab_camera:
    camera_file = st.camera_input("Arahkan sampah ke kamera lalu klik Take Photo")
    if camera_file is not None:
        image_data = camera_file.read()

if image_data is not None:
    model = load_trained_model(model_path)
    img_rgb, input_tensor = preprocess_image(image_data)

    with st.spinner("Menganalisis material sampah..."):
        preds = model.predict(input_tensor, verbose=0)[0]

    top_idx = int(np.argmax(preds))
    top_class = CLASSES[top_idx]
    confidence = float(preds[top_idx]) * 100.0
    info = WASTE_INFO[top_class]

    st.markdown("---")
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.image(img_rgb, caption="Citra Input Pengguna", use_container_width=True)

    with col2:
        st.subheader("Hasil Klasifikasi")
        if confidence < confidence_threshold:
            st.warning(
                f"⚠️ **Tingkat keyakinan rendah ({confidence:.1f}%)**. "
                "Objek kurang jelas atau latar terlalu ramai. Silakan foto ulang dengan pencahayaan lebih baik."
            )
        else:
            st.success(f"**Terdeteksi: {info['nama_id']}**")

        st.metric(label="Tingkat Keyakinan (Confidence)", value=f"{confidence:.2f}%")
        st.markdown(f"**Tempat Sampah:** {info['kategori_tong']}")
        st.markdown(f"**Status Daur Ulang:** {info['status']}")
        st.info(f"💡 **Saran Pemilahan:** {info['panduan']}")

    # Grafik probabilitas semua kelas
    st.markdown("### Distribusi Probabilitas Tiap Kelas")
    prob_df = pd.DataFrame(
        {"Kategori": [WASTE_INFO[c]["nama_id"] for c in CLASSES], "Probabilitas (%)": preds * 100}
    ).sort_values("Probabilitas (%)", ascending=True)
    st.bar_chart(prob_df.set_index("Kategori"))
else:
    st.info("Silakan unggah foto sampah atau gunakan kamera di atas untuk mulai menguji sistem.")