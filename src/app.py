import os
from datetime import datetime
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

# Custom CSS agar tampilan kartu lebih modern dan nyaman di HP
st.markdown(
    """
    <style>
    .waste-card {
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 8px solid;
        background-color: rgba(128, 128, 128, 0.08);
        margin-bottom: 1rem;
    }
    .badge-tong {
        display: inline-block;
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.85rem;
        color: white;
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

WASTE_INFO = {
    "cardboard": {
        "nama_id": "Kardus / Karton",
        "tong": "Tong Biru — Anorganik Kertas",
        "status": "Dapat Didaur Ulang (Layak Bank Sampah)",
        "estimasi_nilai": "Rp 1.500 – Rp 2.500 / kg",
        "panduan": "Lipat atau pipihkan kardus agar hemat ruang. Pastikan tidak basah atau terkena minyak makanan.",
        "warna": "#1E88E5",
        "ikon": "📦",
    },
    "glass": {
        "nama_id": "Kaca / Beling",
        "tong": "Tong Kuning — Anorganik Kemasan",
        "status": "Dapat Didaur Ulang (Penanganan Hati-hati)",
        "estimasi_nilai": "Rp 500 – Rp 1.000 / kg",
        "panduan": "Bilas sisa cairan di dalam botol/toples kaca. Jika pecah, bungkus koran/kardus tebal demi keamanan petugas.",
        "warna": "#E6A800",
        "ikon": "🫙",
    },
    "metal": {
        "nama_id": "Logam / Kaleng",
        "tong": "Tong Kuning — Anorganik Logam",
        "status": "Bernilai Tinggi di Bank Sampah",
        "estimasi_nilai": "Rp 10.000 – Rp 14.000 / kg (Aluminium)",
        "panduan": "Kosongkan isi kaleng, bilas singkat, dan pipihkan sebelum dimasukkan ke wadah daur ulang.",
        "warna": "#546E7A",
        "ikon": "🥫",
    },
    "paper": {
        "nama_id": "Kertas",
        "tong": "Tong Biru — Anorganik Kertas",
        "status": "Dapat Didaur Ulang",
        "estimasi_nilai": "Rp 1.500 – Rp 2.500 / kg",
        "panduan": "Pisahkan kertas kering (HVS, koran, majalah) dari tisu atau kertas pembungkus makanan yang berminyak.",
        "warna": "#1E88E5",
        "ikon": "📄",
    },
    "plastic": {
        "nama_id": "Plastik",
        "tong": "Tong Kuning — Anorganik Plastik",
        "status": "Dapat Didaur Ulang (Layak Bank Sampah)",
        "estimasi_nilai": "Rp 2.000 – Rp 4.000 / kg",
        "panduan": "Kosongkan botol/gelas plastik, lepas label jika mudah, dan remukkan untuk menghemat volume sampah.",
        "warna": "#FB8C00",
        "ikon": "🧴",
    },
    "trash": {
        "nama_id": "Sampah Residu / Campuran",
        "tong": "Tong Abu-abu / Hitam — Residu TPA",
        "status": "Tidak Dapat Didaur Ulang",
        "estimasi_nilai": "Tidak bernilai jual (Buang ke TPA)",
        "panduan": "Masukkan ke tempat sampah residu tertutup. Hindari mencampur residu dengan sampah daur ulang bersih.",
        "warna": "#424242",
        "ikon": "🗑️",
    },
}

# Inisialisasi riwayat sesi pengguna
if "scan_history" not in st.session_state:
    st.session_state.scan_history = []


@st.cache_resource
def load_trained_model(model_path):
    return keras.models.load_model(model_path)


def preprocess_image(image_bytes):
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_AREA)
    input_tensor = np.expand_dims(img_resized.astype(np.float32), axis=0)
    return img_rgb, input_tensor


def predict_single(model, image_bytes):
    img_rgb, input_tensor = preprocess_image(image_bytes)
    preds = model.predict(input_tensor, verbose=0)[0]
    top_idx = int(np.argmax(preds))
    top_class = CLASSES[top_idx]
    confidence = float(preds[top_idx]) * 100.0
    return img_rgb, top_class, confidence, preds


def add_to_history(source_name, top_class, confidence):
    info = WASTE_INFO[top_class]
    st.session_state.scan_history.insert(
        0,
        {
            "Waktu": datetime.now().strftime("%H:%M:%S"),
            "Sumber": source_name,
            "Kategori": f"{info['ikon']} {info['nama_id']}",
            "Keyakinan": f"{confidence:.1f}%",
            "Tempat Sampah": info["tong"],
            "is_recyclable": top_class != "trash",
        },
    )


# --- SIDEBAR ---
st.sidebar.title("⚙️ Pengaturan Sistem")
available_models = {}
if os.path.exists("models/baseline_model.keras"):
    available_models["Baseline MobileNetV2 (Akurasi 81.46%)"] = "models/baseline_model.keras"
if os.path.exists("models/improved_model.keras"):
    available_models["Improved MobileNetV2 (Akurasi 79.63%)"] = "models/improved_model.keras"

if not available_models:
    st.error("File model tidak ditemukan di folder `models/`.")
    st.stop()

selected_label = st.sidebar.selectbox("Arsitektur Model:", list(available_models.keys()))
model_path = available_models[selected_label]
confidence_threshold = st.sidebar.slider("Ambang Batas Keyakinan (%)", 40, 90, 60, 5)

st.sidebar.markdown("---")
st.sidebar.subheader("📚 Panduan Kode Warna Tong")
st.sidebar.markdown(
    """
    - 🔵 **Biru:** Kertas & Kardus kering
    - 🟡 **Kuning:** Plastik, Logam, & Kaca
    - ⚫ **Abu-abu/Hitam:** Residu (tidak bisa didaur ulang)
    """
)
st.sidebar.caption("Tip: Foto 1 objek sampah di tengah layar dengan cahaya cukup untuk hasil paling akurat.")

# --- HEADER UTAMA ---
st.title("♻️ Smart Waste Classifier")
st.caption("Asisten Pemilahan Sampah Rumah Tangga & Estimasi Bank Sampah Berbasis Deep Learning")

model = load_trained_model(model_path)

# --- TABS FITUR UTAMA ---
tab_single, tab_batch, tab_history = st.tabs(
    ["📸 Pindai Satuan / Kamera", "🗂️ Pindai Banyak Sekaligus", "📊 Riwayat & Rekap"]
)

# ==========================================
# TAB 1: PINDAI SATUAN / KAMERA
# ==========================================
with tab_single:
    input_mode = st.radio(
        "Pilih metode input:",
        ["Unggah dari Galeri / File", "Ambil Foto Langsung (Kamera)"],
        horizontal=True,
    )

    image_bytes = None
    source_label = "Upload"

    if input_mode == "Unggah dari Galeri / File":
        uploaded = st.file_uploader("Pilih foto sampah (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])
        if uploaded is not None:
            image_bytes = uploaded.read()
            source_label = uploaded.name
    else:
        cam_photo = st.camera_input("Arahkan kamera ke 1 objek sampah lalu ambil foto")
        if cam_photo is not None:
            image_bytes = cam_photo.read()
            source_label = "Kamera Langsung"

    if image_bytes is not None:
        with st.spinner("Mengklasifikasikan material sampah..."):
            img_rgb, top_class, confidence, preds = predict_single(model, image_bytes)

        info = WASTE_INFO[top_class]

        # Simpan otomatis ke riwayat jika belum tercatat persis sama di urutan teratas
        if (
            not st.session_state.scan_history
            or st.session_state.scan_history[0]["Sumber"] != source_label
            or st.session_state.scan_history[0]["Keyakinan"] != f"{confidence:.1f}%"
        ):
            add_to_history(source_label, top_class, confidence)

        st.markdown("---")
        col_img, col_res = st.columns([1, 1.3])

        with col_img:
            st.image(img_rgb, caption=source_label, use_container_width=True)

        with col_res:
            if confidence < confidence_threshold:
                st.warning(
                    f"⚠️ **Keyakinan Rendah ({confidence:.1f}%)** — Objek kurang jelas atau latar terlalu ramai. "
                    "Disarankan mengambil ulang foto dengan pencahayaan lebih terang."
                )

            st.markdown(
                f"""
                <div class="waste-card" style="border-left-color: {info['warna']};">
                    <span class="badge-tong" style="background-color: {info['warna']};">{info['tong']}</span>
                    <h3 style="margin: 0.2rem 0;">{info['ikon']} {info['nama_id']}</h3>
                    <p style="margin: 0.2rem 0;"><strong>Tingkat Keyakinan:</strong> {confidence:.2f}%</p>
                    <p style="margin: 0.2rem 0;"><strong>Status:</strong> {info['status']}</p>
                    <p style="margin: 0.2rem 0;"><strong>Estimasi Harga Bank Sampah:</strong> {info['estimasi_nilai']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.info(f"💡 **Langkah Pemilahan:** {info['panduan']}")

        with st.expander("🔍 Lihat Detail Probabilitas 6 Kelas (Explainable AI)"):
            prob_df = pd.DataFrame(
                {
                    "Kategori": [f"{WASTE_INFO[c]['ikon']} {WASTE_INFO[c]['nama_id']}" for c in CLASSES],
                    "Probabilitas (%)": np.round(preds * 100, 2),
                }
            ).sort_values("Probabilitas (%)", ascending=True)
            st.bar_chart(prob_df.set_index("Kategori"))
    else:
        st.info("Pilih foto sampah dari galeri atau gunakan kamera untuk mulai memindai.")

# ==========================================
# TAB 2: PINDAI BANYAK SEKALIGUS (BATCH)
# ==========================================
with tab_batch:
    st.write("Unggah beberapa foto sampah sekaligus untuk mengelompokkannya ke tempat sampah yang sesuai.")
    batch_files = st.file_uploader(
        "Pilih beberapa gambar sekaligus", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="batch"
    )

    if batch_files:
        if st.button(f"Proses {len(batch_files)} Foto Sekarang", type="primary"):
            batch_results = []
            progress = st.progress(0)

            for idx, file in enumerate(batch_files):
                f_bytes = file.read()
                img_rgb, top_class, confidence, _ = predict_single(model, f_bytes)
                info = WASTE_INFO[top_class]
                add_to_history(file.name, top_class, confidence)
                batch_results.append((file.name, img_rgb, info, confidence))
                progress.progress((idx + 1) / len(batch_files))

            st.success(f"Selesai memindai {len(batch_results)} item sampah!")
            cols = st.columns(2)
            for i, (fname, img_rgb, info, conf) in enumerate(batch_results):
                with cols[i % 2]:
                    st.image(img_rgb, use_container_width=True)
                    st.markdown(f"**{info['ikon']} {info['nama_id']}** ({conf:.1f}%)")
                    st.caption(f"{info['tong']} • {fname}")
                    st.markdown("---")

# ==========================================
# TAB 3: RIWAYAT & REKAP DAMPAK
# ==========================================
with tab_history:
    history = st.session_state.scan_history
    if not history:
        st.info("Belum ada sampah yang dipindai pada sesi ini. Coba pindai gambar di tab pertama atau kedua.")
    else:
        total_scan = len(history)
        total_daur_ulang = sum(1 for item in history if item["is_recyclable"])
        total_residu = total_scan - total_daur_ulang

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Sampah Dipindai", f"{total_scan} item")
        m2.metric("Layak Daur Ulang", f"{total_daur_ulang} item")
        m3.metric("Sampah Residu", f"{total_residu} item")

        df_hist = pd.DataFrame(history).drop(columns=["is_recyclable"])
        st.dataframe(df_hist, use_container_width=True, hide_index=True)

        col_dl, col_clr = st.columns([1, 1])
        with col_dl:
            csv_data = df_hist.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Unduh Laporan Pemilahan (CSV)",
                data=csv_data,
                file_name=f"rekap_pemilahan_sampah_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )
        with col_clr:
            if st.button("🗑️ Hapus Riwayat Sesi"):
                st.session_state.scan_history = []
                st.rerun()