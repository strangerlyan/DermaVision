import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from PIL import Image

from tensorflow.keras.applications.efficientnet import preprocess_input

# ======================
# KONFIGURASI HALAMAN
# ======================

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

# Path logo
LOGO_PATH = os.path.join(BASE_PATH, "logo fix.png")

# Cek logo
def is_logo_valid():
    try:
        if os.path.exists(LOGO_PATH):
            img = Image.open(LOGO_PATH)
            img.verify()
            return True
        return False
    except Exception:
        return False

LOGO_VALID = is_logo_valid()

try:
    if LOGO_VALID:
        logo_icon = Image.open(LOGO_PATH)
        logo_icon.thumbnail((32, 32))
        st.set_page_config(
            page_title="DermaVision - Deteksi Kanker Kulit Sejak Dini!",
            page_icon=logo_icon,
            layout="wide",
            initial_sidebar_state="expanded"
        )
    else:
        st.set_page_config(
            page_title="DermaVision - Deteksi Kanker Kulit Sejak Dini!",
            page_icon="🧬",
            layout="wide",
            initial_sidebar_state="expanded"
        )
except Exception:
    st.set_page_config(
        page_title="DermaVision - Deteksi Kanker Kulit Sejak Dini!",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

# ======================
# PATH FILE (SESUAI FOLDER ANDA)
# ======================
MODEL_PATH = os.path.join(BASE_PATH, "cnn_ham10000_final2.keras")
CLASS_PATH = os.path.join(BASE_PATH, "class_names.npy")
META_PATH = os.path.join(BASE_PATH, "meta_cols.npy")

# ======================
# LOAD MODEL
# ======================

# ======================
# PREPROCESS
# ======================

def preprocess(img):
    img = img.convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img)
    img = np.expand_dims(img, 0)
    img = preprocess_input(img.astype("float32"))
    return img

# ======================
# METADATA
# ======================

def build_metadata(age, sex, localization, meta_cols):
    meta = np.zeros((1, len(meta_cols)), dtype="float32")
    
    meta[0, 0] = age
    
    if sex == "Laki-laki":
        meta[0, 1] = 0
    else:
        meta[0, 1] = 1
    
    loc = "loc_" + localization
    if loc in meta_cols:
        idx = meta_cols.index(loc)
        meta[0, idx] = 1
    
    return meta

# ======================
# FUNGSI LOAD LOKASI DARI CSV
# ======================

@st.cache_data
def load_locations_from_csv():
    METADATA_CSV = os.path.join(BASE_PATH, "HAM10000_metadata.csv")
    if os.path.exists(METADATA_CSV):
        try:
            df = pd.read_csv(METADATA_CSV)
            if 'localization' in df.columns:
                locations = sorted(df['localization'].dropna().unique())
                return locations
        except Exception as e:
            st.sidebar.warning(f"⚠️ Gagal membaca metadata CSV: {e}")
    
    return [
        "abdomen", "acral", "back", "chest", "ear", "face", "foot",
        "genital", "hand", "head", "lower extremity", "neck", "scalp",
        "trunk", "unknown", "upper extremity"
    ]

def get_location_label(location_en):
    location_map = {
        "abdomen": "Perut",
        "acral": "Akral (Ujung Jari/Kaki)",
        "back": "Punggung",
        "chest": "Dada",
        "ear": "Telinga",
        "face": "Wajah",
        "foot": "Kaki",
        "genital": "Area Kelamin",
        "hand": "Tangan",
        "head": "Kepala",
        "lower extremity": "Ekstremitas Bawah",
        "neck": "Leher",
        "scalp": "Kulit Kepala",
        "trunk": "Badan (Trunkus)",
        "unknown": "Tidak Diketahui",
        "upper extremity": "Ekstremitas Atas"
    }
    return location_map.get(location_en, location_en)

# ======================
# DAFTAR KELAS DAN INFORMASI
# ======================

CLASS_INFO = {
    "akiec": {
        "nama_lengkap": "Actinic Keratoses dan Intraepithelial Carcinoma",
        "deskripsi": "Lesi prakanker yang berpotensi berkembang menjadi karsinoma sel skuamosa apabila tidak ditangani.",
        "kategori": "Ganas"
    },
    "bcc": {
        "nama_lengkap": "Basal Cell Carcinoma",
        "deskripsi": "Kanker kulit ganas yang paling umum ditemukan, namun memiliki tingkat metastasis yang relatif rendah.",
        "kategori": "Ganas"
    },
    "bkl": {
        "nama_lengkap": "Benign Keratosis-like Lesions",
        "deskripsi": "Kelompok lesi jinak yang mencakup berbagai kelainan seperti seborrheic keratosis dan solar lentigo.",
        "kategori": "Jinak"
    },
    "df": {
        "nama_lengkap": "Dermatofibroma",
        "deskripsi": "Tumor kulit jinak yang umumnya tidak berbahaya.",
        "kategori": "Jinak"
    },
    "mel": {
        "nama_lengkap": "Melanoma",
        "deskripsi": "Jenis kanker kulit yang paling berbahaya karena memiliki kemampuan menyebar ke organ lain dan menyebabkan kematian apabila tidak dideteksi sejak dini.",
        "kategori": "Ganas"
    },
    "nv": {
        "nama_lengkap": "Melanocytic Nevus",
        "deskripsi": "Tahi lalat atau lesi pigmen yang umumnya bersifat jinak dan merupakan kelas dengan jumlah data terbesar pada dataset HAM10000.",
        "kategori": "Jinak"
    },
    "vasc": {
        "nama_lengkap": "Vascular Lesions",
        "deskripsi": "Kelainan kulit yang berkaitan dengan pembuluh darah dan umumnya bersifat jinak.",
        "kategori": "Jinak"
    }
}

# ======================
# SIDEBAR
# ======================

def render_sidebar():
    try:
        if LOGO_VALID and os.path.exists(LOGO_PATH):
            st.sidebar.image(LOGO_PATH, width=100)
        else:
            st.sidebar.image("https://img.icons8.com/fluency/96/000000/skin.png", width=80)
    except Exception:
        st.sidebar.image("https://img.icons8.com/fluency/96/000000/skin.png", width=80)
    
    st.sidebar.markdown(
        """
        <h2 style='text-align:center; color:#2c3e50;'>
            DermaVision
        </h2>
        <p style='text-align:center; color:#7f8c8d; font-size:14px;'>
            Inovasi Skrining Kanker Kulit<br>Berbasis CNN & Transfer Learning
        </p>
        """, 
        unsafe_allow_html=True
    )
    st.sidebar.markdown("---")
    
    tab_options = ["🩺 Deteksi Kulit", "📊 Rekap & Visualisasi", "📋 Riwayat & Download"]
    selected_tab = st.sidebar.radio("Navigasi", tab_options, index=0)
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        """
        **📌 Informasi:**
        - Model: CNN Transfer Learning
        - Dataset: HAM10000
        - 7 Kelas Lesi Kulit
        """
    )
    
    return selected_tab

# ======================
# HEADER
# ======================

def render_header():
    st.markdown(
        """
        <h1 style='text-align:center; color:#2c3e50;'>
            DermaVision
        </h1>
        <h4 style='text-align:center; color:#7f8c8d; font-weight:normal;'>
            Inovasi Skrining Kanker Kulit Berbasis CNN & Transfer Learning
        </h4>
        <hr>
        """, 
        unsafe_allow_html=True
    )

# ======================
# TAB 1: DETEKSI KULIT
# ======================

def render_detection_tab(model, class_names, meta_cols):
    render_header()
    
    st.markdown("### 📝 Input Data Pasien")
    
    locations_en = load_locations_from_csv()
    locations_id = ["Pilih lokasi"] + [get_location_label(loc) for loc in locations_en]
    location_mapping = {get_location_label(loc): loc for loc in locations_en}
    
    with st.form("detection_form", clear_on_submit=False):
        col1, col2 = st.columns(2)
        
        with col1:
            nama = st.text_input("👤 Nama Pasien", placeholder="Masukkan nama lengkap")
            usia = st.number_input("🎂 Usia (Tahun)", min_value=0, max_value=120, step=1, value=30)
            jenis_kelamin = st.selectbox("⚧️ Jenis Kelamin", ["Pilih", "Laki-laki", "Perempuan"])
        
        with col2:
            lokasi_label = st.selectbox("📍 Letak Anomali Kulit", locations_id)
            uploaded_file = st.file_uploader(
                "📷 Upload Foto Anomali Kulit (JPG/JPEG/PNG, max 2MB)",
                type=["jpg", "jpeg", "png"],
                help="Upload gambar lesi kulit. Format: JPG, JPEG, PNG. Maksimal 2MB."
            )
        
        submitted = st.form_submit_button("🔍 Deteksi Sekarang", use_container_width=True, type="primary")
    
    if submitted:
        errors = []
        if not nama.strip():
            errors.append("Nama harus diisi")
        if jenis_kelamin == "Pilih":
            errors.append("Jenis kelamin harus dipilih")
        if lokasi_label == "Pilih lokasi":
            errors.append("Letak anomali harus dipilih")
        if uploaded_file is None:
            errors.append("Foto anomali kulit harus diupload")
        
        if errors:
            for err in errors:
                st.error(f"❌ {err}")
            return
        
        lokasi_value = location_mapping.get(lokasi_label, lokasi_label)
        
        if uploaded_file is not None:
            file_size = uploaded_file.size
            if file_size > 2 * 1024 * 1024:
                st.error(f"❌ Ukuran file terlalu besar: {file_size / 1024 / 1024:.2f} MB. Maksimal 2 MB.")
                return
        
        if model is None or class_names is None or meta_cols is None:
            st.error("❌ Model belum dimuat. Silakan periksa file model.")
            return
        
        try:
            img = Image.open(uploaded_file)
            
            st.markdown("### 🖼️ Gambar yang Diupload")
            col_img1, col_img2, col_img3 = st.columns([1, 2, 1])
            with col_img2:
                st.image(img, caption="Foto Anomali Kulit", use_container_width=True)
            
            # Preprocessing
            img_array = preprocess(img)
            if img_array is None:
                st.error("❌ Gagal preprocessing gambar")
                return
            
            # Build metadata
            meta = build_metadata(usia, jenis_kelamin, lokasi_value, meta_cols)
            
            # ============ PREDIKSI ============
            with st.spinner("🔄 Menganalisis gambar menggunakan CNN..."):
                pred = model.predict([img_array, meta], verbose=0)
                pred = np.nan_to_num(pred)
                st.write("Debug Prediksi Raw:", pred)
                
                st.write("Prediction Output")
                st.write(pred)

                st.write("NaN:", np.isnan(pred).any())
                st.write("Inf:", np.isinf(pred).any())
            # =================================
            
            idx = np.argmax(pred[0])
            confidence = float(np.nan_to_num(pred[0][idx], nan=0.0)) * 100
            predicted_class = class_names[idx]
            
            st.markdown("### 📊 Hasil Deteksi")
            #st.balloons()
            
            if predicted_class in ["akiec", "bcc", "mel"]:
                kategori = "Ganas"
                card_color = "#e74c3c"
                emoji = "⚠️"
            elif predicted_class in ["bkl", "df", "nv", "vasc"]:
                kategori = "Jinak"
                card_color = "#3498db"
                emoji = "🟦"
            else:
                kategori = "Normal"
                card_color = "#2ecc71"
                emoji = "✅"
            
            class_info = CLASS_INFO.get(predicted_class, {})
            nama_penyakit = class_info.get("nama_lengkap", predicted_class)
            deskripsi = class_info.get("deskripsi", "")
            
            col_res1, col_res2 = st.columns([1, 2])
            
            with col_res1:
                st.markdown(
                    f"""
                    <div style="
                        background-color: {card_color}20;
                        padding: 20px;
                        border-radius: 10px;
                        border-left: 5px solid {card_color};
                    ">
                        <h3>{emoji} Kategori: {kategori}</h3>
                        <p><b>Kelas:</b> {predicted_class}</p>
                        <p><b>Keyakinan:</b> {confidence:.2f}%</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            with col_res2:
                if kategori == "Normal":
                    st.success(
                        """
                        **✅ Hasil Deteksi: Kulit Normal**
                        
                        Tidak terdeteksi tanda-tanda tumor ataupun kanker kulit pada area yang diperiksa.
                        
                        Namun, tetap jaga kesehatan kulit dengan:
                        - Gunakan tabir surya setiap hari
                        - Hindari paparan sinar matahari berlebihan
                        - Lakukan pemeriksaan kulit rutin
                        """
                    )
                else:
                    st.markdown(
                        f"""
                        <div style="
                            background-color: #f8f9fa;
                            padding: 15px;
                            border-radius: 10px;
                        ">
                            <h4>📋 {nama_penyakit}</h4>
                            <p>{deskripsi}</p>
                            <p><b>Rekomendasi:</b> Segera konsultasikan dengan dokter spesialis kulit (dermatologis) untuk penanganan lebih lanjut.</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            
            st.markdown("### 📊 Probabilitas per Kelas")
            
            prob_df = pd.DataFrame({
                "Kelas": class_names,
                "Probabilitas (%)": pred[0] * 100
            })
            prob_df = prob_df.sort_values("Probabilitas (%)", ascending=False)
            
            color_map = {
                "akiec": "#e74c3c",
                "bcc": "#e67e22",
                "bkl": "#3498db",
                "df": "#1abc9c",
                "mel": "#c0392b",
                "nv": "#2ecc71",
                "vasc": "#9b59b6"
            }
            colors = [color_map.get(k, "#95a5a6") for k in prob_df["Kelas"]]
            
            fig, ax = plt.subplots(figsize=(10, 4))
            bars = ax.bar(prob_df["Kelas"], prob_df["Probabilitas (%)"], color=colors)
            
            for bar, val in zip(bars, prob_df["Probabilitas (%)"]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                        f"{val:.1f}%", ha="center", va="bottom", fontweight="bold")
            
            ax.set_ylabel("Probabilitas (%)")
            ax.set_xlabel("Kelas Lesi Kulit")
            ax.set_title("Distribusi Probabilitas Prediksi", fontweight="bold")
            ax.set_ylim(0, 105)
            ax.grid(axis="y", linestyle="--", alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.success(f"✅ Hasil: {predicted_class} dengan confidence {confidence:.2f}%")
            
        except Exception as e:
            st.error(f"❌ Terjadi kesalahan: {e}")
            import traceback
            st.code(traceback.format_exc())

# ======================
# TAB 2: REKAP & VISUALISASI
# ======================

def render_visualization_tab():
    st.markdown(
        """
        <h1 style='text-align:center; color:#2c3e50;'>
            📊 Rekap & Visualisasi Data Deteksi
        </h1>
        <hr>
        """,
        unsafe_allow_html=True
    )
    st.info("ℹ️ Fitur ini akan menampilkan rekap dan visualisasi dari riwayat deteksi.")
    st.warning("⚠️ Belum ada data deteksi. Silakan lakukan deteksi terlebih dahulu pada tab 'Deteksi Kulit'.")

# ======================
# TAB 3: RIWAYAT & DOWNLOAD
# ======================

def render_history_tab():
    st.markdown(
        """
        <h1 style='text-align:center; color:#2c3e50;'>
            📋 Riwayat Deteksi & Download Data
        </h1>
        <hr>
        """,
        unsafe_allow_html=True
    )
    st.info("ℹ️ Fitur ini akan menampilkan riwayat deteksi dan download data.")
    st.warning("⚠️ Belum ada data deteksi. Silakan lakukan deteksi terlebih dahulu pada tab 'Deteksi Kulit'.")

# ======================
# MAIN PROGRAM
# ======================

def main():
    # Load model dan data
    model, class_names, meta_cols = load_model_and_data()
    
    if model is None or class_names is None or meta_cols is None:
        st.error("""
        ### ❌ Gagal Memuat Model atau Data Pendukung
            
        Pastikan file-file berikut ada di path yang benar:
        - `cnn_ham10000_final2.keras`
        - `class_names.npy`
        - `meta_cols.npy`
            
        **Path yang digunakan:** `{}`
        """.format(BASE_PATH))
        return
    
    # Render sidebar dan pilih tab
    selected_tab = render_sidebar()
    
    if selected_tab == "🩺 Deteksi Kulit":
        render_detection_tab(model, class_names, meta_cols)
    elif selected_tab == "📊 Rekap & Visualisasi":
        render_visualization_tab()
    elif selected_tab == "📋 Riwayat & Download":
        render_history_tab()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <p style='text-align:center; color:#7f8c8d; font-size:12px;'>
            DermaVision © 2025 | Inovasi Skrining Kanker Kulit<br>Berbasis CNN & Transfer Learning<br>
            Dataset: HAM10000
        </p>
        """,
        unsafe_allow_html=True
    )

# ======================
# EKSEKUSI
# ======================

if __name__ == "__main__":
    main()