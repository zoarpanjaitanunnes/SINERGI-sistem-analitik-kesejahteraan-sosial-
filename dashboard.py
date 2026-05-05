import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans

# --- PENGATURAN HALAMAN & CUSTOM CSS ---
st.set_page_config(page_title="SINERGI Kesejahteraan", page_icon="✨", layout="wide", initial_sidebar_state="collapsed")

# Injeksi CSS biar metrik dan tampilannya lebih "Figma" banget
st.markdown("""
<style>
    /* Styling untuk Metric Box */
    [data-testid="stMetric"] {
        background-color: #1E293B;
        border-radius: 10px;
        padding: 15px 20px;
        border-left: 5px solid #3B82F6;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    [data-testid="stMetricValue"] {
        font-size: 28px !important;
        font-weight: bold;
    }
    /* Ganti warna garis pembatas */
    hr {
        border-color: #334155;
    }
</style>
""", unsafe_allow_html=True)

# HEADER DASHBOARD
with st.container(border=True):
    col_logo, col_title = st.columns([1, 11])
    with col_logo:
        st.image("https://cdn-icons-png.flaticon.com/512/3135/3135679.png", width=60) # Icon garuda/gov (dummy)
    with col_title:
        st.title("SINERGI: Sistem Analitik Kesejahteraan Sosial")
        st.markdown("*Otomatisasi Penyeleksian Prioritas Bantuan Sosial Menggunakan Machine Learning (K-Means Clustering)*")

st.markdown("<br>", unsafe_allow_html=True)

# BIKIN TAB ELEGAN
tab1, tab2 = st.tabs(["🌍 TAHAP 1: PEMETAAN MAKRO (ALOKASI WILAYAH)", "🏠 TAHAP 2: PEMETAAN MIKRO (PRIORITAS KELUARGA)"])

# ==========================================
# TAB 1: ANALISIS MAKRO
# ==========================================
with tab1:
    st.info("💡 **Objective:** Sistem menganalisis data agregat makro untuk merekomendasikan Kabupaten/Kota yang paling membutuhkan intervensi program kesejahteraan.")
    
    @st.cache_data
    def load_data_makro():
        df = pd.read_csv(r'Klasifikasi Tingkat Kemiskinan di Indonesia.csv', sep=';')
        for col in df.columns:
            if col not in ['Provinsi', 'Kab/Kota']:
                df[col] = df[col].astype(str).str.replace(',', '.', regex=False).str.strip()
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    
    try:
        df_makro = load_data_makro()
        kolom_angka = df_makro.select_dtypes(include=['float64', 'int64']).columns.tolist()
        if 'Klasifikasi Kemiskinan' in kolom_angka: 
            kolom_angka.remove('Klasifikasi Kemiskinan')
            
        target_default = ['Persentase Penduduk Miskin (P0) Menurut Kabupaten/Kota (Persen)', 'Indeks Pembangunan Manusia']
        default_pilihan = [col for col in target_default if col in kolom_angka]
        if len(default_pilihan) < 2 and len(kolom_angka) >= 2:
            default_pilihan = kolom_angka[:2]

        with st.container(border=True):
            st.markdown("#### ⚙️ Konfigurasi Parameter Analisis")
            fitur_makro = st.multiselect(
                "Pilih Indikator Makro (Multi-dimensi) untuk dianalisis oleh AI:", 
                options=kolom_angka, 
                default=default_pilihan, 
                key="makro"
            )
        
        if len(fitur_makro) >= 2:
            df_clean = df_makro.dropna(subset=fitur_makro).copy()
            kmeans_makro = KMeans(n_clusters=3, random_state=42)
            df_clean['Klaster'] = kmeans_makro.fit_predict(df_clean[fitur_makro])
            
            label_map = {0: 'Prioritas 2 (Intervensi Menengah)', 1: 'Prioritas 1 (Intervensi Mendesak)', 2: 'Prioritas 3 (Kondisi Stabil)'}
            df_clean['Status Wilayah'] = df_clean['Klaster'].map(label_map)
            
            # BUNGKUS CHART DALAM KOTAK (CARD UI)
            with st.container(border=True):
                st.markdown("#### 📈 Hasil Klasterisasi Wilayah")
                col_chart1, col_chart2 = st.columns(2)
                with col_chart1:
                    fig_pie1 = px.pie(df_clean, names='Status Wilayah', hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
                    fig_pie1.update_layout(paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=20, b=20, l=20, r=20))
                    st.plotly_chart(fig_pie1, use_container_width=True)
                with col_chart2:
                    fig_scatter1 = px.scatter(df_clean, x=fitur_makro[0], y=fitur_makro[1], color='Status Wilayah', hover_data=['Kab/Kota'], color_discrete_sequence=px.colors.qualitative.Pastel)
                    fig_scatter1.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=20, b=20, l=20, r=20))
                    st.plotly_chart(fig_scatter1, use_container_width=True)
                
            # BUNGKUS TABEL DALAM KOTAK
            with st.container(border=True):
                st.markdown("#### 📋 Direktori Prioritas Wilayah")
                tab_p1, tab_p2, tab_p3 = st.tabs(["🔴 Prioritas 1 (Mendesak)", "🟡 Prioritas 2 (Menengah)", "🟢 Prioritas 3 (Stabil)"])
                with tab_p1:
                    st.dataframe(df_clean[df_clean['Status Wilayah'] == 'Prioritas 1 (Intervensi Mendesak)'][['Provinsi', 'Kab/Kota', 'Status Wilayah']].reset_index(drop=True), use_container_width=True)
                with tab_p2:
                    st.dataframe(df_clean[df_clean['Status Wilayah'] == 'Prioritas 2 (Intervensi Menengah)'][['Provinsi', 'Kab/Kota', 'Status Wilayah']].reset_index(drop=True), use_container_width=True)
                with tab_p3:
                    st.dataframe(df_clean[df_clean['Status Wilayah'] == 'Prioritas 3 (Kondisi Stabil)'][['Provinsi', 'Kab/Kota', 'Status Wilayah']].reset_index(drop=True), use_container_width=True)
                
        else:
            st.warning("⚠️ Silakan pilih minimal 2 indikator makro.")
            
    except Exception as e:
        st.error(f"Sistem gagal mengeksekusi: {e}")


# ==========================================
# TAB 2: ANALISIS MIKRO
# ==========================================
with tab2:
    st.info("💡 **Objective:** Menganalisis data mikro (Survei Lapangan/DTKS) untuk menyeleksi rumah tangga sasaran (KPM) secara transparan dan objektif.")
    
    @st.cache_data
    def generate_data_keluarga():
        np.random.seed(42)
        n_data = 150 
        data = {
            'No_KK': [f"33740{str(i).zfill(4)}" for i in range(1, n_data+1)],
            'Pendapatan_Bulanan_Rp': np.random.randint(500000, 8000000, n_data),
            'Jumlah_Tanggungan_Jiwa': np.random.randint(1, 7, n_data),
            'Kapasitas_Listrik_VA': np.random.choice([450, 900, 1300, 2200], n_data),
            'Skor_Fisik_Bangunan': np.random.randint(1, 10, n_data) 
        }
        return pd.DataFrame(data)
    
    df_mikro = generate_data_keluarga()
    
    # METRIC DASHBOARD
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("👥 Total Data DTKS", f"{len(df_mikro)} Kepala Keluarga")
    col_m2.metric("📍 Lokasi Survei", "Kota Semarang")
    col_m3.metric("🧠 Status Mesin AI", "Siap Memproses 🟢")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    with st.expander("🔍 Intip Sampel Data Mentah DTKS Warga (Sebelum Diproses AI)"):
        st.dataframe(df_mikro.head(10), use_container_width=True)
    
    with st.container(border=True):
        st.markdown("#### ⚡ Eksekusi Engine Penyeleksian KPM")
        st.markdown("Klik tombol di bawah untuk menjalankan algoritma *Clustering* pada data lapangan.")
        
        if st.button("🚀 JALANKAN AI SELEKSI BANSOS", type="primary", use_container_width=True):
            with st.spinner('Memproses pola kemiskinan multi-variabel...'):
                fitur_mikro = ['Pendapatan_Bulanan_Rp', 'Jumlah_Tanggungan_Jiwa', 'Kapasitas_Listrik_VA', 'Skor_Fisik_Bangunan']
                
                kmeans_mikro = KMeans(n_clusters=3, random_state=42)
                df_mikro['Kode_Klaster'] = kmeans_mikro.fit_predict(df_mikro[fitur_mikro])
                
                avg_pendapatan = df_mikro.groupby('Kode_Klaster')['Pendapatan_Bulanan_Rp'].mean()
                klaster_rentan = avg_pendapatan.idxmin()
                klaster_mandiri = avg_pendapatan.idxmax()
                
                def tentukan_status(k):
                    if k == klaster_rentan: return "Prioritas 1 (Sangat Rentan)"
                    elif k == klaster_mandiri: return "Prioritas 3 (Mandiri / Tidak Layak Bansos)"
                    else: return "Prioritas 2 (Rentan)"
                    
                df_mikro['Status_Rekomendasi_AI'] = df_mikro['Kode_Klaster'].apply(tentukan_status)
                
                # BUNGKUS CHART MIKRO DALAM KOTAK
                st.markdown("<br>", unsafe_allow_html=True)
                with st.container(border=True):
                    col_chart3, col_chart4 = st.columns(2)
                    with col_chart3:
                        fig_pie2 = px.pie(df_mikro, names='Status_Rekomendasi_AI', title="Sebaran Tingkat Kesejahteraan", color_discrete_map={
                            "Prioritas 1 (Sangat Rentan)": "#ef4444", "Prioritas 2 (Rentan)": "#f59e0b", "Prioritas 3 (Mandiri / Tidak Layak Bansos)": "#10b981"
                        })
                        fig_pie2.update_layout(paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=40, b=20, l=20, r=20))
                        st.plotly_chart(fig_pie2, use_container_width=True)
                    with col_chart4:
                        fig_scatter2 = px.scatter(
                            df_mikro, x='Skor_Fisik_Bangunan', y='Pendapatan_Bulanan_Rp', 
                            color='Status_Rekomendasi_AI', size='Jumlah_Tanggungan_Jiwa', hover_data=['No_KK', 'Kapasitas_Listrik_VA'],
                            title="Peta Ekonomi Rumah Tangga",
                            color_discrete_map={"Prioritas 1 (Sangat Rentan)": "#ef4444", "Prioritas 2 (Rentan)": "#f59e0b", "Prioritas 3 (Mandiri / Tidak Layak Bansos)": "#10b981"}
                        )
                        fig_scatter2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=40, b=20, l=20, r=20))
                        st.plotly_chart(fig_scatter2, use_container_width=True)
                    
                st.success("✅ **Sistem Selesai!** AI telah berhasil mengklasifikasikan status kelayakan seluruh rumah tangga.")
                
                # BUNGKUS TABEL MIKRO DALAM KOTAK
                with st.container(border=True):
                    st.markdown("#### 🎯 Hasil Rekomendasi Target KPM (Keluarga Penerima Manfaat)")
                    
                    df_mikro_display = df_mikro.copy()
                    df_mikro_display['Pendapatan_Bulanan_Rp'] = df_mikro_display['Pendapatan_Bulanan_Rp'].apply(lambda x: f"Rp {x:,.0f}".replace(",", "."))
                    kolom_tampil = ['No_KK', 'Pendapatan_Bulanan_Rp', 'Jumlah_Tanggungan_Jiwa', 'Kapasitas_Listrik_VA', 'Skor_Fisik_Bangunan', 'Status_Rekomendasi_AI']
                    
                    tab_m1, tab_m2, tab_m3 = st.tabs(["🔴 Target Stiker: Prioritas 1", "🟡 Evaluasi: Prioritas 2", "🟢 Non-Target: Prioritas 3"])
                    
                    with tab_m1:
                        st.dataframe(df_mikro_display[df_mikro_display['Status_Rekomendasi_AI'] == "Prioritas 1 (Sangat Rentan)"][kolom_tampil].reset_index(drop=True), use_container_width=True)
                    with tab_m2:
                        st.dataframe(df_mikro_display[df_mikro_display['Status_Rekomendasi_AI'] == "Prioritas 2 (Rentan)"][kolom_tampil].reset_index(drop=True), use_container_width=True)
                    with tab_m3:
                        st.dataframe(df_mikro_display[df_mikro_display['Status_Rekomendasi_AI'] == "Prioritas 3 (Mandiri / Tidak Layak Bansos)"][kolom_tampil].reset_index(drop=True), use_container_width=True)