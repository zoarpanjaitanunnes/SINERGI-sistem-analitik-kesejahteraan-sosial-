from sklearn.decomposition import PCA
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

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

tab1, tab2, tab3= st.tabs(["🌍 TAHAP 1: PEMETAAN MAKRO (ALOKASI WILAYAH)", "🏠 TAHAP 2: PEMETAAN MIKRO (PRIORITAS KELUARGA)", "🤖 SINERGI AI: Auto-Optimizer"])

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

            if st.checkbox("✅ Pilih Semua Indikator", value=False, key="select_all"):
                st.session_state["makro"] = kolom_angka
                st.caption(f"📊 {len(kolom_angka)} indikator aktif — semua variabel BPS digunakan.")

            fitur_makro = st.multiselect(
                "Pilih Indikator Makro (Multi-dimensi) untuk dianalisis oleh AI:", 
                options=kolom_angka, 
                key="makro"
            )
        
        if len(fitur_makro) >= 2:
            if st.button("🚀 JALANKAN ANALISIS MAKRO", type="primary", use_container_width=True):
                with st.spinner('AI sedang memproses data...'):
                    df_clean = df_makro.dropna(subset=fitur_makro).copy()
                    
                    scaler_makro = StandardScaler()
                    X_makro_scaled = scaler_makro.fit_transform(df_clean[fitur_makro])
                    
                    kmeans_makro = KMeans(n_clusters=3, random_state=42, n_init=10)
                    df_clean['Klaster'] = kmeans_makro.fit_predict(X_makro_scaled)
                    main_feature = fitur_makro[0]
                    avg_val = df_clean.groupby('Klaster')[main_feature].mean()

                    is_positive_indicator = any(word in main_feature.upper() for word in ['IPM', 'INDEKS PEMBANGUNAN', 'PENGELUARAN'])

                    if is_positive_indicator:
                        klaster_stabil = avg_val.idxmax()
                        klaster_mendesak = avg_val.idxmin()
                    else:
                        klaster_mendesak = avg_val.idxmax()
                        klaster_stabil = avg_val.idxmin()

                    klaster_menengah = [i for i in [0, 1, 2] if i not in [klaster_mendesak, klaster_stabil]][0]

                    label_map = {
                        klaster_mendesak: 'Prioritas 1 (Intervensi Mendesak)',
                        klaster_menengah: 'Prioritas 2 (Intervensi Menengah)',
                        klaster_stabil: 'Prioritas 3 (Kondisi Stabil)'
                    }
                    
                    df_clean['Status Wilayah'] = df_clean['Klaster'].map(label_map)
                
                    sil_score_makro = silhouette_score(X_makro_scaled, df_clean['Klaster'])
                    st.session_state['makro_results'] = {
                        'df_clean': df_clean,
                        'X_scaled': X_makro_scaled,
                        'sil_score': sil_score_makro,
                        'fitur': fitur_makro,
                        'kmeans_obj': kmeans_makro
                    }
                    st.session_state['makro_jalan'] = True

            if st.session_state.get('makro_jalan'):
                res = st.session_state['makro_results']
                df_clean = res['df_clean']
                sil_score_makro = res['sil_score']
                fitur_res = res['fitur']

                col_acc1, col_acc2 = st.columns(2)
                with col_acc1:
                    st.metric("🎯 Akurasi Model (Silhouette Score)", f"{sil_score_makro:.2f}")
                with col_acc2:
                    status_ml = "Kuat" if sil_score_makro > 0.5 else "Cukup"
                    st.write(f"**Interpretasi AI:** Pengelompokan wilayah bersifat **{status_ml}**.")

                with st.expander("📊 Lihat Dasar Matematis Penentuan Jumlah Klaster (Elbow Method)"):
                    distortions = []
                    K_range = range(1, 8)
                    for k in K_range:
                        k_model = KMeans(n_clusters=k, random_state=42, n_init=10)
                        k_model.fit(df_clean[fitur_res])
                        distortions.append(k_model.inertia_)
                    
                    fig_elbow = px.line(x=list(K_range), y=distortions, markers=True, title="Elbow Method")
                    fig_elbow.update_layout(xaxis_title="Jumlah Klaster (K)", yaxis_title="Inertia/Distortion")
                    st.plotly_chart(fig_elbow, use_container_width=True)
                
                with st.container(border=True):
                    st.markdown("#### 📈 Hasil Klasterisasi Wilayah")
                    col_chart1, col_chart2 = st.columns(2)
                    with col_chart1:
                        fig_pie1 = px.pie(df_clean, names='Status Wilayah', hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
                        st.plotly_chart(fig_pie1, use_container_width=True)
                    with col_chart2:
                        fig_scatter1 = px.scatter(df_clean, x=fitur_res[0], y=fitur_res[1], color='Status Wilayah', hover_data=['Kab/Kota'], color_discrete_sequence=px.colors.qualitative.Pastel)
                        st.plotly_chart(fig_scatter1, use_container_width=True)
                    
                with st.container(border=True):
                    st.markdown("#### 📋 Direktori Prioritas Wilayah")
                    tab_p1, tab_p2, tab_p3 = st.tabs(["🔴 Prioritas 1", "🟡 Prioritas 2", "🟢 Prioritas 3"])
                    with tab_p1:
                        st.dataframe(df_clean[df_clean['Status Wilayah'] == 'Prioritas 1 (Intervensi Mendesak)'][['Provinsi', 'Kab/Kota', 'Status Wilayah']].reset_index(drop=True), use_container_width=True)
                    with tab_p2:
                        st.dataframe(df_clean[df_clean['Status Wilayah'] == 'Prioritas 2 (Intervensi Menengah)'][['Provinsi', 'Kab/Kota', 'Status Wilayah']].reset_index(drop=True), use_container_width=True)
                    with tab_p3:
                        st.dataframe(df_clean[df_clean['Status Wilayah'] == 'Prioritas 3 (Kondisi Stabil)'][['Provinsi', 'Kab/Kota', 'Status Wilayah']].reset_index(drop=True), use_container_width=True)


                st.markdown("---")
                st.subheader("📌 Insight Strategis Nasional")

                total_wilayah = len(df_clean)
                counts = df_clean['Status Wilayah'].value_counts()
                p1_pct = (counts.get('Prioritas 1 (Intervensi Mendesak)', 0) / total_wilayah) * 100
                p3_pct = (counts.get('Prioritas 3 (Kondisi Stabil)', 0) / total_wilayah) * 100

                col_stat1, col_stat2, col_stat3 = st.columns(3)
                col_stat1.metric("🔴 Tingkat Kerentanan Tinggi (Persentase wilayah yang masuk Prioritas 1)", f"{p1_pct:.1f}%", help="Persentase wilayah yang masuk Prioritas 1")
                col_stat2.metric("🟢 Tingkat Kemandirian (Persentase wilayah yang masuk Kondisi Stabil)", f"{p3_pct:.1f}%", help="Persentase wilayah yang masuk Kondisi Stabil")
                col_stat3.metric("📍 Total Titik Pantau (Total Wilayah)", f"{total_wilayah} Kab/Kota")

                with st.container(border=True):
                    col_info1, col_info2 = st.columns(2)
                    
                    with col_info1:
                        st.error("🚨 **Wilayah Konsentrasi Kemiskinan Tertinggi**")
                        top_p1_prov = df_clean[df_clean['Status Wilayah'] == 'Prioritas 1 (Intervensi Mendesak)']['Provinsi'].value_counts().idxmax()
                        count_p1 = df_clean[df_clean['Status Wilayah'] == 'Prioritas 1 (Intervensi Mendesak)']['Provinsi'].value_counts().max()
                        st.write(f"Provinsi **{top_p1_prov}** tercatat memiliki konsentrasi wilayah Prioritas 1 terbanyak ({count_p1} Kab/Kota).")
                        st.caption("Direkomendasikan untuk penambahan alokasi anggaran Bansos Makro.")

                    with col_info2:
                        st.success("✅ **Wilayah Paling Stabil**")
                        top_p3_prov = df_clean[df_clean['Status Wilayah'] == 'Prioritas 3 (Kondisi Stabil)']['Provinsi'].value_counts().idxmax()
                        count_p3 = df_clean[df_clean['Status Wilayah'] == 'Prioritas 3 (Kondisi Stabil)']['Provinsi'].value_counts().max()
                        st.write(f"Provinsi **{top_p3_prov}** menunjukkan performa kesejahteraan terbaik ({count_p3} Kab/Kota Stabil).")
                        st.caption("Dapat dijadikan referensi studi banding tata kelola ekonomi daerah.")
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
    def load_data_mikro_kpm():
        try:
            df = pd.read_csv('Dataset Mikro.csv', sep=',')
            return df
        except Exception as e:
            st.error(f"File 'Dataset Mikro.csv' tidak ditemukan. Error: {e}")
            return None
 
    df_mikro_raw = load_data_mikro_kpm()
 
    if df_mikro_raw is not None:
 
        # INFO 9 KRITERIA
        with st.expander("📋 Lihat 9 Kriteria Kemiskinan Kemensos yang Digunakan"):
            kriteria = {
                "K1 - Tempat Tinggal": "Tidak memiliki tempat berteduh/tinggal sehari-hari",
                "K2 - Pekerjaan": "Tidak bekerja atau tidak berpenghasilan tetap",
                "K3 - Konsumsi Pangan": "Pernah khawatir tidak makan atau pernah tidak makan dalam setahun terakhir",
                "K4 - Pengeluaran": "Pengeluaran pangan lebih besar dari 50% total pengeluaran",
                "K5 - Pakaian": "Tidak ada pengeluaran untuk pakaian selama 1 tahun terakhir",
                "K6 - Lantai Rumah": "Berlantai tanah atau plesteran",
                "K7 - Dinding Rumah": "Berdinding bambu, papan, seng, kardus, atau terpal",
                "K8 - Sanitasi": "Tidak memiliki jamban sendiri atau menggunakan jamban komunitas",
                "K9 - Listrik": "Daya listrik 450 watt atau bukan listrik",
            }
            for k, v in kriteria.items():
                st.markdown(f"**{k}:** {v}")
 
        with st.expander("🔍 Intip Dataset Mikro (200 Data Warga)"):
            st.dataframe(df_mikro_raw, use_container_width=True)
 
        with st.container(border=True):
            st.markdown("#### ⚡ Eksekusi Engine Penyeleksian KPM")
            st.markdown("Sistem akan menghitung **Skor Kemiskinan** tiap keluarga berdasarkan 9 kriteria Kemensos, lalu K-Means mengelompokkan secara otomatis.")
 
        if st.button("🚀 JALANKAN AI SELEKSI BANSOS", type="primary", use_container_width=True, key="btn_mikro_final"):
            with st.spinner('AI sedang memproses 9 kriteria kemiskinan per keluarga...'):
                df_m = df_mikro_raw.copy()
 
                # ========================
                # SCORING 9 KRITERIA KEMENSOS
                # Skor 1 = memenuhi kriteria miskin, 0 = tidak
                # ========================
 
                # K1: Tempat Tinggal
                df_m['Skor_K1'] = df_m['Tempat Tinggal'].apply(
                    lambda x: 1 if str(x).strip() in ['Tidak Punya', 'Menumpang'] else 0
                )
 
                # K2: Pekerjaan & Penghasilan
                df_m['Skor_K2'] = df_m['Penghasilan Tetap'].apply(
                    lambda x: 1 if str(x).strip() == 'Tidak' else 0
                )
 
                # K3: Konsumsi Pangan
                df_m['Skor_K3'] = df_m['Konsumsi Pangan'].apply(
                    lambda x: 1 if str(x).strip() in ['Pernah Tidak Makan', 'Khawatir Tidak Makan'] else 0
                )
 
                # K4: Pengeluaran Pangan > 50%
                df_m['Skor_K4'] = df_m['Pangan > 50% Pengeluaran'].apply(
                    lambda x: 1 if str(x).strip() == 'Ya' else 0
                )
 
                # K5: Pakaian
                df_m['Skor_K5'] = df_m['Beli Pakaian Setahun'].apply(
                    lambda x: 1 if str(x).strip() == 'Tidak' else 0
                )
 
                # K6: Lantai
                df_m['Skor_K6'] = df_m['Jenis Lantai'].apply(
                    lambda x: 1 if str(x).strip() in ['Tanah', 'Plesteran'] else 0
                )
 
                # K7: Dinding
                df_m['Skor_K7'] = df_m['Jenis Dinding'].apply(
                    lambda x: 1 if str(x).strip() in ['Bambu', 'Papan', 'Seng', 'Kardus', 'Terpal', 'Rumbia'] else 0
                )
 
                # K8: Sanitasi
                df_m['Skor_K8'] = df_m['Sanitasi'].apply(
                    lambda x: 1 if str(x).strip() in ['Tidak Ada', 'Komunitas'] else 0
                )
 
                # K9: Listrik
                df_m['Skor_K9'] = df_m['Listrik (VA)'].apply(
                    lambda x: 1 if str(x).strip() in ['450', 'Bukan Listrik'] else 0
                )
 
                # TOTAL SKOR (0-9)
                skor_cols = ['Skor_K1','Skor_K2','Skor_K3','Skor_K4','Skor_K5',
                             'Skor_K6','Skor_K7','Skor_K8','Skor_K9']
                df_m['Total_Skor_Kemiskinan'] = df_m[skor_cols].sum(axis=1)
 
                # ========================
                # K-MEANS CLUSTERING
                # ========================
                fitur_m = skor_cols + ['Total_Skor_Kemiskinan']
 
                scaler_m = StandardScaler()
                X_m_scaled = scaler_m.fit_transform(df_m[fitur_m])
 
                kmeans_m = KMeans(n_clusters=3, random_state=42, n_init=10)
                df_m['Klaster_M'] = kmeans_m.fit_predict(X_m_scaled)
 
                avg_skor = df_m.groupby('Klaster_M')['Total_Skor_Kemiskinan'].mean()
                id_p1 = avg_skor.idxmax()   
                id_p3 = avg_skor.idxmin()   
                id_p2 = [i for i in [0,1,2] if i not in [id_p1, id_p3]][0]
 
                mapping_m = {
                    id_p1: "Prioritas 1 (Sangat Layak Bantuan)",
                    id_p2: "Prioritas 2 (Rentan / Perlu Evaluasi)",
                    id_p3: "Prioritas 3 (Stabil/ Non-Target)"
                }
                df_m['Hasil_AI'] = df_m['Klaster_M'].map(mapping_m)
 
                # Simpan ke session state
                st.session_state['mikro_results'] = df_m
                st.session_state['mikro_jalan'] = True
 
            st.success("✅ Seleksi Berhasil! AI telah memproses 200 profil keluarga berdasarkan 9 Kriteria Kemensos.")
 
        # ========================
        # TAMPILKAN HASIL
        # ========================
        if st.session_state.get('mikro_jalan'):
            df_m = st.session_state['mikro_results']
            skor_cols = ['Skor_K1','Skor_K2','Skor_K3','Skor_K4','Skor_K5',
                         'Skor_K6','Skor_K7','Skor_K8','Skor_K9']
 
            # CHART
            with st.container(border=True):
                st.markdown("#### 📊 Visualisasi Hasil Klasterisasi")
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    st.plotly_chart(px.pie(
                        df_m, names='Hasil_AI',
                        title="Proporsi Kelayakan KPM",
                        color_discrete_map={
                            "Prioritas 1 (Sangat Layak Bantuan)": "#ef4444",
                            "Prioritas 2 (Rentan / Perlu Evaluasi)": "#f59e0b",
                            "Prioritas 3 (Stabil/ Non-Target)": "#10b981"
                        }
                    ), use_container_width=True)
                with col_m2:
                    st.plotly_chart(px.histogram(
                        df_m, x='Total_Skor_Kemiskinan',
                        color='Hasil_AI', barmode='overlay',
                        title="Distribusi Skor Kemiskinan (0-9 Kriteria)",
                        labels={'Total_Skor_Kemiskinan': 'Jumlah Kriteria Terpenuhi'},
                        color_discrete_map={
                            "Prioritas 1 (Sangat Layak Bantuan)": "#ef4444",
                            "Prioritas 2 (Rentan / Perlu Evaluasi)": "#f59e0b",
                            "Prioritas 3 (Stabil/ Non-Target)": "#10b981"
                        }
                    ), use_container_width=True)
 
            # RADAR CHART - Rata-rata skor per kriteria per klaster
            with st.container(border=True):
                st.markdown("#### 🕸️ Profil Kemiskinan Per Klaster (Radar Chart)")
                radar_data = df_m.groupby('Hasil_AI')[skor_cols].mean().reset_index()
                radar_melted = radar_data.melt(id_vars='Hasil_AI', var_name='Kriteria', value_name='Rata-rata Skor')
                label_map_radar = {
                    'Skor_K1': 'K1-Tempat Tinggal', 'Skor_K2': 'K2-Pekerjaan',
                    'Skor_K3': 'K3-Pangan', 'Skor_K4': 'K4-Pengeluaran',
                    'Skor_K5': 'K5-Pakaian', 'Skor_K6': 'K6-Lantai',
                    'Skor_K7': 'K7-Dinding', 'Skor_K8': 'K8-Sanitasi', 'Skor_K9': 'K9-Listrik'
                }
                radar_melted['Kriteria'] = radar_melted['Kriteria'].map(label_map_radar)
                fig_radar = px.line_polar(
                    radar_melted, r='Rata-rata Skor', theta='Kriteria',
                    color='Hasil_AI', line_close=True,
                    color_discrete_map={
                        "Prioritas 1 (Sangat Layak Bantuan)": "#ef4444",
                        "Prioritas 2 (Rentan / Perlu Evaluasi)": "#f59e0b",
                        "Prioritas 3 (Stabil/ Non-Target)": "#10b981"
                    }
                )
                fig_radar.update_traces(fill='toself', opacity=0.5)
                st.plotly_chart(fig_radar, use_container_width=True)
 
            # TABEL HASIL
            with st.container(border=True):
                st.markdown("#### 🎯 Hasil Rekomendasi Target KPM")
                kolom_tampil = ['Nama', 'Status Pekerjaan', 'Pengeluaran Perbulan',
                                'Jenis Lantai', 'Sanitasi', 'Listrik (VA)',
                                'Total_Skor_Kemiskinan', 'Hasil_AI']
 
                tab_res1, tab_res2, tab_res3 = st.tabs([
                    "🔴 Daftar Target P1", "🟡 Daftar Evaluasi P2", "🟢 Daftar Non-Target P3"
                ])
                with tab_res1:
                    df_p1 = df_m[df_m['Hasil_AI'] == "Prioritas 1 (Sangat Layak Bantuan)"][kolom_tampil].reset_index(drop=True)
                    st.caption(f"Total: {len(df_p1)} keluarga")
                    st.dataframe(df_p1, use_container_width=True)
                with tab_res2:
                    df_p2 = df_m[df_m['Hasil_AI'] == "Prioritas 2 (Rentan / Perlu Evaluasi)"][kolom_tampil].reset_index(drop=True)
                    st.caption(f"Total: {len(df_p2)} keluarga")
                    st.dataframe(df_p2, use_container_width=True)
                with tab_res3:
                    df_p3 = df_m[df_m['Hasil_AI'] == "Prioritas 3 (Stabil/ Non-Target)"][kolom_tampil].reset_index(drop=True)
                    st.caption(f"Total: {len(df_p3)} keluarga")
                    st.dataframe(df_p3, use_container_width=True)
 
            st.markdown("---")
            X_m_scaled_eval = StandardScaler().fit_transform(df_m[skor_cols + ['Total_Skor_Kemiskinan']])
            sil_m = silhouette_score(X_m_scaled_eval, df_m['Klaster_M'])
            c_met1, c_met2, c_met3 = st.columns(3)
            c_met1.metric("📊 Silhouette Score", f"{sil_m:.2f}")
            c_met2.metric("📋 Kriteria Digunakan", "9 Kriteria Kemensos")
            c_met3.metric("👥 Total KK Diproses", f"{len(df_m)} Keluarga")

# ==========================================
# TAB 3: EVALUASI & AUTO-OPTIMIZER AI (TANPA PCA)
# ==========================================
with tab3:
    st.subheader("💡 SINERGI AI: Auto-Optimizer")
    st.markdown("Mesin ini akan mencari kombinasi indikator yang menghasilkan **Akurasi (Silhouette Score) tertinggi** pada data asli.")
    
    if st.button("🔍 CARI KOMBINASI INDIKATOR TERBAIK", type="primary", use_container_width=True):
        import itertools
        
        cols = kolom_angka
        best_score = -1
        best_combination = []
        
        progress_bar = st.progress(0)
        all_combos = []
        for r in range(2, 5):
            all_combos.extend(list(itertools.combinations(cols, r)))
            
        total_comb = len(all_combos)
        
        with st.spinner(f'Mengevaluasi {total_comb} kombinasi data...'):
            for i, combo in enumerate(all_combos):
                df_temp = df_makro.dropna(subset=list(combo))
                if len(df_temp) > 10:
                    scaler_temp = StandardScaler()
                    X_scaled_temp = scaler_temp.fit_transform(df_temp[list(combo)])
                    
                    km_temp = KMeans(n_clusters=3, random_state=42, n_init=5)
                    labels_temp = km_temp.fit_predict(X_scaled_temp)
                    
                    score_temp = silhouette_score(X_scaled_temp, labels_temp)
                    
                    if score_temp > best_score:
                        best_score = score_temp
                        best_combination = combo
                
                progress_bar.progress((i + 1) / total_comb)
        
        st.session_state['best_combo'] = {
            'score': best_score,
            'features': best_combination
        }
        st.balloons()

    if 'best_combo' in st.session_state:
        res_best = st.session_state['best_combo']
        with st.container(border=True):
            st.success(f"### 🎉 Kombinasi Akurasi Tertinggi Ditemukan!")
            col_res1, col_res2 = st.columns([1, 2])
            with col_res1:
                st.metric("Skor Maksimal", f"{res_best['score']:.4f}")
            with col_res2:
                st.write("**Gunakan Indikator ini di Tab 1:**")
                for f in res_best['features']:
                    st.write(f"- {f}")
            