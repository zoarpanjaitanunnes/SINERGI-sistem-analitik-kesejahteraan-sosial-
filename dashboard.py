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

        with st.container(border=True):
            st.markdown("#### ⚙️ Konfigurasi Parameter Analisis")

            fitur_makro = st.multiselect(
                "Pilih Indikator Makro (Multi-dimensi) untuk dianalisis oleh AI:", 
                options=kolom_angka,
                help="Pilih minimal 2 indikator untuk melakukan clustering",
                key="makro"
            )

            jumlah_klaster = st.slider(
                "🔢 Jumlah Klaster (K):",
                min_value=2,
                max_value=6,
                value=3,
                help="K=2 paling sederhana, K=3 standar, K=4-6 lebih detail tapi skor bisa turun"
            )

            info_k = {
                2: "K=2: Wilayah dibagi jadi **Layak Bantuan** dan **Mandiri**. Dilengkapi ranking internal untuk menentukan urutan prioritas.",
                3: "K=3: Standar — **Mendesak, Menengah, Stabil**. Paling umum dipakai.",
                4: "K=4: Lebih detail — membedakan tingkat menengah jadi 2 sub-kelompok.",
                5: "K=5: Sangat detail — cocok untuk analisis mendalam.",
                6: "K=6: Paling granular — pastikan datamu cukup besar untuk ini.",
            }
            st.caption(info_k.get(jumlah_klaster, ""))

        if len(fitur_makro) >= 2:
            if st.button("🚀 JALANKAN ANALISIS MAKRO", type="primary", use_container_width=True):
                with st.spinner('AI sedang memproses data...'):
                    df_clean = df_makro.dropna(subset=fitur_makro).copy()
                    
                    scaler_makro = StandardScaler()
                    X_makro_scaled = scaler_makro.fit_transform(df_clean[fitur_makro])
                    
                    kmeans_makro = KMeans(n_clusters=jumlah_klaster, random_state=42, n_init=10)
                    df_clean['Klaster'] = kmeans_makro.fit_predict(X_makro_scaled)

                    main_feature = fitur_makro[0]
                    avg_val = df_clean.groupby('Klaster')[main_feature].mean()
                    is_positive_indicator = any(word in main_feature.upper() for word in ['IPM', 'INDEKS PEMBANGUNAN', 'PENGELUARAN'])

                    if is_positive_indicator:
                        urutan_klaster = avg_val.sort_values(ascending=True).index.tolist()
                    else:
                        urutan_klaster = avg_val.sort_values(ascending=False).index.tolist()

                    def buat_label(jumlah_klaster):
                        if jumlah_klaster == 2:
                            labels = ['Layak Menerima Bantuan', 'Mandiri / Tidak Layak']
                        elif jumlah_klaster == 3:
                            labels = ['Prioritas 1 (Intervensi Mendesak)', 'Prioritas 2 (Intervensi Menengah)', 'Prioritas 3 (Kondisi Stabil)']
                        elif jumlah_klaster == 4:
                            labels = ['Prioritas 1 (Sangat Mendesak)', 'Prioritas 2 (Mendesak)', 'Prioritas 3 (Menengah)', 'Prioritas 4 (Kondisi Stabil)']
                        elif jumlah_klaster == 5:
                            labels = ['Prioritas 1 (Sangat Mendesak)', 'Prioritas 2 (Mendesak)', 'Prioritas 3 (Menengah)', 'Prioritas 4 (Cukup Stabil)', 'Prioritas 5 (Kondisi Stabil)']
                        else:
                            labels = [f'Prioritas {i+1}' for i in range(jumlah_klaster)]
                        return {urutan_klaster[i]: labels[i] for i in range(jumlah_klaster)}

                    label_map = buat_label(jumlah_klaster)
                    df_clean['Status Wilayah'] = df_clean['Klaster'].map(label_map)

                    label_mendesak = list(label_map.values())[0]

                    # Hitung skor ranking per indikator
                    skor_total = pd.Series(0.0, index=df_clean.index)
                    for fitur in fitur_makro:
                        if is_positive_indicator:
                            # Indikator positif: nilai rendah = makin buruk
                            skor_total += df_clean[fitur].rank(ascending=True)
                        else:
                            # Indikator negatif: nilai tinggi = makin buruk
                            skor_total += df_clean[fitur].rank(ascending=False)

                    df_clean['Skor_Ranking'] = skor_total

                    # Ranking hanya untuk klaster paling mendesak
                    df_clean['Ranking_Prioritas'] = '-'
                    mask = df_clean['Status Wilayah'] == label_mendesak
                    df_clean.loc[mask, 'Ranking_Prioritas'] = (
                        df_clean.loc[mask, 'Skor_Ranking']
                        .rank(ascending=True)
                        .astype(int)
                        .apply(lambda x: f"#{x}")
                    )

                    sil_score_makro = silhouette_score(X_makro_scaled, df_clean['Klaster'])
                    st.session_state['makro_results'] = {
                        'df_clean': df_clean,
                        'X_scaled': X_makro_scaled,
                        'sil_score': sil_score_makro,
                        'fitur': fitur_makro,
                        'kmeans_obj': kmeans_makro,
                        'label_map': label_map,
                        'jumlah_klaster': jumlah_klaster,
                        'label_mendesak': label_mendesak
                    }
                    st.session_state['makro_jalan'] = True

            if st.session_state.get('makro_jalan'):
                res = st.session_state['makro_results']
                df_clean = res['df_clean']
                sil_score_makro = res['sil_score']
                fitur_res = res['fitur']
                label_map = res['label_map']
                jumlah_klaster_res = res['jumlah_klaster']
                label_mendesak = res['label_mendesak']
                semua_label = list(label_map.values())

                # METRIK
                col_acc1, col_acc2, col_acc3 = st.columns(3)
                with col_acc1:
                    st.metric("🎯 Silhouette Score", f"{sil_score_makro:.2f}")
                with col_acc2:
                    status_ml = "Kuat ✅" if sil_score_makro > 0.5 else "Cukup ⚠️"
                    st.metric("📊 Kualitas Klaster", status_ml)
                with col_acc3:
                    st.metric("🔢 Jumlah Klaster (K)", jumlah_klaster_res)

                # ELBOW METHOD
                with st.expander("📊 Lihat Dasar Matematis Penentuan Jumlah Klaster (Elbow Method)"):
                    distortions = []
                    K_range = range(1, 8)
                    for k in K_range:
                        k_model = KMeans(n_clusters=k, random_state=42, n_init=10)
                        k_model.fit(df_clean[fitur_res])
                        distortions.append(k_model.inertia_)
                    
                    fig_elbow = px.line(x=list(K_range), y=distortions, markers=True, title="Elbow Method — Penentuan K Optimal")
                    fig_elbow.add_vline(x=jumlah_klaster_res, line_dash="dash", line_color="red", annotation_text=f"K={jumlah_klaster_res} (dipilih)")
                    fig_elbow.update_layout(xaxis_title="Jumlah Klaster (K)", yaxis_title="Inertia/Distortion")
                    st.plotly_chart(fig_elbow, use_container_width=True)

                # CHART
                with st.container(border=True):
                    st.markdown("#### 📈 Hasil Klasterisasi Wilayah")
                    col_chart1, col_chart2 = st.columns(2)
                    with col_chart1:
                        fig_pie1 = px.pie(df_clean, names='Status Wilayah', hole=0.4,
                                          color_discrete_sequence=px.colors.qualitative.Pastel)
                        st.plotly_chart(fig_pie1, use_container_width=True)
                    with col_chart2:
                        fig_scatter1 = px.scatter(df_clean, x=fitur_res[0], y=fitur_res[1],
                                                   color='Status Wilayah', hover_data=['Kab/Kota', 'Ranking_Prioritas'],
                                                   color_discrete_sequence=px.colors.qualitative.Pastel)
                        st.plotly_chart(fig_scatter1, use_container_width=True)

                # DIREKTORI — dengan ranking di klaster mendesak
                with st.container(border=True):
                    st.markdown("#### 📋 Direktori Prioritas Wilayah")

                    emoji_list = ['🔴', '🟠', '🟡', '🔵', '🟢']
                    tab_labels = [f"{emoji_list[i] if i < len(emoji_list) else '⚪'} {label}" for i, label in enumerate(semua_label)]
                    tabs_direktori = st.tabs(tab_labels)

                    for i, (tab_dir, label) in enumerate(zip(tabs_direktori, semua_label)):
                        with tab_dir:
                            if label == label_mendesak:
                                # Tampilkan dengan ranking, diurutkan
                                df_filtered = df_clean[df_clean['Status Wilayah'] == label][
                                    ['Ranking_Prioritas', 'Provinsi', 'Kab/Kota', 'Status Wilayah']
                                ].copy()
                                df_filtered['Rank_Sort'] = df_filtered['Ranking_Prioritas'].str.replace('#', '').astype(int)
                                df_filtered = df_filtered.sort_values('Rank_Sort').drop(columns='Rank_Sort').reset_index(drop=True)
                                st.caption(f"Total: {len(df_filtered)} Kab/Kota — diurutkan dari yang **paling mendesak** (Rank #1)")
                            else:
                                df_filtered = df_clean[df_clean['Status Wilayah'] == label][
                                    ['Provinsi', 'Kab/Kota', 'Status Wilayah']
                                ].reset_index(drop=True)
                                st.caption(f"Total: {len(df_filtered)} Kab/Kota")
                            
                            st.dataframe(df_filtered, use_container_width=True)

                # INSIGHT STRATEGIS
                st.markdown("---")
                st.subheader("📌 Insight Strategis Nasional")

                total_wilayah = len(df_clean)
                counts = df_clean['Status Wilayah'].value_counts()
                label_stabil = semua_label[-1]

                p1_pct = (counts.get(label_mendesak, 0) / total_wilayah) * 100
                p_stabil_pct = (counts.get(label_stabil, 0) / total_wilayah) * 100

                col_stat1, col_stat2, col_stat3 = st.columns(3)
                col_stat1.metric("🔴 Tingkat Kerentanan Tinggi", f"{p1_pct:.1f}%", help=f"Persentase wilayah {label_mendesak}")
                col_stat2.metric("🟢 Tingkat Kemandirian", f"{p_stabil_pct:.1f}%", help=f"Persentase wilayah {label_stabil}")
                col_stat3.metric("📍 Total Titik Pantau", f"{total_wilayah} Kab/Kota")

                # TOP 10 PALING MENDESAK
                with st.container(border=True):
                    st.markdown("#### 🏆 Top 10 Wilayah Paling Mendesak")
                    df_top10 = df_clean[df_clean['Status Wilayah'] == label_mendesak].copy()
                    df_top10['Rank_Sort'] = df_top10['Ranking_Prioritas'].str.replace('#', '').astype(int)
                    df_top10 = df_top10.sort_values('Rank_Sort').head(10)[
                        ['Ranking_Prioritas', 'Provinsi', 'Kab/Kota'] + fitur_res
                    ].reset_index(drop=True)
                    st.dataframe(df_top10, use_container_width=True)

                with st.container(border=True):
                    col_info1, col_info2 = st.columns(2)
                    with col_info1:
                        st.error("🚨 **Wilayah Konsentrasi Kemiskinan Tertinggi**")
                        top_p1_prov = df_clean[df_clean['Status Wilayah'] == label_mendesak]['Provinsi'].value_counts().idxmax()
                        count_p1 = df_clean[df_clean['Status Wilayah'] == label_mendesak]['Provinsi'].value_counts().max()
                        st.write(f"Provinsi **{top_p1_prov}** tercatat memiliki konsentrasi wilayah {label_mendesak} terbanyak ({count_p1} Kab/Kota).")
                        st.caption("Direkomendasikan untuk penambahan alokasi anggaran Bansos Makro.")
                    with col_info2:
                        st.success("✅ **Wilayah Paling Stabil**")
                        top_p3_prov = df_clean[df_clean['Status Wilayah'] == label_stabil]['Provinsi'].value_counts().idxmax()
                        count_p3 = df_clean[df_clean['Status Wilayah'] == label_stabil]['Provinsi'].value_counts().max()
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
# TAB 3: AUTO-OPTIMIZER (FIXED K, ALL COMBINATIONS)
# ==========================================
with tab3:
    st.subheader("🤖 SINERGI AI: Auto-Optimizer")
    st.markdown("Pilih jumlah klaster **(K)**, lalu sistem otomatis mencoba **semua kombinasi indikator** dan merekomendasikan yang terbaik.")

    with st.container(border=True):
        st.markdown("#### ⚙️ Konfigurasi Optimizer")

        k_pilihan = st.selectbox(
            "🔢 Pilih Jumlah Klaster (K):",
            options=[2, 3, 4, 5, 6],
            index=1,
            help="Sistem akan mencari kombinasi indikator terbaik untuk K ini"
        )
        info_k = {
            2: "K=2: Wilayah dibagi jadi **Layak Bantuan** dan **Mandiri**.",
            3: "K=3: Standar — **Mendesak, Menengah, Stabil**.",
            4: "K=4: Lebih detail — 4 tingkatan prioritas.",
            5: "K=5: Sangat detail — 5 tingkatan.",
            6: "K=6: Paling granular.",
        }
        st.caption(info_k.get(k_pilihan, ""))

        # Estimasi — pakai SEMUA kolom_angka, max kombinasi 3 indikator (bisa diubah)
        import itertools
        MAX_INDIKATOR = len(kolom_angka)
        total_estimasi = sum(
            len(list(itertools.combinations(kolom_angka, r)))
            for r in range(2, MAX_INDIKATOR + 1)
        )
        estimasi_detik = total_estimasi * 0.05
        estimasi_str = f"~{estimasi_detik:.0f} detik" if estimasi_detik < 60 else f"~{estimasi_detik/60:.1f} menit"
        st.caption(f"🧮 Total kombinasi yang akan diuji: **{total_estimasi:,}** | Estimasi waktu: **{estimasi_str}**")

    if st.button(f"🔍 CARI KOMBINASI TERBAIK UNTUK K={k_pilihan}", type="primary", use_container_width=True):
        best_score = -1
        best_combination = []
        all_results = []

        all_combos = []
        for r in range(2, len(kolom_angka) + 1): 
                all_combos.extend(list(itertools.combinations(kolom_angka, r)))

        total_comb = len(all_combos)
        progress_bar = st.progress(0)
        status_text = st.empty()

        with st.spinner(f'Mengevaluasi {total_comb:,} kombinasi untuk K={k_pilihan}...'):
            for i, combo in enumerate(all_combos):
                df_temp = df_makro.dropna(subset=list(combo))
                if len(df_temp) > 10:
                    try:
                        scaler_temp = StandardScaler()
                        X_scaled_temp = scaler_temp.fit_transform(df_temp[list(combo)])
                        km_temp = KMeans(n_clusters=k_pilihan, random_state=42, n_init=5)
                        labels_temp = km_temp.fit_predict(X_scaled_temp)
                        score_temp = silhouette_score(X_scaled_temp, labels_temp)

                        all_results.append({
                            'Kombinasi Indikator': ' + '.join(combo),
                            'Jumlah Indikator': len(combo),
                            'Silhouette Score': round(score_temp, 4),
                        })

                        if score_temp > best_score:
                            best_score = score_temp
                            best_combination = combo

                    except Exception:
                        pass

                progress_bar.progress((i + 1) / total_comb)
                status_text.caption(f"🔄 Memproses kombinasi {i+1:,} / {total_comb:,}...")

        status_text.empty()
        progress_bar.empty()

        st.session_state['optimizer_results'] = {
            'best_score': best_score,
            'best_combination': best_combination,
            'best_k': k_pilihan,
            'all_results': all_results
        }
        st.balloons()

    # TAMPILKAN HASIL
    if 'optimizer_results' in st.session_state:
        opt = st.session_state['optimizer_results']
        best_score = opt['best_score']
        best_combination = opt['best_combination']
        best_k = opt['best_k']
        df_results = pd.DataFrame(opt['all_results'])

        # HASIL TERBAIK — HERO SECTION
        with st.container(border=True):
            st.success(f"### 🎉 Kombinasi Terbaik untuk K={best_k} Ditemukan!")

            kualitas = "Sangat Kuat 🌟" if best_score > 0.7 else "Kuat ✅" if best_score > 0.5 else "Cukup ⚠️"
            col_best1, col_best2 = st.columns(2)
            col_best1.metric("🏆 Silhouette Score Terbaik", f"{best_score:.4f}", help="Semakin mendekati 1.0 = klaster makin jelas terpisah")
            col_best2.metric("📊 Kualitas Klaster", kualitas)

            st.markdown("**✅ Indikator yang direkomendasikan — salin ke Tab 1:**")
            for idx, f in enumerate(best_combination, 1):
                st.markdown(f"**{idx}.** `{f}`")

        st.markdown("---")

        # TOP 10 — BAR CHART SEDERHANA
        with st.container(border=True):
            st.markdown(f"#### 🏅 Top 10 Kombinasi Terbaik (K={best_k})")
            st.caption("Semakin panjang batangnya = semakin baik kualitas pengelompokannya.")

            # Untuk CHART tetap top 10
        df_top10_chart = (
            df_results
            .sort_values('Silhouette Score', ascending=False)
            .head(10)
            .reset_index(drop=True)
        )
        df_top10_chart.index += 1
        df_top10_chart['Label'] = df_top10_chart.apply(
            lambda row: f"⭐ {row['Kombinasi Indikator']}" if row.name == 1 else row['Kombinasi Indikator'],
            axis=1
        )

        fig_bar = px.bar(
            df_top10_chart,
            x='Silhouette Score',
            y='Label',
            orientation='h',
            color='Silhouette Score',
            color_continuous_scale='RdYlGn',
            text='Silhouette Score',
            title=f"Top 10 Kombinasi Terbaik (K={best_k})"
        )
        fig_bar.update_traces(texttemplate='%{text:.4f}', textposition='outside')
        fig_bar.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            xaxis_title="Silhouette Score (makin tinggi = makin baik)",
            yaxis_title="",
            coloraxis_showscale=False,
            height=420
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # Untuk TABEL tampilkan semua kombinasi
        df_all = (
            df_results
            .sort_values('Silhouette Score', ascending=False)
            .reset_index(drop=True)
        )
        df_all.index += 1
        st.caption(f"📋 Menampilkan semua {len(df_all):,} kombinasi yang diuji")
        st.dataframe(
            df_all[['Kombinasi Indikator', 'Jumlah Indikator', 'Silhouette Score']],
            use_container_width=True,
            height=400  # scroll kalau banyak
        )