import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import butter, filtfilt

# --- 1. Fungsi Helper (Signal Processing) ---

def remove_spikes(data, threshold=3):
    """Menghapus spike ekstrem menggunakan Z-score dan interpolasi."""
    mean = data.mean()
    std = data.std()
    # Tandai data yang melebihi threshold sebagai NaN
    filtered = data.where((data - mean).abs() <= (threshold * std), np.nan)
    # Isi gap NaN dengan interpolasi linear agar sinyal kontinu
    return filtered.interpolate(method='linear').ffill().bfill()

def low_pass_filter(data, cutoff_hours, fs_per_hour=60):
    """Filter digital untuk membuang noise frekuensi tinggi."""
    nyq = 0.5 * fs_per_hour
    cutoff = 1 / cutoff_hours 
    normal_cutoff = cutoff / nyq
    # Menggunakan filter Butterworth order 1 agar tidak terlalu agresif
    b, a = butter(1, normal_cutoff, btype='low', analog=False)
    # filtfilt digunakan agar tidak ada pergeseran fase (delay) pada grafik
    return filtfilt(b, a, data)

# --- 2. Konfigurasi Halaman ---
st.set_page_config(page_title="Water Level Analytics Pro", layout="wide")
st.title("Water Level Analytics")

uploaded_file = st.file_uploader("Unggah file .csv", type=["csv"])

if uploaded_file:
    # Load & Pre-process Data
    df = pd.read_csv(uploaded_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').sort_index()

    # --- 3. Sidebar Panel ---
    st.sidebar.header("Konfigurasi Analisis")
    
    durations_map = {"1 Jam": "1H", "3 Jam": "3H", "12 Jam": "12H", "25 Jam": "25H"}
    window_map = {"1 Jam": 60, "3 Jam": 180, "12 Jam": 720, "25 Jam": 1500}
    
    analysis_mode = st.sidebar.radio("Mode Tampilan", 
                                   ["Single Duration", "Hourly Overlay"],
                                   help="Pilih apakah ingin melihat satu durasi saja atau perbandingan semua durasi sekaligus.")

    method = st.sidebar.selectbox("Metode", 
                                ["Averaging", "Moving Average", "Filtering (Low Pass)"])
    
    if analysis_mode == "Single Duration":
        selected_label = st.sidebar.select_slider("Pilih Interval Waktu", options=list(durations_map.keys()))
    
    st.sidebar.markdown("---")
    st.sidebar.header("Visualisasi")
    show_raw = st.sidebar.checkbox("Tampilkan Raw Data", value=True)
    show_filtered = st.sidebar.checkbox("Tampilkan Filtered Data", value=True)

    # --- 4. Pemrosesan Data Inti ---
    # Langkah 1: Bersihkan data dari spike
    processed_df = df.copy()
    processed_df['filtered_wl'] = remove_spikes(processed_df['water_level'])
    
    # Fungsi Internal untuk kalkulasi dinamis
    def calculate_results(m, label, data_series):
        # Pastikan data tidak mengandung NaN sebelum LPF
        clean_input = data_series.ffill().bfill()
        
        if m == "Averaging":
            return data_series.resample(durations_map[label]).mean()
        elif m == "Moving Average":
            return data_series.rolling(window=window_map[label], center=True).mean()
        else: # Filtering (Low Pass)
            try:
                cutoff_val = int(label.split()[0])
                lpf_val = low_pass_filter(clean_input, cutoff_val)
                return pd.Series(lpf_val, index=data_series.index)
            except Exception as e:
                return pd.Series(index=data_series.index)

    # --- 5. Interface Utama (Tabs) ---
    tab1, tab2 = st.tabs(["Grafik Analisis", "Preview & Statistik Data"])

    with tab1:
        fig = go.Figure()

        # Layer Bawah: Raw Data
        if show_raw:
            fig.add_trace(go.Scatter(x=df.index, y=df['water_level'], 
                                     name="Raw Data", 
                                     line=dict(color='rgba(200, 200, 200, 0.4)', width=1)))

        # Layer Tengah: Filtered Data
        if show_filtered:
            fig.add_trace(go.Scatter(x=processed_df.index, y=processed_df['filtered_wl'], 
                                     name="Filtered Data", 
                                     line=dict(color='rgba(50, 50, 50, 0.6)', width=1.5)))

        # Layer Atas: Hasil Analisis (Warna Terang)
        colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA']
        
        if analysis_mode == "Single Duration":
            res = calculate_results(method, selected_label, processed_df['filtered_wl'])
            fig.add_trace(go.Scatter(x=res.index, y=res.values, 
                                     name=f"{method}: {selected_label}",
                                     line=dict(color='#FFD700', width=3)))
        else:
            # Mode Overlay Semua Jam
            for i, label in enumerate(durations_map.keys()):
                res = calculate_results(method, label, processed_df['filtered_wl'])
                fig.add_trace(go.Scatter(x=res.index, y=res.values, 
                                         name=f"{label} ({method})",
                                         line=dict(color=colors[i], width=2)))

        fig.update_layout(
            title=f"Analisis Water Level: {method}",
            xaxis_title="Waktu", yaxis_title="Water Level",
            hovermode="x unified", template="plotly_white", height=650,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Data Table Preview")
        col_table, col_stats = st.columns([2, 1])
        
        with col_table:
            # Tampilkan 100 baris pertama untuk performa
            st.write("**Raw Data dan Filtered Data**")
            st.dataframe(processed_df[['water_level', 'filtered_wl']].head(100), use_container_width=True)
        
        with col_stats:
            st.write("**Statistik Perbandingan**")
            comparison = pd.DataFrame({
                "Raw Data": df['water_level'].describe(),
                "Filtered Data": processed_df['filtered_wl'].describe()
            })
            st.table(comparison)
        
        # Opsi Download Data Filtered
        csv_data = processed_df.to_csv().encode('utf-8')
        st.download_button("Download Hasil Filtered Data (CSV)", csv_data, 
                           "filtered_data.csv", "text/csv", key='download-csv')

    # --- 6. Ringkasan Metrik (Berdasarkan Hasil Terakhir) ---
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    # Ambil nilai terakhir dari loop atau single duration untuk display
    c1.metric("Max Level", f"{res.max():.2f}")
    c2.metric("Min Level", f"{res.min():.2f}")
    c3.metric("Rata-rata", f"{res.mean():.2f}")
    c4.metric("Status Data", "Cleaned" if not res.isnull().all() else "Error/Empty")

else:
    st.info("Silakan unggah file CSV di sidebar untuk melihat analisis.")
