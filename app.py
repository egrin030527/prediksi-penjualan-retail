
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── KONFIGURASI HALAMAN ──────────────────────────────────────
st.set_page_config(
    page_title="Prediksi Penjualan Retail",
    page_icon="🏪",
    layout="wide"
)

# ── LOAD MODEL ───────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("model_final.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# ── DATA HISTORIS ─────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.DataFrame({
        "bulan_ke"        : list(range(1, 13)),
        "bulan"           : ["Jan","Feb","Mar","Apr","Mei","Jun",
                              "Jul","Agu","Sep","Okt","Nov","Des"],
        "total_penjualan" : [35450,44060,28990,33870,53150,36715,
                              35465,36960,23620,46580,34920,44690]
    })

df = load_data()

# ── HEADER ───────────────────────────────────────────────────
st.title("🏪 Sistem Prediksi Penjualan Produk")
st.markdown("**Optimasi Stok Gudang Toko Retail** · Algoritma Linear Regression")
st.divider()

# ── LAYOUT ───────────────────────────────────────────────────
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📊 Input Prediksi")
    bulan_input = st.number_input(
        "Masukkan Bulan ke-N",
        min_value=1, max_value=36,
        value=13, step=1,
        help="Bulan ke-13 = Jan 2024, ke-14 = Feb 2024, dst"
    )

    if st.button("🔮 Prediksi Sekarang", width='stretch'):
        hasil = model.predict([[bulan_input]])[0]
        st.session_state["hasil"] = hasil
        st.session_state["bulan"] = bulan_input

    if "hasil" in st.session_state:
        hasil = st.session_state["hasil"]
        bulan = st.session_state["bulan"]

        st.success(f"✅ Hasil Prediksi Bulan ke-{bulan}")
        st.metric(
            label="Total Penjualan Prediksi",
            value=f"Rp {hasil:,.0f}",
            delta=f"{((hasil - df['total_penjualan'].mean()) / df['total_penjualan'].mean() * 100):+.1f}% vs rata-rata"
        )

        rekomendasi = hasil / df["total_penjualan"].mean()
        if rekomendasi > 1.1:
            st.warning(f"📦 Tambah stok **{rekomendasi:.2f}x** dari rata-rata")
        elif rekomendasi < 0.9:
            st.info(f"📦 Kurangi stok — hanya **{rekomendasi:.2f}x** dari rata-rata")
        else:
            st.success(f"📦 Stok normal — **{rekomendasi:.2f}x** dari rata-rata")

        st.markdown("---")
        st.markdown("**Detail Model:**")
        st.code(
            f"y = {model.intercept_:,.0f} + {model.coef_[0]:,.0f} × {bulan}\n"
            f"y = {hasil:,.0f}",
            language="python"
        )

with col2:
    st.subheader("📈 Tren Historis + Garis Regresi")

    fig, ax = plt.subplots(figsize=(10, 4))

    # bar historis
    colors = ["#2ecc71" if v == max(df["total_penjualan"])
               else "#e74c3c" if v == min(df["total_penjualan"])
               else "#85C1E9"
               for v in df["total_penjualan"]]
    ax.bar(df["bulan"], df["total_penjualan"],
           color=colors, alpha=0.85, edgecolor="white", label="Data historis")

    # garis regresi
    y_reg = model.intercept_ + model.coef_[0] * df["bulan_ke"].values
    ax.plot(df["bulan"], y_reg, color="#8e44ad",
            linewidth=2, linestyle="--", label="Garis regresi")

    # garis prediksi
    if "hasil" in st.session_state:
        ax.axhline(
            st.session_state["hasil"],
            color="#e74c3c", linewidth=1.5,
            linestyle=":", alpha=0.8,
            label=f"Prediksi bln ke-{st.session_state['bulan']}: Rp {st.session_state['hasil']:,.0f}"
        )

    ax.set_xlabel("Bulan")
    ax.set_ylabel("Total Penjualan")
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k")
    )
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)

st.divider()

# ── TABEL DATA HISTORIS ──────────────────────────────────────
st.subheader("🗂️ Data Historis Penjualan (Jan–Des 2023)")
df_display = df.copy()
df_display["total_penjualan"] = df_display["total_penjualan"].apply(
    lambda x: f"Rp {x:,}"
)
st.dataframe(df_display, width='stretch', hide_index=True)

st.divider()
st.caption("Proyek Data Mining ST167 · Linear Regression · Retail Sales Dataset · 2024")
