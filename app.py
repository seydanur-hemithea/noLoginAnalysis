import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import requests
from io import StringIO, BytesIO
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# 1. Sayfa Ayarları
st.set_page_config(page_title="Hemithea Portfolio Analiz", layout="wide")

# Mobil CSS
st.markdown("""
    <style>
    .main > div { padding: 0.5rem; }
    iframe { width: 100% !important; }
    </style>
    """, unsafe_allow_html=True)

# --- GITHUB RAW DÖNÜŞTÜRÜCÜ ---
def to_raw(url):
    if "github.com" in url and "raw" not in url:
        return url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
    return url

@st.cache_data(ttl=600)
def load_github_data(secim):
    linkler = {
        "Efendi Analizi": "https://github.com/seydanur-hemithea/appHemitheaNetwork2/blob/main/Efendi.csv",
        "Game of Thrones": "https://github.com/seydanur-hemithea/appHemitheaNetwork2/blob/main/GoT.csv"
    }
    try:
        raw_url = to_raw(linkler[secim])
        res = requests.get(raw_url)
        return pd.read_csv(StringIO(res.text))
    except:
        return pd.DataFrame({'Kaynak': ['Örnek'], 'Hedef': ['Veri']})

# --- ANA AKIŞ ---
st.title("🌐 Hemithea Portfolio Network Analytics")

secim = st.sidebar.selectbox("Veri Seti Seç", ["Efendi Analizi", "Game of Thrones"])
data = load_github_data(secim)

if data is not None and not data.empty:
    cols = data.columns.tolist()
    src, tgt = cols[0], cols[1]
    st.success("✅ Analiz Hazır!")

    tab1, tab2, tab3 = st.tabs(["🕸️ Ağ Haritası", "📈 Metrikler", "📄 Veri"])
    G = nx.from_pandas_edgelist(data, source=src, target=tgt)

    with tab1:
        degree_cent = nx.degree_centrality(G)
        betweenness = nx.betweenness_centrality(G)

        metrics_df = pd.DataFrame({
            'node': list(degree_cent.keys()),
            'degree': list(degree_cent.values()),
            'betweenness': list(betweenness.values())
        })

        # KNN ve Renklendirme
        if len(metrics_df) > 3:
            X = metrics_df[['degree', 'betweenness']].values
            y = (metrics_df['betweenness'] > metrics_df['betweenness'].mean())
