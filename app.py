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
        "Efendi Analizi": "https://raw.githubusercontent.com/seydanur-hemithea/noLoginAnalysis/main/Efendi.csv",
        "Game of Thrones": "https://raw.githubusercontent.com/seydanur-hemithea/noLoginAnalysis/main/GoT.csv"
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
            y = (metrics_df['betweenness'] > metrics_df['betweenness'].mean()).astype(int)
            X_scaled = StandardScaler().fit_transform(X)
            n_neighbors = max(1, min(3, len(metrics_df)-1))
            knn = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X_scaled, y)
            metrics_df['color'] = pd.Series(knn.predict(X_scaled)).map({1: "#e74c3c", 0: "#3498db"})
        else:
            metrics_df['color'] = "#3498db"
