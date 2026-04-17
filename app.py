import streamlit as st
import pandas as pd
import networkx as nx
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import requests
from io import StringIO
from pyvis.network import Network
import streamlit.components.v1 as components


# Sayfa ayarları
st.set_page_config(page_title="Hemithea Portfolio Analiz", layout="wide")

# CSS
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
    except Exception as e:
        st.error(f"Veri yüklenemedi: {e}")
        return pd.DataFrame()

# --- ANA AKIŞ ---
st.title("🌐 Hemithea Portfolio Network Analytics")

secim = st.sidebar.selectbox("Veri Seti Seç", ["Efendi Analizi", "Game of Thrones"])
data = load_github_data(secim)

if data is not None and not data.empty:
    st.success("✅ Analiz Hazır!")
    st.write("Veri boyutu:", data.shape)
    st.write(data.head())

    tab1, tab2, tab3 = st.tabs(["🕸️ Ağ Haritası", "📈 Metrikler", "📄 Veri"])

    # Sütunları otomatik seç
    cols = data.columns.tolist()
    if "Source" in cols and "Target" in cols:
        G = nx.from_pandas_edgelist(data, source="Source", target="Target", edge_attr="Weight")
    elif "source" in cols and "target" in cols:
        edge_attrs = [c for c in cols if c not in ["source", "target"]]
        G = nx.from_pandas_edgelist(data, source="source", target="target", edge_attr=edge_attrs)
    else:
        st.error("CSV'de source/target sütunları bulunamadı!")
        G = None

    if G:
        with tab1:
            degree_cent = nx.degree_centrality(G)
            betweenness = nx.betweenness_centrality(G)
        
            metrics_df = pd.DataFrame({
                'node': list(degree_cent.keys()),
                'degree': list(degree_cent.values()),
                'betweenness': list(betweenness.values())
            })
        
            # KNN ve renklendirme
            if len(metrics_df) > 3:
                X = metrics_df[['degree', 'betweenness']].values
                y = (metrics_df['betweenness'] > metrics_df['betweenness'].mean()).astype(int)
                X_scaled = StandardScaler().fit_transform(X)
                n_neighbors = max(1, min(3, len(metrics_df)-1))
                knn = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X_scaled, y)
                metrics_df['color'] = pd.Series(knn.predict(X_scaled)).map({1: "#e74c3c", 0: "#3498db"})
            else:
                metrics_df['color'] = "#3498db"
        
            # --- Pyvis ile görsel ağ ---
            net = Network(height="600px", width="100%")
            for i, row in metrics_df.iterrows():
                net.add_node(row['node'], color=row['color'],
                             title=f"Degree: {row['degree']}, Betweenness: {row['betweenness']}")
            for u, v, d in G.edges(data=True):
                net.add_edge(u, v, title=str(d))
            
            # HTML üret
            html_str = net.generate_html()
            
            # Streamlit'e göm
            components.html(html_str, height=600, scrolling=True)


            # HTML dosyası
            html_str = net.generate_html()
            st.download_button(
                label="📥 Ağ Haritasını HTML indir",
                data=html_str,
                file_name="network.html",
                mime="text/html"
            )

            # PNG için NetworkX + Matplotlib
            plt.figure(figsize=(8,6))
            nx.draw(G, with_labels=True, node_color="skyblue", edge_color="gray")
            plt.savefig("network.png")
            with open("network.png", "rb") as f:
                st.download_button("📥 Ağ Haritası (PNG)", f, "network.png", "image/png")



        with tab2:
            st.dataframe(metrics_df)
            # CSV indir
            csv_data = metrics_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Metrikleri CSV indir",
                data=csv_data,
                file_name="metrics.csv",
                mime="text/csv"
            )
            
            # PNG indir (matplotlib bar chart örneği)
            plt.figure(figsize=(8,6))
            plt.bar(metrics_df['node'], metrics_df['degree'], color=metrics_df['color'])
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig("metrics.png")
            with open("metrics.png", "rb") as f:
                st.download_button(
                    label="📥 Metrikleri PNG indir",
                    data=f,
                    file_name="metrics.png",
                    mime="image/png"
                )


        with tab3:
            st.dataframe(data)
