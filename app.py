import streamlit as st
import pandas as pd
import networkx as nx
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import requests
from io import StringIO
from pyvis.network import Network
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

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
            # 1. Gelişmiş Metrik Hesaplamaları
            # Ağın derinliğini anlamak için yeni metrikler ekliyoruz
            degree_cent = nx.degree_centrality(G)
            betweenness = nx.betweenness_centrality(G)
            closeness = nx.closeness_centrality(G)
            clustering = nx.clustering(G)
            pagerank = nx.pagerank(G)
            k_core = nx.core_number(G)
            
            # HITS (Hubs ve Authorities) - İki değer döner
            try:
                hubs, authorities = nx.hits(G, max_iter=1000)
            except:
                hubs = {n: 0 for n in G.nodes()}
                authorities = {n: 0 for n in G.nodes()}

            # 2. DataFrame Oluşturma
            metrics_df = pd.DataFrame({
                'node': list(G.nodes()),
                'degree': [degree_cent[n] for n in G.nodes()],
                'betweenness': [betweenness[n] for n in G.nodes()],
                'closeness': [closeness[n] for n in G.nodes()],
                'clustering': [clustering[n] for n in G.nodes()],
                'pagerank': [pagerank[n] for n in G.nodes()],
                'k_core': [k_core[n] for n in G.nodes()],
                'hubs': [hubs[n] for n in G.nodes()],
                'auth': [authorities[n] for n in G.nodes()]
            }).fillna(0) # Eksik değerleri temizle

            # 3. KNN için Özellik Matrisi ve Ölçeklendirme
            features = ['degree', 'betweenness', 'closeness', 'clustering', 'pagerank', 'k_core', 'hubs', 'auth']
            X = metrics_df[features].values
            
            # Ölçeklendirme (StandardScaler): KNN için hayati önem taşır
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # 4. KNN Sınıflandırma
            # Hedef: Betweenness (stratejik önem) + PageRank (etki gücü) kombinasyonu
            y_target = (metrics_df['betweenness'] > metrics_df['betweenness'].mean()) | \
                       (metrics_df['pagerank'] > metrics_df['pagerank'].mean())
            y = y_target.astype(int)

            # Dinamik K seçimi (Ağ boyutunun karekökü)
            k_val = max(3, int(len(metrics_df)**0.5))
            if k_val % 2 == 0: k_val += 1
            
            knn = KNeighborsClassifier(n_neighbors=k_val)
            knn.fit(X_scaled, y)
            predictions = knn.predict(X_scaled)

            # Sonuçları DataFrame'e ekle
            metrics_df['color'] = ["#e74c3c" if p == 1 else "#3498db" for p in predictions]
            metrics_df['role'] = ["Kritik" if p == 1 else "Normal" for p in predictions]

            # 5. Pyvis ile Görselleştirme
            net = Network(height="600px", width="100%", bgcolor='#222222', font_color='white')
            
            for _, row in metrics_df.iterrows():
                net.add_node(row['node'], color=row['color'], 
                             title=f"Rol: {row['role']}<br>PageRank: {row['pagerank']:.3f}<br>Degree: {row['degree']:.2f}")
            
            for u, v, d in G.edges(data=True):
                net.add_edge(u, v)
            
            html_str = net.generate_html()
            components.html(html_str, height=600, scrolling=True)

            # İndirme Butonları
            col1, col2 = st.columns(2)
            with col1:
                st.download_button("📥 Ağ Haritasını HTML indir", html_str, "network.html", "text/html")
            with col2:
                fig, ax = plt.subplots(figsize=(8,6))
                nx.draw(G, with_labels=True, node_color=metrics_df['color'].tolist(), ax=ax)
                st.pyplot(fig)


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
