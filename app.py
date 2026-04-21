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

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            
            /* Modern Streamlit sürümleri için ek önlemler */
            [data-testid="stDecoration"] {visibility: hidden;}
            [data-testid="stStatusWidget"] {visibility: hidden;}
            div.block-container {padding-top: 1rem;}
            </style>
            """

st.markdown(hide_streamlit_style, unsafe_allow_html=True)




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
    # 1. Metrik Hesaplamaları
    # Ağdaki her düğüm için 5 farklı topolojik özellik hesaplıyoruz
    degree_cent = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G)
    closeness = nx.closeness_centrality(G)
    clustering = nx.clustering(G)
    
    # Eigenvector centrality bazı ağlarda yakınsama hatası verebilir, try-except ile korumaya aldık
    try:
        eigen = nx.eigenvector_centrality(G, max_iter=1000)
    except:
        eigen = {node: 0 for node in G.nodes()}

    # 2. DataFrame Oluşturma
    metrics_df = pd.DataFrame({
        'node': list(G.nodes()),
        'degree': [degree_cent[n] for n in G.nodes()],
        'betweenness': [betweenness[n] for n in G.nodes()],
        'closeness': [closeness[n] for n in G.nodes()],
        'clustering': [clustering[n] for n in G.nodes()],
        'eigen': [eigen[n] for n in G.nodes()]
    })

    # 3. Random Forest ile Tahminleme ve Rol Atama
    # Veri setimiz yeterliyse (min 5 düğüm) modeli eğitiyoruz
    if len(metrics_df) > 5:
        # Hedef: Betweenness ortalamanın üzerinde olan düğümleri "Kritik" (1) olarak etiketliyoruz
        y = (metrics_df['betweenness'] > metrics_df['betweenness'].mean()).astype(int)
        X = metrics_df[['degree', 'closeness', 'clustering', 'eigen']]
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        predictions = rf.predict(X)
        
        # Renk ve Rol atamaları
        metrics_df['color'] = ["#e74c3c" if p == 1 else "#3498db" for p in predictions]
        metrics_df['role'] = ["Kritik" if p == 1 else "Normal" for p in predictions]
    else:
        metrics_df['color'] = "#3498db"
        metrics_df['role'] = "Normal"

    # 4. Pyvis ile Görselleştirme
    net = Network(height="600px", width="100%", bgcolor='#222222', font_color='white')
    
    # Düğümleri eklerken Random Forest'tan gelen renkleri kullanıyoruz
    for _, row in metrics_df.iterrows():
        net.add_node(row['node'], color=row['color'], 
                     title=f"Role: {row['role']}<br>Degree: {row['degree']:.2f}")
    
    # Kenarları ekle
    for u, v, d in G.edges(data=True):
        net.add_edge(u, v)
    
    # HTML üret ve Streamlit'e göm
    html_str = net.generate_html()
    components.html(html_str, height=600, scrolling=True)

    # İndirme Butonları
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("📥 Ağ Haritasını HTML indir", html_str, "network.html", "text/html")
    
    with col2:
        # PNG önizlemesi
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
