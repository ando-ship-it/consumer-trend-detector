"""
Consumer Trend Signal Detector — Streamlit Dashboard (Day 9)
Single-page layout with dark/clean styling.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Consumer Trend Detector",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* hide default Streamlit header/footer */
  #MainMenu, footer, header {visibility: hidden;}

  /* app background */
  .stApp { background: #0f1117; }

  /* top banner */
  .banner {
    background: linear-gradient(135deg, #1a1f2e 0%, #16213e 100%);
    border-radius: 12px;
    padding: 28px 36px 20px;
    margin-bottom: 24px;
    border: 1px solid #2a2f45;
  }
  .banner h1 { color: #e2e8f0; font-size: 2rem; margin: 0 0 4px; }
  .banner p  { color: #94a3b8; margin: 0; font-size: 0.95rem; }

  /* section headers */
  .section-title {
    color: #7dd3fc;
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin: 32px 0 12px;
    border-left: 3px solid #3b82f6;
    padding-left: 10px;
  }

  /* metric cards */
  div[data-testid="metric-container"] {
    background: #1e2535;
    border: 1px solid #2d3555;
    border-radius: 10px;
    padding: 16px 20px;
  }
  div[data-testid="metric-container"] label { color: #94a3b8 !important; font-size: 0.8rem !important; }
  div[data-testid="metric-container"] div[data-testid="stMetricValue"] { color: #e2e8f0 !important; font-size: 1.6rem !important; }

  /* search box */
  .stTextInput input {
    background: #1e2535 !important;
    border: 1px solid #3b82f6 !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
  }

  /* review card */
  .review-card {
    background: #1a1f2e;
    border: 1px solid #2a2f45;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 8px;
    font-size: 0.88rem;
    color: #cbd5e1;
    line-height: 1.5;
  }
  .badge-pos { color: #4ade80; font-weight: 700; }
  .badge-neg { color: #f87171; font-weight: 700; }
  .badge-neu { color: #94a3b8; font-weight: 700; }
  .sim-score { color: #7dd3fc; font-weight: 600; }
  .cluster-tag { color: #a78bfa; font-style: italic; font-size: 0.82rem; }

  /* divider */
  hr { border-color: #2a2f45 !important; }

  /* selectbox / multiselect */
  div[data-baseweb="select"] > div {
    background: #1e2535 !important;
    border-color: #2d3555 !important;
    color: #e2e8f0 !important;
  }
</style>
""", unsafe_allow_html=True)

# ── Load artifacts ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df      = pd.read_csv("outputs/df_clustered.csv")
    trend   = pd.read_csv("outputs/trend_score.csv")
    cbm     = pd.read_csv("outputs/cluster_by_month.csv", index_col="month")
    cbm.index = cbm.index.astype(str)
    umap_df = pd.read_csv("outputs/umap_2d.csv")
    return df, trend, cbm, umap_df

@st.cache_data
def load_sample():
    df_s = pd.read_csv("outputs/df_sample.csv")
    emb  = np.load("outputs/embeddings.npy")
    return df_s, emb

@st.cache_resource
def load_embedder():
    candidates = [
        os.path.join(os.path.dirname(__file__),
            "embedding_model/models--sentence-transformers--all-MiniLM-L6-v2"
            "/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf"),
        os.path.expanduser(
            "~/Downloads/rag-travel-assistant-master/embedding_model"
            "/models--sentence-transformers--all-MiniLM-L6-v2"
            "/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf"),
    ]
    for p in candidates:
        if os.path.isdir(p):
            return SentenceTransformer(p)
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

df, trend_score_df, cluster_by_month, umap_df = load_data()
df_sample, embeddings = load_sample()
embedder = load_embedder()

# ── Banner ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="banner">
  <h1>📡 Consumer Trend Signal Detector</h1>
  <p>NLP analysis of Google Play Store reviews · 11,190 reviews · 2015 – 2020 · 12 topic clusters</p>
</div>
""", unsafe_allow_html=True)

# ── KPI row ───────────────────────────────────────────────────────────────────
pos_pct = round((df["sentiment"] == "Positive").mean() * 100, 1)
neg_pct = round((df["sentiment"] == "Negative").mean() * 100, 1)
top_trend = trend_score_df.sort_values("trend_score", ascending=False).iloc[0]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total reviews",    f"{len(df):,}")
c2.metric("Topic clusters",   "12")
c3.metric("Positive",         f"{pos_pct}%")
c4.metric("Negative",         f"{neg_pct}%")
c5.metric("Top trend signal", top_trend["cluster"])

st.markdown("<hr>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 1 — Trend signals + Time-series
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">📈 Trend Signals</div>', unsafe_allow_html=True)

left, right = st.columns([1, 2])

with left:
    # Trend score bar chart
    ts = trend_score_df.sort_values("trend_score").copy()
    ts["growth_pct"] = (ts["growth_rate"] * 100).round(1)
    ts["neg_pct"]    = (ts["neg_share"]   * 100).round(1)

    fig_trend = go.Figure(go.Bar(
        x=ts["trend_score"],
        y=ts["cluster"],
        orientation="h",
        marker=dict(
            color=ts["trend_score"],
            colorscale=[[0, "#1e3a5f"], [0.5, "#3b82f6"], [1, "#f97316"]],
            showscale=False,
        ),
        customdata=ts[["growth_pct", "neg_pct"]].values,
        hovertemplate="<b>%{y}</b><br>Score: %{x:.5f}<br>Growth: %{customdata[0]}%<br>Neg share: %{customdata[1]}%<extra></extra>",
    ))
    fig_trend.update_layout(
        height=380, margin=dict(l=0, r=10, t=10, b=10),
        paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
        xaxis=dict(title="Trend score", color="#94a3b8", gridcolor="#1e2535"),
        yaxis=dict(color="#cbd5e1", tickfont=dict(size=11)),
        font=dict(color="#cbd5e1"),
    )
    st.plotly_chart(fig_trend, use_container_width=True)

with right:
    # Time-series selector
    default_clusters = ["Task management", "Premium / paywall friction", "Ads friction", "Habit tracking"]
    sel = st.multiselect(
        "Clusters to plot",
        options=list(cluster_by_month.columns),
        default=[c for c in default_clusters if c in cluster_by_month.columns],
        label_visibility="collapsed",
    )

    if sel:
        palette = px.colors.qualitative.Plotly
        fig_ts = go.Figure()
        for i, col in enumerate(sel):
            fig_ts.add_trace(go.Scatter(
                x=list(cluster_by_month.index),
                y=cluster_by_month[col].values,
                name=col,
                mode="lines",
                line=dict(width=2, color=palette[i % len(palette)]),
                hovertemplate=f"<b>{col}</b><br>Month: %{{x}}<br>Share: %{{y:.1%}}<extra></extra>",
            ))
        # X ticks every 6 months
        all_idx = list(cluster_by_month.index)
        tick_vals = all_idx[::6]
        fig_ts.update_layout(
            height=380, margin=dict(l=0, r=10, t=10, b=60),
            paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
            xaxis=dict(tickvals=tick_vals, tickangle=-45, color="#94a3b8",
                       gridcolor="#1e2535", tickfont=dict(size=10)),
            yaxis=dict(tickformat=".0%", color="#94a3b8", gridcolor="#1e2535"),
            legend=dict(bgcolor="#1a1f2e", bordercolor="#2a2f45",
                        font=dict(color="#cbd5e1", size=11)),
            font=dict(color="#cbd5e1"),
            hovermode="x unified",
        )
        st.plotly_chart(fig_ts, use_container_width=True)
    else:
        st.info("Select at least one cluster.")

st.markdown("<hr>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 2 — UMAP cluster map + Cluster explorer
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">🗺 Cluster Map & Review Explorer</div>', unsafe_allow_html=True)

map_col, exp_col = st.columns([2, 1])

with map_col:
    fig_umap = px.scatter(
        umap_df, x="x", y="y",
        color="cluster_label",
        hover_data={"review_text": True, "sentiment": True, "x": False, "y": False},
        color_discrete_sequence=px.colors.qualitative.Plotly,
        opacity=0.6,
        template="plotly_dark",
    )
    fig_umap.update_traces(marker=dict(size=3))
    fig_umap.update_layout(
        height=420, margin=dict(l=0, r=0, t=10, b=10),
        paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
        legend=dict(title="Cluster", bgcolor="#1a1f2e", bordercolor="#2a2f45",
                    font=dict(color="#cbd5e1", size=10)),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    st.plotly_chart(fig_umap, use_container_width=True)

with exp_col:
    chosen = st.selectbox("Topic cluster", sorted(df["cluster_label"].unique()), label_visibility="visible")
    sent_f = st.radio("Sentiment", ["All", "Positive", "Negative", "Neutral"], horizontal=True)

    sub = df[df["cluster_label"] == chosen]
    if sent_f != "All":
        sub = sub[sub["sentiment"] == sent_f]

    pos_c = (sub["sentiment"] == "Positive").sum()
    neg_c = (sub["sentiment"] == "Negative").sum()
    st.caption(f"**{len(sub):,} reviews** · {pos_c} pos / {neg_c} neg")

    samples = sub["review_text"].dropna().sample(min(8, len(sub)), random_state=42)
    for rev in samples:
        st.markdown(f'<div class="review-card">{rev}</div>', unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 3 — Semantic search
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">🔍 Semantic Search</div>', unsafe_allow_html=True)

q_col, res_col = st.columns([1, 2])

with q_col:
    query  = st.text_input("", placeholder="e.g. app crashes after update", label_visibility="collapsed")
    top_k  = st.slider("Results", 3, 20, 8)
    filter_sent = st.radio("Filter", ["All", "Positive", "Negative"], horizontal=True, key="search_sent")

with res_col:
    if query.strip():
        with st.spinner("Searching..."):
            qvec  = embedder.encode([query], normalize_embeddings=True)
            sims  = cosine_similarity(qvec, embeddings)[0]
            top_i = sims.argsort()[::-1]

        results = df_sample.iloc[top_i].copy()
        results["_sim"] = sims[top_i]
        if filter_sent != "All":
            results = results[results["sentiment"] == filter_sent]
        results = results.head(top_k)

        badge_map = {"Positive": "badge-pos", "Negative": "badge-neg", "Neutral": "badge-neu"}
        shown = 0
        for _, row in results.iterrows():
            cls = badge_map.get(row["sentiment"], "badge-neu")
            st.markdown(
                f'<div class="review-card">'
                f'<span class="sim-score">{row["_sim"]:.3f}</span> · '
                f'<span class="{cls}">{row["sentiment"]}</span> · '
                f'<span class="cluster-tag">{row["cluster_label"]}</span><br>'
                f'{row["review_text"]}'
                f'</div>',
                unsafe_allow_html=True,
            )
            shown += 1
        if shown == 0:
            st.info("No results for this sentiment filter.")
    else:
        st.markdown(
            '<div class="review-card" style="color:#475569;text-align:center;padding:40px;">'
            'Type a query on the left to search reviews by meaning, not keywords.'
            '</div>',
            unsafe_allow_html=True,
        )
