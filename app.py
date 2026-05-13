"""
Consumer Trend Signal Detector — Streamlit Dashboard
"""

import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Consumer Trend Detector",
    page_icon="📊",
    layout="wide",
)

# ── Load artifacts ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("outputs/df_clustered.csv")
    trend = pd.read_csv("outputs/trend_score.csv")
    cbm = pd.read_csv("outputs/cluster_by_month.csv", index_col="month")
    return df, trend, cbm

@st.cache_data
def load_sample():
    df_s = pd.read_csv("outputs/df_sample.csv")
    emb = np.load("outputs/embeddings.npy")
    return df_s, emb

@st.cache_resource
def load_embedder():
    model_path = (
        "embedding_model/models--sentence-transformers--all-MiniLM-L6-v2"
        "/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
    )
    return SentenceTransformer(model_path)

df, trend_score_df, cluster_by_month = load_data()
df_sample, embeddings = load_sample()
embedder = load_embedder()

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title("📊 Consumer Trend Detector")
st.sidebar.markdown("Google Play Store reviews · 2015–2020")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "📈 Trend Signals", "🔍 Semantic Search"],
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("Consumer Trend Signal Detector")
    st.markdown(
        "Detects **emerging topic trends** in app store reviews using "
        "NLP clustering and time-series analysis."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Total reviews", f"{len(df):,}")
    col2.metric("Clusters (topics)", "12")
    col3.metric("Date range", "2015 – 2020")

    st.markdown("---")
    st.subheader("Sentiment distribution")
    sent_counts = df["sentiment"].value_counts()
    fig, ax = plt.subplots(figsize=(5, 3))
    colors = {"Positive": "#4caf50", "Negative": "#f44336", "Neutral": "#9e9e9e"}
    bars = ax.bar(
        sent_counts.index,
        sent_counts.values,
        color=[colors.get(s, "#888") for s in sent_counts.index],
    )
    for bar, val in zip(bars, sent_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f"{val:,}", ha="center", fontsize=9)
    ax.set_ylabel("Reviews")
    ax.set_title("Reviews by sentiment")
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.subheader("Cluster size distribution")
    cluster_counts = df["cluster_label"].value_counts().sort_values(ascending=True)
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.barh(cluster_counts.index, cluster_counts.values, color="#1976d2")
    ax2.set_xlabel("Number of reviews")
    ax2.set_title("Reviews per topic cluster")
    st.pyplot(fig2)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — TREND SIGNALS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Trend Signals":
    st.title("📈 Trend Signals")
    st.markdown(
        "**Trend score** = growth rate × negative share. "
        "High score → topic is growing AND users are unhappy → actionable signal."
    )

    # Trend score table
    st.subheader("Trend score ranking")
    display_df = trend_score_df.copy()
    display_df["growth_rate"] = (display_df["growth_rate"] * 100).round(1).astype(str) + "%"
    display_df["neg_share"] = (display_df["neg_share"] * 100).round(1).astype(str) + "%"
    display_df["share_early"] = (display_df["share_early"] * 100).round(1).astype(str) + "%"
    display_df["share_recent"] = (display_df["share_recent"] * 100).round(1).astype(str) + "%"
    display_df["trend_score"] = display_df["trend_score"].round(5)
    display_df.columns = ["Cluster", "Share (early)", "Share (recent)", "Growth", "Neg share", "Trend score"]
    st.dataframe(
        display_df.sort_values("Trend score", ascending=False).reset_index(drop=True),
        use_container_width=True,
    )

    st.markdown("---")

    # Time-series chart
    st.subheader("Topic share over time")

    cluster_options = list(cluster_by_month.columns)
    selected_clusters = st.multiselect(
        "Select clusters to plot",
        options=cluster_options,
        default=["Task management", "Premium / paywall friction", "Ads friction", "Habit tracking"],
    )

    if selected_clusters:
        fig3, ax3 = plt.subplots(figsize=(12, 5))
        for col in selected_clusters:
            if col in cluster_by_month.columns:
                ax3.plot(
                    range(len(cluster_by_month)),
                    cluster_by_month[col].values,
                    label=col,
                    linewidth=2,
                )

        # X-axis ticks every 6 months
        tick_step = 6
        ticks = list(range(0, len(cluster_by_month), tick_step))
        ax3.set_xticks(ticks)
        ax3.set_xticklabels(
            [str(cluster_by_month.index[t]) for t in ticks],
            rotation=45, ha="right", fontsize=8
        )
        ax3.set_ylabel("Share of monthly reviews")
        ax3.set_title("Topic share over time (monthly)")
        ax3.legend(loc="upper left", fontsize=8)
        ax3.grid(alpha=0.3)
        st.pyplot(fig3)
        plt.close()
    else:
        st.info("Select at least one cluster above.")

    st.markdown("---")

    # Cluster explorer
    st.subheader("Explore cluster reviews")
    chosen_cluster = st.selectbox("Choose a cluster", sorted(df["cluster_label"].unique()))
    sentiment_filter = st.radio("Filter by sentiment", ["All", "Positive", "Negative", "Neutral"], horizontal=True)

    subset = df[df["cluster_label"] == chosen_cluster]
    if sentiment_filter != "All":
        subset = subset[subset["sentiment"] == sentiment_filter]

    st.markdown(f"**{len(subset):,} reviews** in this cluster/filter")
    sample_reviews = subset["review_text"].dropna().sample(min(10, len(subset)), random_state=42)
    for i, review in enumerate(sample_reviews, 1):
        st.markdown(f"{i}. {review}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — SEMANTIC SEARCH
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Semantic Search":
    st.title("🔍 Semantic Search")
    st.markdown(
        "Type any query and find the **most semantically similar reviews** "
        "from the dataset. Uses `all-MiniLM-L6-v2` sentence embeddings."
    )

    query = st.text_input("Enter your query:", placeholder="e.g. app crashes after update")
    top_k = st.slider("Number of results", min_value=3, max_value=20, value=10)

    if query.strip():
        with st.spinner("Searching..."):
            query_vec = embedder.encode([query], normalize_embeddings=True)
            sims = cosine_similarity(query_vec, embeddings)[0]
            top_indices = sims.argsort()[::-1][:top_k]

        st.markdown(f"**Top {top_k} results for:** _{query}_")
        results = df_sample.iloc[top_indices][["review_text", "sentiment", "cluster_label"]].copy()
        results["similarity"] = sims[top_indices].round(3)
        results = results.reset_index(drop=True)
        results.index += 1

        # Colour-code sentiment
        def sentiment_badge(s):
            colour = {"Positive": "green", "Negative": "red", "Neutral": "grey"}.get(s, "grey")
            return f'<span style="color:{colour};font-weight:bold">{s}</span>'

        for _, row in results.iterrows():
            st.markdown(
                f"**[{row['similarity']:.3f}]** "
                f"{sentiment_badge(row['sentiment'])} · "
                f"*{row['cluster_label']}*<br>{row['review_text']}",
                unsafe_allow_html=True,
            )
            st.markdown("---")
    else:
        st.info("Enter a query above to search.")
