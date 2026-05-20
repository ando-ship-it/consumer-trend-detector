"""
Consumer Trend Signal Detector — Streamlit Dashboard (Day 9)
Single-page layout with dark/clean styling.
"""

import os
import re
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
  .stApp { background: #f8fafc; }

  /* top banner */
  .banner {
    background: linear-gradient(135deg, #1e40af 0%, #2563eb 100%);
    border-radius: 12px;
    padding: 28px 36px 20px;
    margin-bottom: 24px;
  }
  .banner h1 { color: #ffffff; font-size: 2rem; margin: 0 0 4px; }
  .banner p  { color: #bfdbfe; margin: 0; font-size: 0.95rem; }

  /* section headers */
  .section-title {
    color: #1e40af;
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin: 32px 0 12px;
    border-left: 3px solid #2563eb;
    padding-left: 10px;
  }

  /* metric cards */
  div[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 16px 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
  }
  div[data-testid="metric-container"] label { color: #64748b !important; font-size: 0.8rem !important; }
  div[data-testid="metric-container"] div[data-testid="stMetricValue"] { color: #1e293b !important; font-size: 1.6rem !important; }

  /* search box */
  .stTextInput input {
    background: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    color: #1e293b !important;
    border-radius: 8px !important;
  }

  /* review card */
  .review-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 8px;
    font-size: 0.88rem;
    color: #334155;
    line-height: 1.5;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
  }
  .badge-pos { color: #16a34a; font-weight: 700; }
  .badge-neg { color: #dc2626; font-weight: 700; }
  .badge-neu { color: #64748b; font-weight: 700; }
  .sim-score { color: #2563eb; font-weight: 600; }
  .cluster-tag { color: #7c3aed; font-style: italic; font-size: 0.82rem; }

  /* divider */
  hr { border-color: #e2e8f0 !important; }

  /* insight cards */
  .insight-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 12px 14px;
    margin-bottom: 8px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
  }
  .insight-cluster { color: #1e293b; font-weight: 600; font-size: 0.9rem; margin-bottom: 2px; }
  .insight-growth  { font-size: 0.82rem; font-weight: 600; margin-bottom: 4px; }
  .insight-action  { color: #64748b; font-size: 0.82rem; line-height: 1.4; }
  .insight-header  { color: #1e293b; font-weight: 700; margin-bottom: 4px; }
  .insight-caption { color: #94a3b8; font-size: 0.78rem; margin-bottom: 10px; }

  /* selectbox / multiselect */
  div[data-baseweb="select"] > div {
    background: #ffffff !important;
    border-color: #cbd5e1 !important;
    color: #1e293b !important;
  }
</style>
""", unsafe_allow_html=True)

# ── Load artifacts ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df      = pd.read_csv("outputs/df_clustered.csv")
    trend   = pd.read_csv("outputs/trend_score.csv")
    cbm     = pd.read_csv("outputs/cluster_by_month.csv", index_col="month")
    cbm     = cbm[cbm.index >= "2016-03-01"]
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

# ── Cluster type classification & action points ───────────────────────────────
CLUSTER_TYPE = {
    "Task management":            "feature",
    "Premium / paywall friction": "friction",
    "General positive":           "satisfaction",
    "Ease of use":                "satisfaction",
    "Reminders & notifications":  "feature",
    "Habit tracking":             "feature",
    "Ads friction":               "friction",
    "App stability & updates":    "friction",
    "General task/app praise":    "satisfaction",
    "Sync & login issues":        "friction",
    "Calendar integration":       "feature",
    "Widget issues":              "friction",
}

ACTION_POINTS = {
    "Premium / paywall friction": "Review pricing tiers. Consider free trial or granular feature unlocks.",
    "Ads friction":               "A/B test less intrusive ad formats or cap daily ad frequency.",
    "App stability & updates":    "Prioritise regression testing and crash fixes before the next release.",
    "Sync & login issues":        "Audit OAuth flows and offline-sync reliability.",
    "Widget issues":              "Fix widget reliability before expanding widget features.",
    "Calendar integration":       "Invest in bidirectional Google Calendar / Outlook sync.",
    "Habit tracking":             "Consider streaks, progress charts, and smart reminders.",
    "Task management":            "Focus on UX polish rather than new capabilities.",
    "Reminders & notifications":  "Add granular notification controls and quiet hours.",
    "General positive":           "Protect what is already working — don't over-engineer.",
    "Ease of use":                "Maintain simplicity when adding new features.",
    "General task/app praise":    "Good baseline signal — track over time for shifts.",
}

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
neu_pct = round((df["sentiment"] == "Neutral").mean() * 100, 1)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total reviews",  f"{len(df):,}")
c2.metric("Topic clusters", "12")
c3.metric("Positive",       f"{pos_pct}%")
c4.metric("Negative",       f"{neg_pct}%")
c5.metric("Neutral",        f"{neu_pct}%")

st.markdown("<hr>", unsafe_allow_html=True)
st.caption("📦 Dataset: **15 productivity apps** (Todoist, Microsoft To Do, TickTick, Any.do, Habitica and others) · Google Play Store · **11,190 reviews** · Feb 2015 – Oct 2020")
st.markdown("<hr>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# INSIGHT LAYER — What users love / Pain points / Wanted features
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">🧠 Product Insights</div>', unsafe_allow_html=True)

period = st.radio(
    "Signal window",
    ["3-month signal", "12-month signal", "Early vs Recent (~5yr)"],
    horizontal=True,
    help="Compare the most recent window against the prior equal-length window.",
)

PERIOD_CAPTIONS = {
    "3-month signal":       "📅 Last 3 months (Aug–Oct 2020) vs prior 3 months (May–Jul 2020).",
    "12-month signal":      "📅 Last 12 months (Nov 2019–Oct 2020) vs prior 12 months (Nov 2018–Oct 2019).",
    "Early vs Recent (~5yr)": "📅 Last third of dataset (≈2019–2020) vs first third (≈2015–2016).",
}
st.caption(PERIOD_CAPTIONS[period] + "  Red = needs attention. Green = moving in the right direction.")

# Compute growth_pct based on selected window (all windows use absolute pp change × 100)
ts_insight = trend_score_df.copy()

if period == "3-month signal":
    recent_w = cluster_by_month.iloc[-3:].mean()
    prior_w  = cluster_by_month.iloc[-6:-3].mean()
    def _pct(cluster):
        r = recent_w.get(cluster, 0)
        p = prior_w.get(cluster, 0)
        return round((r - p) * 100, 1)
    ts_insight["growth_pct"] = ts_insight["cluster"].map(_pct)

elif period == "12-month signal":
    recent_w = cluster_by_month.iloc[-12:].mean()
    prior_w  = cluster_by_month.iloc[-24:-12].mean()
    def _pct(cluster):
        r = recent_w.get(cluster, 0)
        p = prior_w.get(cluster, 0)
        return round((r - p) * 100, 1)
    ts_insight["growth_pct"] = ts_insight["cluster"].map(_pct)

else:  # Early vs Recent
    ts_insight["growth_pct"] = (
        (ts_insight["share_recent"] - ts_insight["share_early"]) * 100
    ).round(1)

ts_insight["type"] = ts_insight["cluster"].map(CLUSTER_TYPE)

satisfaction_df = ts_insight[ts_insight["type"] == "satisfaction"].sort_values("share_recent", ascending=False)
friction_df     = ts_insight[ts_insight["type"] == "friction"].sort_values("trend_score", ascending=False)
feature_df      = ts_insight[ts_insight["type"] == "feature"].sort_values("trend_score", ascending=False)

def insight_card(cluster, growth_pct, action, cluster_type="feature"):
    arrow = "↑" if growth_pct > 0 else ("↓" if growth_pct < 0 else "→")
    # Red = needs attention. Green = moving in the right direction.
    # Satisfaction: growth is good (green), decline is bad (red)
    # Friction:     growth is bad (red),  decline is good (green)
    # Feature:      growth = high demand, needs investment (red); decline = neutral (gray)
    if cluster_type == "satisfaction":
        color = "#4ade80" if growth_pct > 0 else ("#f87171" if growth_pct < 0 else "#94a3b8")
    elif cluster_type == "friction":
        color = "#f87171" if growth_pct > 0 else ("#4ade80" if growth_pct < 0 else "#94a3b8")
    else:  # feature
        color = "#f87171" if growth_pct > 0 else "#94a3b8"
    return (
        f'<div class="insight-card">'
        f'<div class="insight-cluster">{cluster}</div>'
        f'<div class="insight-growth" style="color:{color}">{arrow} {abs(growth_pct)}% vs. prior period</div>'
        f'<div class="insight-action">→ {action}</div>'
        f'</div>'
    )

love_col, pain_col, feat_col = st.columns(3)

with love_col:
    st.markdown('<div class="insight-header">✅ What users already love</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight-caption">High-share satisfaction clusters — don\'t break these.</div>', unsafe_allow_html=True)
    for _, row in satisfaction_df.iterrows():
        action = ACTION_POINTS.get(row["cluster"], "")
        st.markdown(insight_card(row["cluster"], row["growth_pct"], action, "satisfaction"), unsafe_allow_html=True)

with pain_col:
    st.markdown('<div class="insight-header">🚨 Pain points</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight-caption">Friction clusters — red = getting worse, green = improving.</div>', unsafe_allow_html=True)
    for _, row in friction_df.iterrows():
        action = ACTION_POINTS.get(row["cluster"], "")
        st.markdown(insight_card(row["cluster"], row["growth_pct"], action, "friction"), unsafe_allow_html=True)

with feat_col:
    st.markdown('<div class="insight-header">💡 Feature demand</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight-caption">Growing feature clusters — invest here next.</div>', unsafe_allow_html=True)
    for _, row in feature_df.iterrows():
        action = ACTION_POINTS.get(row["cluster"], "")
        st.markdown(insight_card(row["cluster"], row["growth_pct"], action, "feature"), unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 1 — Trend signals + Time-series
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">📈 Trend Signals</div>', unsafe_allow_html=True)

left, right = st.columns([1, 2])

with left:
    ts = trend_score_df.copy()
    ts["growth_pct"] = (ts["growth_rate"] * 100).round(1)
    ts["neg_pct"]    = (ts["neg_share"]   * 100).round(1)
    ts = ts.sort_values("growth_pct")

    # Color: growing + high-neg → red; growing + low-neg → teal; declining → gray
    def bar_color(row):
        if row["growth_pct"] <= 0:
            return "#94a3b8"  # gray — declining
        return f"rgba({int(239 * row['neg_share'])}, {int(68 * (1-row['neg_share']))}, {int(68 * row['neg_share'])}, 0.85)"
    ts["color"] = ts.apply(bar_color, axis=1)

    fig_trend = go.Figure(go.Bar(
        x=ts["growth_pct"],
        y=ts["cluster"],
        orientation="h",
        marker=dict(color=ts["color"]),
        customdata=ts[["neg_pct", "growth_pct"]].values,
        hovertemplate="<b>%{y}</b><br>Share change: %{x:+.1f}%<br>Negative reviews: %{customdata[0]:.0f}%<extra></extra>",
    ))
    fig_trend.update_layout(
        height=380, margin=dict(l=0, r=10, t=30, b=10),
        paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc",
        title=dict(text="Share change: early period → recent period", font=dict(size=12, color="#475569"), x=0),
        xaxis=dict(title="Δ share of reviews (%)", color="#64748b", gridcolor="#e2e8f0", ticksuffix="%",
                   zeroline=True, zerolinecolor="#334155", zerolinewidth=1),
        yaxis=dict(color="#334155", tickfont=dict(size=11)),
        font=dict(color="#334155"),
    )
    st.plotly_chart(fig_trend, use_container_width=True)
    st.caption("Bar = how much this topic **grew or shrank** as a share of all reviews (2015–2016 vs 2019–2020). **Color**: red = mostly negative reviews, gray = declining topic.")

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
            paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc",
            xaxis=dict(tickvals=tick_vals, tickangle=-45, color="#64748b",
                       gridcolor="#e2e8f0", tickfont=dict(size=10)),
            yaxis=dict(tickformat=".0%", color="#64748b", gridcolor="#e2e8f0"),
            legend=dict(bgcolor="#ffffff", bordercolor="#e2e8f0",
                        font=dict(color="#334155", size=11)),
            font=dict(color="#334155"),
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

map_col, exp_col = st.columns([1, 2])

with map_col:
    umap_df["snippet"] = umap_df["review_text"].str[:80].fillna("") + "…"
    fig_umap = px.scatter(
        umap_df, x="x", y="y",
        color="cluster_label",
        hover_data={"snippet": True, "sentiment": True, "x": False, "y": False},
        color_discrete_sequence=px.colors.qualitative.Plotly,
        opacity=0.7,
        template="plotly_white",
    )
    fig_umap.update_traces(marker=dict(size=3))
    fig_umap.update_layout(
        height=400, margin=dict(l=0, r=0, t=10, b=120),
        paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc",
        legend=dict(
            title="", orientation="h",
            x=0, y=-0.05, yanchor="top",
            bgcolor="#ffffff", font=dict(color="#334155", size=9),
        ),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    st.plotly_chart(fig_umap, use_container_width=True)
    st.caption("Each dot = one review. Semantically similar reviews cluster together.")

with exp_col:
    chosen = st.selectbox("Topic cluster", sorted(df["cluster_label"].unique()), label_visibility="visible")
    sent_f = st.radio("Sentiment", ["All", "Positive", "Negative", "Neutral"], horizontal=True)

    sub = df[df["cluster_label"] == chosen]
    if sent_f != "All":
        sub = sub[sub["sentiment"] == sent_f]

    pos_c = (sub["sentiment"] == "Positive").sum()
    neg_c = (sub["sentiment"] == "Negative").sum()
    st.caption(f"**{len(sub):,} reviews** · {pos_c} pos / {neg_c} neg")

    samples = sub["review_text"].dropna().sample(min(5, len(sub)), random_state=42)
    for rev in samples:
        excerpt = (rev[:180] + "…") if len(rev) > 180 else rev
        st.markdown(f'<div class="review-card">{excerpt}</div>', unsafe_allow_html=True)

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

st.markdown("<hr>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 4 — Feature Requests from 3-star reviews
# ══════════════════════════════════════════════════════════════════════════════
_n3 = int((df["score"] == 3).sum())
st.markdown('<div class="section-title">💬 Feature Requests (3-star reviews)</div>', unsafe_allow_html=True)
st.caption(
    f"Analysed **{_n3:,} three-star reviews** (out of {len(df):,} total). "
    "3-star users are **disappointed but not gone** — the highest-value retention signal. "
    "⚠️ **Method: regex keywords** (wish / please add / would love…), not NLP clustering — "
    "regex targets wish-language specifically, so categories differ from Trend Signals clusters above."
)

FEATURE_RE = re.compile(
    r"wish|would be great|please add|missing|if only|would love|would like"
    r"|it would be (nice|great|helpful|awesome)|need a|add a|hope you|could you add"
    r"|want to see|feature request|i('d| would) (love|like)|lacking|bring back"
    r"|support for|option to|ability to|allow (us|me|users)",
    re.IGNORECASE,
)

@st.cache_data
def extract_feature_requests():
    subset = df[df["score"] == 3].copy()
    subset = subset[subset["review_text"].notna()]
    hits = subset[subset["review_text"].str.contains(FEATURE_RE)].copy()
    def first_match_sentence(text):
        for sent in re.split(r"(?<=[.!?])\s+", text):
            if FEATURE_RE.search(sent):
                return sent.strip()
        return text[:200]
    hits["request_sentence"] = hits["review_text"].apply(first_match_sentence)
    return hits

FEATURE_TAXONOMY = {
    "Calendar sync":              r"calendar|google cal|ical|sync.{0,10}cal",
    "Notifications & reminders":  r"notification|reminder|alarm|alert",
    "Habit & routine tracking":   r"habit|routine|streak",
    "Widget support":             r"widget",
    "Recurring tasks":            r"recur|repeat.{0,10}(daily|every day|task)|daily (task|routine)",
    "Priority & sorting":         r"priorit|sort.{0,10}task|filter.{0,10}task|arrange.{0,10}task",
    "Backup & export":            r"backup|back.?up|export|restore data|backed up",
    "Month / week view":          r"month.{0,10}view|week.{0,10}view|see.{0,15}(month|week)",
    "Offline mode":               r"offline|without internet|no internet|no wifi",
    "Themes & customization":     r"dark mode|dark theme|color.{0,10}(tag|code|theme|custom)|custom.{0,10}(color|theme|icon)",
    "Task sharing / collab":      r"shar(e|ing).{0,15}(task|list)|collaborat|team.{0,10}task",
    "Due time (not just date)":   r"due time|set time|time picker|time of day",
}

@st.cache_data
def count_feature_taxonomy(fr_df):
    rows = []
    for label, pattern in FEATURE_TAXONOMY.items():
        rx = re.compile(pattern, re.IGNORECASE)
        n = fr_df["review_text"].str.contains(rx, regex=True).sum()
        rows.append({"feature": label, "mentions": int(n)})
    result = pd.DataFrame(rows)
    result = result[result["mentions"] > 0].sort_values("mentions", ascending=True)
    return result

fr_df = extract_feature_requests()
taxonomy_df = count_feature_taxonomy(fr_df)

fr_left, fr_right = st.columns([1, 2])

with fr_left:
    fig_phrases = go.Figure(go.Bar(
        x=taxonomy_df["mentions"],
        y=taxonomy_df["feature"],
        orientation="h",
        marker=dict(color="#6366f1"),
        hovertemplate="<b>%{y}</b><br>%{x} mentions in 3-star reviews<extra></extra>",
    ))
    fig_phrases.update_layout(
        height=460, margin=dict(l=0, r=10, t=30, b=10),
        paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc",
        xaxis=dict(title="mentions in 3-star reviews", color="#64748b", gridcolor="#e2e8f0"),
        yaxis=dict(color="#334155", tickfont=dict(size=11)),
        font=dict(color="#334155"),
        title=dict(text="Most-requested features", font=dict(size=13, color="#1e293b"), x=0),
    )
    st.plotly_chart(fig_phrases, use_container_width=True)
    st.caption(f"**{len(fr_df):,}** feature-request reviews out of **{(df['score']==3).sum():,}** 3-star reviews")

with fr_right:
    # Options ordered by mention count (descending) to match the chart
    ordered_features = taxonomy_df.sort_values("mentions", ascending=False)["feature"].tolist()
    feature_sel = st.selectbox(
        "Browse example reviews by feature request",
        options=ordered_features,
        key="fr_feature",
    )
    rx_sel = re.compile(FEATURE_TAXONOMY[feature_sel], re.IGNORECASE)
    examples = fr_df[fr_df["review_text"].str.contains(rx_sel, regex=True)]["review_text"].dropna()
    examples = examples.sample(min(8, len(examples)), random_state=7)
    for sent in examples:
        st.markdown(f'<div class="review-card">{sent}</div>', unsafe_allow_html=True)

# ROW 5 — AI Key Signals Summary
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">📋 AI Key Signals Summary</div>', unsafe_allow_html=True)
st.caption("Combined read across Trend Signals (NLP clustering, all ratings) and Feature Requests (3-star reviews). Based on 11,190 reviews, Feb 2015 – Oct 2020.")

_ts = pd.read_csv("outputs/trend_score.csv")
_ts["growth_pp"]  = (_ts["growth_rate"] * 100).round(1)
_ts["neg_pct"]    = (_ts["neg_share"]   * 100).round(0).astype(int)

_growing  = _ts[_ts["growth_pp"] > 0].sort_values("growth_pp", ascending=False)
_declining = _ts[_ts["growth_pp"] < 0].sort_values("growth_pp")

sum_left, sum_right = st.columns(2)

with sum_left:
    st.markdown("**📈 Growing topics**")
    _grow_display = _growing[["cluster", "growth_pp", "neg_pct"]].rename(columns={
        "cluster":   "Cluster",
        "growth_pp": "Share Δ (pp)",
        "neg_pct":   "% negative reviews",
    }).reset_index(drop=True)
    st.dataframe(
        _grow_display.style
            .background_gradient(subset=["Share Δ (pp)"],    cmap="Greens")
            .background_gradient(subset=["% negative reviews"], cmap="Reds"),
        hide_index=True, use_container_width=True,
    )

    st.markdown("**📉 Declining topics**")
    _decl_display = _declining[["cluster", "growth_pp", "neg_pct"]].rename(columns={
        "cluster":   "Cluster",
        "growth_pp": "Share Δ (pp)",
        "neg_pct":   "% negative reviews",
    }).reset_index(drop=True)
    st.dataframe(
        _decl_display.style
            .background_gradient(subset=["Share Δ (pp)"],    cmap="Reds_r")
            .background_gradient(subset=["% negative reviews"], cmap="Reds"),
        hide_index=True, use_container_width=True,
    )

    st.markdown("**💬 Most-requested features** *(3-star reviews)*")
    _fr_counts = count_feature_taxonomy(fr_df).sort_values("mentions", ascending=False)
    _fr_top = _fr_counts.head(6).reset_index(drop=True)
    _fr_top.columns = ["Feature request", "Mentions"]
    st.dataframe(
        _fr_top.style.background_gradient(subset=["Mentions"], cmap="Purples"),
        hide_index=True, use_container_width=True,
    )
    st.caption(f"From {len(fr_df):,} feature-request reviews out of {int((df['score']==3).sum()):,} three-star reviews")

with sum_right:
    st.markdown("**💡 Product implications**")
    st.markdown("""
- **Habit tracking → competitive differentiator.**
  Growing trend (+2.9 pp) with only 12% negative reviews — users engage and ask for more
  (streaks, routines). Expanding depth here can strengthen retention among the most active users.

- **Calendar integration → highest retention risk.**
  Sharpest decline (−4.9 pp) and the #1 feature request (51 mentions, 3-star reviews).
  The gap between expectation and delivery is large — closing it has the highest upside for keeping wavering users.

- **Notifications → reliability, not new features.**
  Declining trend (−2.3 pp) and #2 feature request (46 mentions). Users want what's already
  there to work consistently, not a redesign.

- **Sync & login → most urgent technical issue.**
  −4.5 pp with 77% negative reviews. Almost everyone who writes about it is unhappy.
  This is the issue most likely to drive churn in the near term.

- **Monetisation model needs review.**
  Ads and paywall friction declining with 70–74% negative rates. Reducing ad frequency
  or improving free-tier value could recover disengaged users before they leave for good.
""")

