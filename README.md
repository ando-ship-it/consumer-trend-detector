# Consumer Trend Signal Detector

AI-powered system that detects early signals of emerging consumer trends from user-generated app reviews using NLP and time-series analysis.

## Project goal

Identify topics in product reviews that are starting to grow before they become mainstream — helping businesses adapt product and marketing strategies earlier.

## Tech stack

- **NLP:** TF-IDF, sentence-transformers (all-MiniLM-L6-v2), cosine similarity
- **Modeling:** scikit-learn (Logistic Regression, topic modeling)
- **Data:** Google Play Store reviews (scraped via google-play-scraper)
- **Visualization:** matplotlib
- **Planned:** BERTopic, time-series trend detection, Streamlit dashboard

## Structure

```
consumer-trend-detector/
├── data/               # Raw and processed datasets (not tracked in git)
├── notebooks/          # Jupyter notebooks by day
│   └── 01_eda_baseline_consumer_trends.ipynb
├── outputs/            # Saved model outputs and metrics (not tracked in git)
├── requirements.txt
└── README.md
```

## Progress

| Day | Topic | Status |
|-----|-------|--------|
| 1–2 | EDA, text cleaning, TF-IDF baseline (Accuracy 67%, Macro F1 0.60) | ✅ |
| 3 | Improved preprocessing, error analysis, app-level differences, sentence embeddings, semantic search | ✅ |
| 4 | Topic modeling (LDA), KMeans k=6 clustering | ✅ |
| 5 | Time-series topic share, trend detection (growing/declining clusters) | ✅ |
| 6 | KMeans k=12 final model, cluster labeling, trend score table | ✅ |
| 7 | Binary TF-IDF+LR (F1 0.85), Embeddings+LR (F1 0.86), RoBERTa zero-shot (F1 0.91) — model comparison | ✅ |
| 8 | Save all artifacts (embeddings, clusters, trend scores), initial Streamlit app | ✅ |
| 9 | UMAP 2D visualization, single-page dark Plotly dashboard, fix Period index bug | ✅ |
| 10 | Streamlit Cloud deploy, requirements update, README polish | ✅ |
| 11+ | Multi-app comparison, LLM-based cluster naming, PDF export | 🔜 |

## Setup

```bash
pip install -r requirements.txt
```

Data file (`reviews.csv`) is not included in the repo — place it in `data/`.

## Dashboard

The Streamlit app (`app.py`) is a single-page dark dashboard with:
- **Trend Signals** — ranked bar chart + interactive time-series per cluster
- **Cluster Map** — UMAP 2D scatter + review explorer with sentiment filter
- **Semantic Search** — query input with cosine similarity ranking

Run locally:
```bash
python3 -m streamlit run app.py
```

Live demo: *(deploy to Streamlit Cloud — connect `ando-ship-it/consumer-trend-detector`, main file `app.py`)*

![Dashboard](assets/dashboard_screenshot.png)
