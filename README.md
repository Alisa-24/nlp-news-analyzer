# NLP News Intelligence

A Python project that collects BBC news and analyzes each article to identify
organizations, topics, sentiment, and possible environmental incidents.

## Project structure

```text
nlp-scraper/
├── scraper_news.py               # Scraper for BBC news articles
├── nlp_enriched_news.py          # nlp pipeline for topics, sentiment, and environmental scoring
├── main.py                       # Starter script
├── bbc_news_train.csv
├── bbc_news_tests.csv
├── data/                         # Scraped articles
│   └── YYYY-MM-DD/
│       └── <article-id>.json
├── results/
│   ├── training_model.py
│   ├── topic_classifier.pkl
│   ├── learning_curves.png
│   ├── topic_metrics.json
│   ├── enhanced_news.csv         # Current NLP output
│   └── environmental_ranking.json # Earlier ranking export
├── pyproject.toml
├── requirements.txt
├── uv.lock
└── README.md
```

## Setup and run

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/). Run these commands
from the project folder:

```powershell
uv sync
uv run python scraper_news.py
uv run python nlp_enriched_news.py
```

The scraper saves articles in `data/YYYY-MM-DD/<id>.json`. The NLP script reads
those files and saves the results to `results/enhanced_news.csv`.
It prints progress for each article and leaves the source JSON files unchanged.

The scraper collects up to 300 new articles from the past seven days per run.
It removes duplicates by URL or normalized article text. RSS may provide fewer
than 300 articles; later runs can collect new stories.

Pretrained resources download when needed. If the topic model is missing,
it is automatically trained on `bbc_news_train.csv` and evaluated on
`bbc_news_tests.csv`.

## What the NLP pipeline does

All analysis is handled by the `NewsNLP` class in `nlp_enriched_news.py`:

1. **Preprocessing:** spaCy normalizes text, removes stop words and punctuation,
   and produces lemmas, tokens, and sentences.
2. **Entities:** spaCy NER detects `ORG` mentions in the original headline and body.
3. **Topics:** TF-IDF and logistic regression predict Tech, Sport, Business,
   Entertainment, or Politics.
4. **Sentiment:** NLTK VADER predicts positive, negative, or neutral.
5. **Environmental incidents:** sentence embeddings score relevant organization-linked
   sentences against specific disaster phrases.

`process_article()` returns the analysis for one article.
`process_all_articles()` processes the saved collection, ranks candidates,
and exports the CSV.

## CSV output

The CSV contains exactly seven columns:

| Column | Description |
| --- | --- |
| `id` | Unique article ID |
| `article` | Original headline and full article body |
| `entities` | Detected organization names, stored as a JSON list |
| `topic` | Predicted news category |
| `sentiment` | Positive, negative, or neutral label |
| `scandal_score` | Environmental relevance score from 0 to 1; higher means a closer match |
| `top_10` | Whether the article is a top-ten qualifying candidate |

## Topic model

The supplied datasets are `bbc_news_train.csv` and `bbc_news_tests.csv`.
Training removes duplicate and overlapping training texts in memory without
changing either CSV. The saved model achieved **98.37% test accuracy**
(723/735 correct) and **97.65% five-fold validation accuracy**.

To retrain and regenerate the learning curves:

```powershell
uv run python results/training_model.py
```

The script saves:

- `results/topic_classifier.pkl`: TF-IDF vectorizer and logistic regression together.
- `results/learning_curves.png`: training and validation learning curves.
- `results/topic_metrics.json`: evaluation scores and data checks.

## Environmental scoring

The keywords are specific phrases such as **oil spill**, **toxic wastewater
discharge**, **illegal logging**, and **hazardous gas leak**. They cover water
pollution, air pollution, deforestation, soil contamination, marine damage,
and industrial disasters. Standalone words such as fire, gas, or disaster
are not enough to trigger a match.

We use [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
because it is a small pretrained sentence-embedding model that runs on CPU
and captures meaning beyond exact word matches.

Only sentences containing a detected organization and qualified incident
language are scored. **Cosine similarity** compares their embeddings with
the disaster-phrase embeddings. It compares semantic direction rather than
vector magnitude, so related descriptions can match despite different wording.

The article score is its strongest sentence/phrase match. Internally,
`scandal_distance = 1 - scandal_score`. Up to ten articles with supporting
evidence and a score of at least **0.5** are flagged after all articles are
processed. This threshold is configurable with `--min-score`; fewer than ten
articles are flagged if fewer qualify.

These are review candidates, not proof of company responsibility. ORG also
includes institutions and teams, and the model can misinterpret context.

## Main files

| File | Purpose |
| --- | --- |
| `scraper_news.py` | Download full BBC articles and remove duplicates |
| `nlp_enriched_news.py` | Run NLP analysis, show progress, and export the CSV |
| `results/training_model.py` | Train and evaluate the topic classifier |

Use `--help` on either main script for input paths, output paths, and other options.
