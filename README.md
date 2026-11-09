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
│   ├── enhancend_news.csv        # Current NLP output
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
those files and saves the results to `results/enhancend_news.csv`.
It prints progress for each article and leaves the source JSON files unchanged.

The scraper collects up to 300 new articles from the past seven days per run.
It removes duplicates by URL or normalized article text. RSS may provide fewer
than 300 articles; later runs can collect new stories.

Pretrained resources download when needed. If the topic model is missing,
it is automatically trained on `bbc_news_train.csv` and evaluated on
`bbc_news_tests.csv`.

## What the NLP pipeline does

All analysis is handled by the `NewsNLP` class in `nlp_enriched_news.py`:

1. **Preprocessing:** spaCy keeps alphabetic words, removes stop words, and returns
   lowercase lemmas as one cleaned string.
2. **Entities:** spaCy NER returns a list of unique `ORG` names from the original
   headline and body.
3. **Topics:** TF-IDF and logistic regression predict Tech, Sport, Business,
   Entertainment, or Politics.
4. **Sentiment:** NLTK VADER predicts positive, negative, or neutral.
5. **Environmental incidents:** sentence embeddings score relevant organization-linked
   sentences against specific disaster phrases.

`process_article()` returns the analysis for one article.
`process_all_articles()` processes the saved collection, ranks candidates,
and exports the CSV.

## CSV output

The CSV contains these ten columns. `process_all_articles()` returns the same
DataFrame, with Python lists in `org` and `topics` and date values in `date_scraped`.
In the CSV, lists are stored as JSON and dates use `YYYY-MM-DD`.

| Column | Description |
| --- | --- |
| `id` | Unique article ID |
| `url` | Original article URL |
| `date_scraped` | Date the article was scraped, from its saved timestamp |
| `headline` | Original headline |
| `body` | Full original article body |
| `org` | List of detected organization names |
| `topics` | List containing the predicted news category |
| `sentiment` | VADER compound score from -1 to +1; neutral when `-0.05 < score < 0.05` |
| `scandal_distance` | Average minimum cosine distance; lower means closer to environmental disaster phrases |
| `top_10` | Whether the article is one of the ten lowest-distance review candidates |

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
and industrial disasters. These specific phrases are the embedding references;
standalone words such as fire, gas, or disaster are not used as reference phrases.

We use [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
because it is a small pretrained sentence-embedding model that runs on CPU
and captures meaning beyond exact word matches.

Every sentence containing a detected organization is scored, without requiring
an exact keyword or regex match. **Cosine similarity** compares its embedding with
the disaster-phrase embeddings. It compares semantic direction rather than
vector magnitude, so related descriptions can match despite different wording.

For each ORG-containing sentence, the model finds the minimum cosine distance
to any disaster phrase. `scandal_distance` is the average of those sentence
distances. Long sentences can use overlapping windows; only their best window
contributes, so each original sentence counts once.

The CSV stores `scandal_distance` directly, so a lower value means stronger
relevance. Articles without organizations or organization-containing sentences
get distance 1 and no involved organizations.

`detect_scandal(text, entities)` returns the average distance and organization
names from sentences with distance below 0.4 (similarity above 0.6). It matches
whole organization names without case sensitivity, so `AI` does not match `chair`. This strict
entity threshold is separate from the article ranking and may return an empty
list even when an article appears in the review shortlist.

After all articles are processed, the
ten lowest-distance articles with ORG-containing sentence evidence are flagged.
There is no fixed similarity cutoff. If fewer than ten articles contain an
ORG sentence, only those articles can be flagged. Ties keep input order.

These are review candidates, not proof of company responsibility. ORG also
includes institutions and teams, and the model can misinterpret context.

## Main files

| File | Purpose |
| --- | --- |
| `scraper_news.py` | Download full BBC articles and remove duplicates |
| `nlp_enriched_news.py` | Run NLP analysis, show progress, and export the CSV |
| `results/training_model.py` | Train and evaluate the topic classifier |

Use `--help` on either main script for input paths, output paths, and other options.
