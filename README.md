# NLP News Intelligence

Collect BBC news articles and enrich them with organization names, topics,
sentiment scores, and an environmental-incident review shortlist.

The project combines RSS collection, full-article extraction, spaCy, a trained
TF-IDF topic classifier, VADER sentiment analysis, and sentence embeddings.

## Quick start

Requires **Python 3.12+** and **uv**. From the project folder, run:

```bash
uv sync
uv run python scraper_news.py
uv run python nlp_enriched_news.py
```

1. `uv sync` installs the dependencies and the spaCy English model.
2. The scraper saves articles to `data/YYYY-MM-DD/<article-id>.json`.
3. The NLP pipeline reads those articles and exports `results/enhancend_news.csv`.

The pipeline prints progress and leaves the source JSON files unchanged.
`uv run python main.py` is an alternative entry point for the same NLP command.

The VADER lexicon and sentence-embedding model download when first needed.
If the topic classifier is missing, the pipeline trains it using the included datasets.

## Usage

### Collect articles

By default, the scraper reads BBC news and sport RSS feeds and saves up to
**300 new articles from the past seven days**. It removes duplicate stored
records by normalized URL or article text and skips articles already collected.
Redirected articles are saved under their final normalized BBC URL.

```bash
uv run python scraper_news.py --days 3 --max-articles 100 --delay 1
```

To use a specific feed and output folder:

```bash
uv run python scraper_news.py --feed https://feeds.bbci.co.uk/news/business/rss.xml --output-dir data
```

Repeat `--feed` to read multiple feeds. RSS feeds may contain fewer articles than
requested; later runs can collect newly published stories.

### Analyze articles

```bash
uv run python nlp_enriched_news.py
```

To use custom input and output paths:

```bash
uv run python nlp_enriched_news.py --input-dir data --output-file results/news_analysis.csv
```

Input articles must be stored in dated folders, such as `data/2026-11-03/`.
The output must be a CSV outside the input folder and cannot overwrite either
training dataset. Invalid article records are skipped and reported in the progress log.

Use `--help` to see the options for each command:

```bash
uv run python scraper_news.py --help
uv run python nlp_enriched_news.py --help
uv run python results/training_model.py --help
```

## Analysis pipeline

| Step | Method | Result |
| --- | --- | --- |
| Text preprocessing | spaCy lemmatization and stop-word removal | Cleaned headline and body |
| Organization extraction | spaCy named entity recognition | Unique `ORG` names from the original text |
| Topic classification | TF-IDF and logistic regression | Business, Entertainment, Politics, Sport, or Tech |
| Sentiment analysis | NLTK VADER on the original text | Sentiment label and compound score |
| Environmental scoring | Sentence embeddings and cosine distance | Organization-linked evidence and review ranking |

`NewsNLP.process_article()` returns the detailed analysis for a single article.
`NewsNLP.process_all_articles()` processes the collection, ranks review candidates,
exports the CSV, and returns a pandas DataFrame.

## CSV output

The default output is **`results/enhancend_news.csv`**. The filename retains the
project's existing spelling; use `--output-file` to choose another name.

| Column | Description |
| --- | --- |
| `id` | Unique article ID |
| `url` | Stored article URL |
| `date_scraped` | Collection date in `YYYY-MM-DD` format |
| `headline` | Original headline |
| `body` | Full extracted article text |
| `org` | Detected organization names, stored as a JSON list |
| `topics` | Predicted topic, stored as a one-item JSON list |
| `sentiment` | VADER compound score from -1 to +1 |
| `scandal_distance` | Average minimum cosine distance to environmental-disaster phrases; lower means more similar |
| `top_10` | Whether the article is included in the environmental review shortlist |

For sentiment, scores of at least `0.05` are positive, scores of at most `-0.05`
are negative, and scores between those thresholds are neutral. The CSV stores
the numeric score; the detailed article analysis also includes the label.

The returned DataFrame retains Python lists in `org` and `topics` and date values
in `date_scraped`. CSV export serializes the lists as JSON.

## Environmental scoring

The pipeline uses `sentence-transformers/all-MiniLM-L6-v2` to compare sentences
containing detected organizations with 15 environmental-disaster phrases.
These include **oil spill**, **toxic wastewater discharge**, **illegal logging**,
and **hazardous gas leak** across six categories: water pollution, air pollution,
deforestation, soil contamination, marine damage, and industrial disasters.

For each organization-containing sentence:

1. Generate its sentence embedding.
2. Compare it with each reference phrase using cosine similarity.
3. Keep the closest phrase and calculate distance as `1 - similarity`.

The article's `scandal_distance` is the average of these sentence distances.
Long sentences use overlapping token windows, with only the best window
contributing for each original sentence. Organization matching uses whole names
and ignores case, so `AI` does not match `chair`.

Articles without organization-containing sentences receive distance `1` and are
excluded from the shortlist. Among eligible articles, the ten lowest-distance
articles receive `top_10 = True`. Fewer than ten are flagged when fewer qualify.
There is no fixed shortlist cutoff, and ties preserve input order.

The detailed analysis separately reports organizations from sentences with
distance below `0.4` through `scandal_entities`. This stricter threshold does not
determine shortlist membership.

**The shortlist supports human review.** Organization mentions and semantic
similarity do not establish responsibility for an environmental incident.
Detected organizations can also include institutions and sports teams.

## Topic classifier

The classifier predicts five categories using a TF-IDF vectorizer and logistic
regression saved together in one scikit-learn pipeline.

| Metric | Result |
| --- | --- |
| Held-out evaluation accuracy | **98.37%** — 723 of 735 articles |
| Five-fold cross-validation accuracy | **97.65%** |
| Training articles used after cleanup | **1,402** |

Training uses `bbc_news_train.csv`, and held-out evaluation uses
`bbc_news_tests.csv`. Duplicate training stories and training stories shared
with the evaluation dataset are removed in memory; the source CSVs stay unchanged.

To retrain the classifier and regenerate its evaluation artifacts:

```bash
uv run python results/training_model.py
```

This writes:

- `results/topic_classifier.pkl` — fitted vectorizer and classifier.
- `results/learning_curves.png` — training and cross-validation learning curves.
- `results/topic_metrics.json` — evaluation metrics and dataset audit.

## Project structure

```text
nlp-scraper/
├── main.py                       # Alternative NLP entry point
├── scraper_news.py               # BBC RSS collection and article extraction
├── nlp_enriched_news.py          # NLP enrichment, ranking, and CSV export
├── bbc_news_train.csv            # Topic training dataset
├── bbc_news_tests.csv            # Held-out topic evaluation dataset
├── data/                        # Collected articles; ignored by Git
│   └── YYYY-MM-DD/
│       └── <article-id>.json
├── results/
│   ├── training_model.py         # Topic model training and evaluation
│   ├── topic_classifier.pkl      # Saved topic pipeline
│   ├── topic_metrics.json        # Evaluation metrics
│   ├── learning_curves.png       # Learning curve chart
│   ├── enhancend_news.csv        # Default NLP export
│   └── environmental_ranking.json # Earlier ranking export
├── pyproject.toml               # Project metadata and dependencies
├── requirements.txt             # Pinned dependency list
├── uv.lock                      # Dependency lockfile
└── README.md
```
