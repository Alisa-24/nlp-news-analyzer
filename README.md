# News scraper

Collect full article text from BBC News and Sport RSS feeds for the news
intelligence project. This implements collection and duplicate removal only.

The pipeline uses three libraries: `requests` downloads RSS feeds and article
pages, `feedparser` reads feed entries, and `trafilatura` extracts the full
article text from HTML. A few Python functions handle JSON storage and duplicates.

## Run

From the project directory, install the dependencies and run:

```powershell
uv sync
uv run python scraper_news.py
```

Alternatively, install `requirements.txt` into your Python environment with
`python -m pip install -r requirements.txt`, then run `python scraper_news.py`.

By default it checks 21 BBC section feeds, considers articles published within
the last seven days, and saves up to 300 **new** articles per run. Requests are
sequential, spaced by 0.5 seconds, and use a 25-second timeout.
Individual feed or article failures are logged and skipped without automatic retries.
Live blogs, audio/video URLs and pages with fewer than 50 extracted words are
skipped. RSS descriptions are never used as a substitute for article text.

RSS is a recent snapshot, not a searchable archive. A single run may provide
fewer than 300 usable, unique articles. Run again as new items appear to build
the corpus; the scraper stops after processing the available items. Older
stored articles are retained when collecting newer articles.
Feeds and their entries are processed in their listed order.

```powershell
# Small collection from one feed
uv run python scraper_news.py --feed https://feeds.bbci.co.uk/news/rss.xml --max-articles 5

# Change the destination or publication lookback
uv run python scraper_news.py --output-dir data/bbc --days 7

# Clean existing scraper records without downloading anything
uv run python scraper_news.py --deduplicate-only
```

Use `--help` for all options. Repeat `--feed` to select multiple feeds.

## Stored articles

UTF-8 JSON files are grouped by publication date in UTC:

```text
data/
  YYYY-MM-DD/
    <stable-article-id>.json
```

Each record contains `id` (a deterministic UUID), `url`, `date` (publication
timestamp), `date_scraped` (collection timestamp), `headline`, `body` (full
article text with paragraph breaks), and `source` (`BBC News`). This keeps text
and metadata together for a later NLP pipeline. The data folder is Git-ignored.
The classifier can later read `headline` and `body` from these files; topic
labels and model training are separate steps. The existing 300 records use
the same format and remain compatible with this simpler scraper.

At startup the scraper scans these dated folders and deletes redundant BBC
records, keeping the first file in sorted path order. Duplicate detection uses
either the canonical article URL (ignoring tracking parameters, fragments and
BBC `.com`/`.co.uk` host differences) or a SHA-256 hash of the body after Unicode,
case and whitespace normalization. It also checks these keys before every save,
so reruns skip stored URLs and identical text across different URLs. This is
exact normalized-text matching; similar reporting on the same event is kept.
Existing URLs are kept as first collected, so later edits are not refreshed.
Malformed JSON and files without the scraper's `source` marker are left alone.