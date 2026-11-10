"""Read BBC RSS feeds, download full articles, and save unique JSON records."""

import argparse
import calendar
import hashlib
import json
import math
import time
import unicodedata
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import feedparser
import requests
import trafilatura

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "data"
DEFAULT_FEEDS = tuple(
    f"https://feeds.bbci.co.uk/news/{section}rss.xml"
    for section in (
        "",
        "world/",
        "uk/",
        "business/",
        "politics/",
        "technology/",
        "science_and_environment/",
        "health/",
        "education/",
        "entertainment_and_arts/",
        "england/",
        "scotland/",
        "wales/",
        "northern_ireland/",
        "world/africa/",
        "world/asia/",
        "world/europe/",
        "world/latin_america/",
        "world/middle_east/",
        "world/us_and_canada/",
    )
) + ("https://feeds.bbci.co.uk/sport/rss.xml",)


def canonical_url(url):
    """Treat BBC domains and links with tracking parameters as the same URL."""
    parts = urlsplit(url)
    if (
        parts.scheme not in {"http", "https"}
        or parts.hostname
        not in {
            "bbc.co.uk",
            "www.bbc.co.uk",
            "bbc.com",
            "www.bbc.com",
        }
        or not parts.path.startswith(("/news/", "/sport/"))
    ):
        raise ValueError(f"Not a BBC article URL: {url}")
    return urlunsplit(("https", "www.bbc.co.uk", parts.path.rstrip("/"), "", ""))


def body_hash(body):
    normalized = " ".join(unicodedata.normalize("NFKC", body).casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def remove_duplicates(output_dir):
    """Delete duplicate BBC records and return the remaining URL/text indexes."""
    output_dir = Path(output_dir)
    urls, bodies = set(), set()
    removed = 0
    for path in sorted(output_dir.glob("*/*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(record, dict) or record.get("source") != "BBC News":
                continue
            if not all(
                isinstance(record.get(key), str) and record[key].strip()
                for key in ("id", "url", "date", "date_scraped", "headline", "body")
            ):
                raise ValueError("Incomplete article record")
            url, fingerprint = canonical_url(record["url"]), body_hash(record["body"])
        except (OSError, ValueError, TypeError) as exc:
            print(f"Leaving unreadable file {path}: {exc}")
            continue
        if url in urls or fingerprint in bodies:
            path.unlink()
            removed += 1
        else:
            urls.add(url)
            bodies.add(fingerprint)
    print(f"Removed {removed} duplicates; {len(urls)} stored articles remain.")
    return urls, bodies


def save_article(output_dir, url, published, headline, body):
    output_dir = Path(output_dir)
    article_id = str(uuid.uuid5(uuid.NAMESPACE_URL, url))
    record = {
        "id": article_id,
        "url": url,
        "date": published.isoformat(),
        "date_scraped": datetime.now(UTC).isoformat(),
        "headline": headline,
        "body": body,
        "source": "BBC News",
    }
    folder = output_dir / published.date().isoformat()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{article_id}.json"
    temporary = path.with_suffix(".tmp")
    try:
        temporary.write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def validate_options(days, max_articles, delay, timeout):
    """Reject invalid request settings before accessing stored articles."""
    if days < 1 or max_articles < 1:
        raise ValueError("--days and --max-articles must be positive")
    if not math.isfinite(delay) or delay < 0 or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError(
            "--delay must be finite and nonnegative; --timeout must be finite and positive"
        )


def scrape(
    output_dir=DEFAULT_OUTPUT, feeds=DEFAULT_FEEDS, days=7, max_articles=300, delay=0.5, timeout=25
):
    validate_options(days, max_articles, delay, timeout)
    output_dir = Path(output_dir)
    urls, bodies = remove_duplicates(output_dir)
    attempted = set(urls)
    now = datetime.now(UTC)
    cutoff = now - timedelta(days=days)
    saved = skipped = failed = successful_feeds = 0

    with requests.Session() as session:
        session.headers["User-Agent"] = "BBCNewsResearchScraper/0.1"
        for feed_url in feeds:
            if saved >= max_articles:
                break
            print(f"Reading feed: {feed_url}")
            try:
                response = session.get(feed_url, timeout=timeout)
                response.raise_for_status()
                entries = feedparser.parse(response.content).entries
                if not entries:
                    raise ValueError("Feed contains no readable entries")
                successful_feeds += 1
            except (requests.RequestException, ValueError) as exc:
                print(f"Skipping feed: {exc}")
                continue
            finally:
                time.sleep(delay)

            for entry in entries:
                if saved >= max_articles:
                    break
                try:
                    url = canonical_url(entry.get("link", ""))
                    headline = entry.get("title", "").strip()
                    date_parts = entry.get("published_parsed")
                    if not date_parts or not headline:
                        continue
                    published = datetime.fromtimestamp(calendar.timegm(date_parts), UTC)
                except (AttributeError, TypeError, ValueError, OverflowError):
                    continue
                if any(
                    part in urlsplit(url).path.split("/")
                    for part in ("live", "videos", "av", "sounds")
                ):
                    continue
                if not cutoff <= published <= now:
                    continue
                if url in attempted:
                    skipped += 1
                    continue
                attempted.add(url)
                print(f"Scraping: {url}")
                try:
                    response = session.get(url, timeout=timeout)
                    response.raise_for_status()
                    final_url = canonical_url(response.url)
                    if final_url in urls:
                        skipped += 1
                        continue
                    attempted.add(final_url)
                    if "html" not in response.headers.get("Content-Type", "").lower():
                        raise ValueError("Response is not an HTML article")
                    # Use HTML bytes to preserve UTF-8 even if HTTP omits a charset.
                    body = trafilatura.extract(
                        response.content, include_comments=False, include_tables=False
                    )
                    if not body or len(body.split()) < 50:
                        raise ValueError("No substantial article text found")
                    fingerprint = body_hash(body)
                    if fingerprint in bodies:
                        skipped += 1
                        continue
                    path = save_article(output_dir, final_url, published, headline, body)
                    urls.add(final_url)
                    bodies.add(fingerprint)
                    saved += 1
                    print(f"Saved in {path}")
                except (requests.RequestException, ValueError) as exc:
                    failed += 1
                    print(f"Skipping {url}: {exc}")
                finally:
                    time.sleep(delay)

    print(
        f"Finished: {saved} saved, {skipped} skipped, {failed} failed; {len(urls)} stored articles."
    )
    if len(urls) < 300:
        print("Fewer than 300 articles stored. Run again as new RSS items appear.")
    return 1 if not successful_feeds or (failed and not saved) else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument(
        "--max-articles", type=int, default=300, help="Maximum new articles per run"
    )
    parser.add_argument("--feed", action="append", help="BBC RSS URL; repeat for multiple feeds")
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds between requests")
    parser.add_argument("--timeout", type=float, default=25, help="Request timeout in seconds")
    parser.add_argument("--deduplicate-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        validate_options(args.days, args.max_articles, args.delay, args.timeout)
    except ValueError as error:
        parser.error(str(error))
    if args.deduplicate_only:
        remove_duplicates(args.output_dir.resolve())
        return 0
    return scrape(
        args.output_dir.resolve(),
        args.feed or DEFAULT_FEEDS,
        args.days,
        args.max_articles,
        args.delay,
        args.timeout,
    )


if __name__ == "__main__":
    raise SystemExit(main())
