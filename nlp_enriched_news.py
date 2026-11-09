"""NLP analysis and environmental-disaster ranking for news articles."""

import argparse
from datetime import date
import json
from pathlib import Path
import pickle
import sys

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from spacy.matcher import PhraseMatcher
import pandas as pd


class NewsNLP:
    ENVIRONMENTAL_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    DISASTER_KEYWORDS = {
        "Water pollution": ("oil spill", "toxic wastewater discharge", "chemical contamination"),
        "Air pollution": ("toxic emissions", "hazardous gas leak", "industrial air pollution"),
        "Deforestation": ("illegal logging", "forest clearance", "unauthorized deforestation"),
        "Soil contamination": ("toxic waste dumping", "industrial soil contamination"),
        "Marine damage": ("marine oil spill", "chemical discharge into the ocean"),
        "Industrial disasters": ("industrial chemical leak", "hazardous waste spill"),
    }
    DISASTER_PHRASES = tuple(phrase for phrases in DISASTER_KEYWORDS.values() for phrase in phrases)
    PHRASE_CATEGORIES = {
        phrase: category for category, phrases in DISASTER_KEYWORDS.items() for phrase in phrases
    }

    TOPICS = {
        "tech": "Tech",
        "sport": "Sport",
        "business": "Business",
        "entertainment": "Entertainment",
        "politics": "Politics",
    }

    def __init__(self, spacy_model="en_core_web_sm", topic_model_path=None, verbose=False):
        self.verbose = verbose
        self._log(f"Loading spaCy model: {spacy_model}...")
        self.nlp = spacy.load(spacy_model, exclude=["parser"])
        self.nlp.add_pipe("sentencizer")
        self._log("spaCy model ready.")

        if topic_model_path is None:
            topic_model_path = Path(__file__).resolve().parent / "results" / "topic_classifier.pkl"
        self.topic_model_path = Path(topic_model_path)

        # Load these models only when they are needed, then reuse them.
        self.topic_classifier = None
        self.sentiment_analyzer = None
        self.environmental_model = None
        self.disaster_embeddings = None

    def _log(self, message):
        if self.verbose:
            print(message, flush=True)

    def preprocess_text(self, text):
        document = self.nlp(text, disable=["ner"])
        return " ".join(
            token.lemma_.lower() for token in document
            if token.is_alpha and not token.is_stop
        )

    def detect_entities(self, headline, body):
        organizations = []
        for document in self.nlp.pipe(
            [headline, body], disable=["tagger", "attribute_ruler", "lemmatizer"]
        ):
            for entity in document.ents:
                if entity.label_ == "ORG" and entity.text not in organizations:
                    organizations.append(entity.text)
        return organizations

    def load_topic_classifier(self, model_path=None):
        path = self.topic_model_path if model_path is None else Path(model_path)

        # The training script saves the vectorizer and classifier together.
        if not path.is_file():
            from results.training_model import ROOT, train_model

            print(f"Topic model not found: {path}. Training it now...", flush=True)
            train_model(
                train_path=ROOT / "bbc_news_train.csv",
                test_path=ROOT / "bbc_news_tests.csv",
                output_dir=path.parent,
                model_filename=path.name,
            )

        self._log(f"  Loading topic model: {path}")
        with path.open("rb") as model_file:
            classifier = pickle.load(model_file)

        predict = getattr(classifier, "predict", None)
        if not callable(predict):
            raise ValueError("The saved topic model must have a predict method.")

        self.topic_classifier = classifier
        self.topic_model_path = path
        self._log("  Topic model ready.")

    def detect_topic(self, headline, body):
        text = f"{headline}\n\n{body}".strip()
        if not text:
            raise ValueError("Topic prediction requires a headline or article body.")
        if self.topic_classifier is None:
            self.load_topic_classifier()

        prediction = self.topic_classifier.predict([text])[0]
        category = str(prediction).strip().lower()
        if category not in self.TOPICS:
            raise ValueError(f"The topic model returned an unsupported category: {category}")
        return self.TOPICS[category]

    def load_sentiment_analyzer(self):
        self._log("  Loading NLTK VADER sentiment analyzer...")
        try:
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
        except LookupError:
            self._log("  VADER lexicon missing. Downloading vader_lexicon...")
            downloaded = nltk.download("vader_lexicon", quiet=True)
            if not downloaded:
                raise RuntimeError(
                    "Could not download the NLTK VADER lexicon. Check your connection and try again."
                )
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self._log("  Sentiment analyzer ready.")

    def detect_sentiment(self, headline, body):
        text = f"{headline}\n\n{body}".strip()
        if not text:
            raise ValueError("Sentiment analysis requires a headline or article body.")
        if self.sentiment_analyzer is None:
            self.load_sentiment_analyzer()

        # Use original text so VADER can read negation, punctuation and capitals.
        scores = self.sentiment_analyzer.polarity_scores(text)
        compound_score = scores["compound"]

        if compound_score >= 0.05:
            sentiment = "positive"
        elif compound_score <= -0.05:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        return {"sentiment": sentiment, "sentiment_scores": scores}

    def detect_scandal(self, text, entities):
        if not entities:
            return 1.0, []
        result = self.detect_environmental_disaster("", text, entities)
        return result["scandal_distance"], result["scandal_entities"]

    def detect_environmental_disaster(self, headline, body, entities=None):
        if entities is None:
            entities = self.detect_entities(headline, body)

        result = {
            "environmental_risk_score": 0.0,
            "scandal_distance": 1.0,
            "scandal_entities": [],
            "environmental_evidence": [],
            "environmental_model": self.ENVIRONMENTAL_MODEL,
            "top_10": False,
        }
        if not entities:
            self._log("  ORG-containing sentences to score: 0")
            return result

        # Match full organization names rather than substrings.
        matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        matcher.add("ORG", [self.nlp.make_doc(name) for name in entities])
        sentences = []
        sentencizer = self.nlp.get_pipe("sentencizer")
        for field, text in (("headline", headline), ("body", body)):
            document = sentencizer(self.nlp.make_doc(text))
            for sentence in document.sents:
                sentence_doc = sentence.as_doc()
                organizations = list(dict.fromkeys(
                    sentence_doc[start:end].text
                    for _, start, end in matcher(sentence_doc)
                ))
                if not organizations:
                    continue

                sentences.append({
                    "field": field,
                    "sentence": sentence.text,
                    "start_char": sentence.start_char,
                    "end_char": sentence.end_char,
                    "organizations": organizations,
                })

        self._log(f"  ORG-containing sentences to score: {len(sentences)}")
        if not sentences:
            return result

        # Load the encoder and embed the disaster phrases once.
        if self.environmental_model is None:
            self._log(f"  Loading sentence embedding model: {self.ENVIRONMENTAL_MODEL}...")
            import torch
            from sentence_transformers import SentenceTransformer

            torch.set_num_threads(min(4, torch.get_num_threads()))
            self.environmental_model = SentenceTransformer(self.ENVIRONMENTAL_MODEL, device="cpu")
            self._log(f"  Embedding {len(self.DISASTER_PHRASES)} disaster phrases...")
            self.disaster_embeddings = self.environmental_model.encode(
                list(self.DISASTER_PHRASES), normalize_embeddings=True, show_progress_bar=False,
            )
            self._log("  Environmental model ready.")

        chunks = []
        source_sentences = []
        tokenizer = self.environmental_model.tokenizer
        window_size = self.environmental_model.max_seq_length - tokenizer.num_special_tokens_to_add(pair=False)

        # Window long sentences so evidence near the end is not truncated.
        for sentence in sentences:
            text = sentence["sentence"]
            tokens = tokenizer(
                text, add_special_tokens=False, return_offsets_mapping=True,
                truncation=False, verbose=False,
            )
            offsets = tokens["offset_mapping"]
            if len(offsets) <= window_size:
                chunks.append(text)
                source_sentences.append(sentence)
                continue

            for start in range(0, len(offsets), max(1, window_size - 32)):
                end = min(start + window_size, len(offsets))
                chunks.append(text[offsets[start][0]:offsets[end - 1][1]])
                source_sentences.append(sentence)
                if end == len(offsets):
                    break

        embeddings = self.environmental_model.encode(
            chunks, normalize_embeddings=True, show_progress_bar=False,
        )
        similarities = cosine_similarity(embeddings, self.disaster_embeddings)

        best_matches = {}
        for index, sentence in enumerate(source_sentences):
            phrase_index = int(similarities[index].argmax())
            phrase = self.DISASTER_PHRASES[phrase_index]
            score = min(1.0, max(-1.0, float(similarities[index, phrase_index])))
            distance = 1.0 - score
            key = (sentence["field"], sentence["start_char"], sentence["end_char"])
            if key not in best_matches or score > best_matches[key]["similarity"]:
                best_matches[key] = {
                    **sentence,
                    "matched_text": chunks[index],
                    "matched_phrase": phrase,
                    "matched_category": self.PHRASE_CATEGORIES[phrase],
                    "similarity": score,
                    "distance": distance,
                }

        evidence = sorted(best_matches.values(), key=lambda match: match["distance"])
        # Each original sentence contributes once, even when split into windows.
        average_distance = float(sum(match["distance"] for match in evidence) / len(evidence))
        scandal_entities = []
        seen_entities = set()
        for match in evidence:
            if match["distance"] < 0.4:
                for name in match["organizations"]:
                    key = name.casefold()
                    if key not in seen_entities:
                        seen_entities.add(key)
                        scandal_entities.append(name)

        result["environmental_evidence"] = evidence
        result["scandal_distance"] = average_distance
        result["environmental_risk_score"] = max(0.0, 1.0 - average_distance)
        result["scandal_entities"] = scandal_entities
        return result

    def process_article(self, article, include_topic=True, include_environment=True):
        headline = article["headline"]
        body = article["body"]
        if not isinstance(headline, str) or not isinstance(body, str):
            raise TypeError("Article headline and body must be strings.")

        self._log("Preprocessing text...")
        cleaned_headline = self.preprocess_text(headline)
        cleaned_body = self.preprocess_text(body)
        cleaned_text = f"{cleaned_headline} {cleaned_body}".strip()
        self._log(f"  Retained tokens: {len(cleaned_text.split())}")

        # Enrich a copy; the original article stays unchanged.
        result = article.copy()
        result.update({
            "cleaned_headline": cleaned_headline,
            "cleaned_body": cleaned_body,
            "cleaned_text": cleaned_text,
        })
        self._log("Detecting organizations...")
        entities = self.detect_entities(headline, body)
        result["org"] = entities
        names = ", ".join(entities) if entities else "None"
        self._log(f"  Organizations: {names}")

        if include_topic:
            self._log("Detecting topic...")
            result["topic"] = self.detect_topic(headline, body)
            self._log(f"  Topic: {result['topic']}")

        self._log("Analyzing sentiment...")
        sentiment = self.detect_sentiment(headline, body)
        result.update(sentiment)
        self._log(f"  Sentiment: {sentiment['sentiment']}")
        self._log(f"  Compound score: {sentiment['sentiment_scores']['compound']:.4f}")
        if include_environment:
            self._log("Detecting environmental incidents...")
            result.update(self.detect_environmental_disaster(headline, body, entities))
            self._log(f"  Scandal score: {result['environmental_risk_score']:.4f}")
            self._log(f"  Scandal distance: {result['scandal_distance']:.4f}")
            names = ", ".join(result["scandal_entities"]) if result["scandal_entities"] else "None"
            self._log(f"  Strong-match organizations (distance < 0.4): {names}")
        return result

    def process_all_articles(self, input_dir=None, output_file=None, verbose=True):
        self.verbose = verbose
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        root = Path(__file__).resolve().parent
        input_dir = Path(input_dir) if input_dir is not None else root / "data"
        output_file = Path(output_file) if output_file is not None else root / "results" / "enhancend_news.csv"
        input_dir = input_dir.resolve()
        output_file = output_file.resolve()

        if not input_dir.is_dir():
            raise ValueError(f"Input folder does not exist: {input_dir}")
        protected_csvs = {root / "bbc_news_train.csv", root / "bbc_news_tests.csv"}
        if output_file.is_relative_to(input_dir) or output_file in protected_csvs:
            raise ValueError("Save results outside the source data and training CSVs.")
        if output_file.suffix.lower() != ".csv":
            raise ValueError("The output file must have a .csv extension.")

        paths = []
        for path in sorted(input_dir.glob("*/*.json")):
            try:
                date.fromisoformat(path.parent.name)
            except ValueError:
                continue
            paths.append(path)

        self._log(f"\nReading articles from: {input_dir}")
        self._log(f"Found {len(paths)} article files.")
        articles = []
        skipped = 0
        for index, path in enumerate(paths, start=1):
            self._log(f"\n[{index}/{len(paths)}] Loading: {path.name}")
            try:
                article = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(article, dict) or not all(
                    isinstance(article.get(field), str) and article[field].strip()
                    for field in ("id", "url", "date_scraped", "headline", "body")
                ):
                    raise ValueError("Missing id, url, date_scraped, headline or body")
                article["date_scraped"] = date.fromisoformat(article["date_scraped"][:10])
            except (OSError, ValueError) as error:
                skipped += 1
                self._log(f"[{index}/{len(paths)}] Skipping {path}: {error}")
                continue

            self._log(f"[{index}/{len(paths)}] Processing: {article['headline']}")
            self._log(f"ID: {article['id']}")
            articles.append(self.process_article(article))
            self._log(f"[{index}/{len(paths)}] Completed.")

        if not articles:
            raise ValueError("No valid articles found in dated folders.")

        # Rank after every article has a score.
        self._log("\nRanking environmental review candidates...")
        self._log("Ranking rule: select the 10 lowest-distance articles with ORG evidence.")
        articles.sort(key=lambda article: article["scandal_distance"])
        candidate_rank = 0
        for article in articles:
            article["environmental_rank"] = None
            article["top_10"] = False
            if article["environmental_evidence"]:
                candidate_rank += 1
                article["environmental_rank"] = candidate_rank
                article["top_10"] = candidate_rank <= 10
        self._log(f"Scored candidates: {candidate_rank}")
        self._log(f"Top-10 articles flagged: {min(candidate_rank, 10)}")

        self._log("\nTop 10 flagged articles (review candidates):")
        flagged_articles = [article for article in articles if article["top_10"]]
        if not flagged_articles:
            self._log("  None.")
        for rank, article in enumerate(flagged_articles, start=1):
            self._log(f"{rank}. {article['headline']}")
            self._log(f"   ID: {article['id']} | Scandal distance: {article['scandal_distance']:.4f}")

        # Keep lists in the DataFrame; encode them as JSON only for the CSV.
        rows = []
        for article in articles:
            rows.append({
                "id": article["id"],
                "url": article["url"],
                "date_scraped": article["date_scraped"],
                "headline": article["headline"],
                "body": article["body"],
                "org": article["org"],
                "topics": [article["topic"]],
                "sentiment": article["sentiment_scores"]["compound"],
                "scandal_distance": article["scandal_distance"],
                "top_10": article["top_10"],
            })

        results = pd.DataFrame(rows)
        csv_results = results.copy()
        for column in ("org", "topics"):
            csv_results[column] = csv_results[column].map(lambda names: json.dumps(names, ensure_ascii=False))
        output_file.parent.mkdir(parents=True, exist_ok=True)
        self._log(f"\nSaving {output_file}...")
        temporary = output_file.with_suffix(".tmp")
        try:
            csv_results.to_csv(temporary, index=False, encoding="utf-8")
            temporary.replace(output_file)
        finally:
            temporary.unlink(missing_ok=True)

        self._log(f"Saved: {len(results)} articles")
        self._log(f"Skipped: {skipped} articles")
        self._log("Finished.")
        return results


def main():
    parser = argparse.ArgumentParser(description="Enrich all stored articles and save the results to CSV.")
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-file", type=Path)
    args = parser.parse_args()
    try:
        NewsNLP(verbose=True).process_all_articles(args.input_dir, args.output_file)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
