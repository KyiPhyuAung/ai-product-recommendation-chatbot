import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


STOPWORDS = {
    "i", "want", "need", "looking", "look", "find", "show", "give", "me",
    "please", "a", "an", "the", "for", "with", "and", "or", "to", "of",
    "in", "on", "by", "from", "best", "good", "cheap", "budget",
    "recommend", "recommended", "recommendation", "buy", "purchase", "get",
    "under", "below", "less", "than", "max", "maximum", "device", "product"
}


class ProductSearch:
    def __init__(self, data_path="models/products_index.csv"):
        self.data_path = data_path
        self.df = None
        self.vectorizer = None
        self.tfidf_matrix = None

    def load(self):
        df = pd.read_csv(self.data_path)

        for col in ["title", "category_name", "productURL", "imgUrl"]:
            if col not in df.columns:
                df[col] = ""
            df[col] = df[col].fillna("").astype(str)

        for col in ["stars", "reviews", "price", "boughtInLastMonth"]:
            if col not in df.columns:
                df[col] = 0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        if "isBestSeller" not in df.columns:
            df["isBestSeller"] = False

        df["isBestSeller"] = df["isBestSeller"].astype(str).str.lower().isin(
            ["true", "1", "yes"]
        )

        df["title_clean"] = df["title"].str.lower()
        df["category_clean"] = df["category_name"].str.lower()
        df["search_text"] = df["title_clean"] + " " + df["category_clean"]

        self.df = df.reset_index(drop=True)

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            lowercase=True,
            ngram_range=(1, 2),
            max_features=150000,
            min_df=2
        )

        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["search_text"])

    def clean_query(self, query):
        q = query.lower()
        q = re.sub(r"[^a-z0-9\s]", " ", q)
        q = re.sub(r"\s+", " ", q).strip()
        return q

    def extract_keywords(self, query):
        q = self.clean_query(query)
        words = re.findall(r"[a-z0-9]+", q)
        return [w for w in words if w not in STOPWORDS and len(w) > 1]

    def parse_price_limit(self, query):
        q = query.lower()
        match = re.search(r"(under|below|less than|max|maximum)\s*\$?\s*(\d+)", q)
        return float(match.group(2)) if match else None

    def word_contains(self, series, word):
        return series.str.contains(rf"\b{re.escape(word)}s?\b", regex=True, na=False)

    def get_main_product_word(self, keywords):
        product_words = [
            "laptop", "iphone", "phone", "smartphone", "speaker", "headphone",
            "headphones", "headset", "earbuds", "monitor", "keyboard", "mouse",
            "camera", "shoes", "shoe", "bag", "watch", "tablet", "charger",
            "case", "stand", "razor", "toy", "toys"
        ]

        for word in keywords:
            if word in product_words:
                return word

        return keywords[-1] if keywords else ""

    def apply_product_rules(self, results, main_word):
        title = results["title_clean"]

        rules = {
            "laptop": {
                "must": r"\blaptop\b|\bnotebook\b|\bchromebook\b|\bmacbook\b",
                "bad": r"backpack|bag|case|sleeve|cooling pad|cooler|stand|charger|adapter|mouse|keyboard|skin|cover"
            },
            "iphone": {
                "must": r"^apple iphone|^iphone|\bapple iphone\b",
                "bad": r"case|cover|charger|charging|cable|cord|adapter|protector|screen protector|glass|film|stand|holder|mount|dock|storage|photo stick|flash drive|usb|memory|for iphone|compatible with iphone|iphone app|microphone|mic|camera|projector|toy|game"
            },
            "phone": {
                "must": r"\bphone\b|\bsmartphone\b|\biphone\b|\bgalaxy\b|\bpixel\b|\boneplus\b|\bmotorola\b|\bnothing phone\b",
                "bad": r"case|cover|charger|charging|cable|cord|adapter|protector|stand|holder|mount|dock|watch|camera|projector|toy"
            },
            "speaker": {
                "must": r"\bspeaker\b|\bspeakers\b",
                "bad": r"stand|holder|mount|projector|monitor|case|cover"
            },
            "headphone": {
                "must": r"headphone|headphones|headset|earbud|earbuds|earphone|earphones",
                "bad": r"case|cover|stand|holder|replacement"
            },
            "headphones": {
                "must": r"headphone|headphones|headset|earbud|earbuds|earphone|earphones",
                "bad": r"case|cover|stand|holder|replacement"
            },
            "monitor": {
                "must": r"\bmonitor\b|\bdisplay\b|\bscreen\b",
                "bad": r"stand|mount|cable|adapter|case|cover|protector"
            },
            "shoes": {
                "must": r"\bshoe\b|\bshoes\b|\bsneaker\b|\bsneakers\b|\bboots\b",
                "bad": r"cleaner|rack|organizer|lace|insert|insole"
            },
            "shoe": {
                "must": r"\bshoe\b|\bshoes\b|\bsneaker\b|\bsneakers\b|\bboots\b",
                "bad": r"cleaner|rack|organizer|lace|insert|insole"
            }
        }

        if main_word not in rules:
            return results

        rule = rules[main_word]

        filtered = results[
            title.str.contains(rule["must"], regex=True, na=False)
            & ~title.str.contains(rule["bad"], regex=True, na=False)
        ].copy()

        return filtered 

    def search(self, query, top_n=5):
        keywords = self.extract_keywords(query)
        max_price = self.parse_price_limit(query)

        if not keywords:
            return pd.DataFrame(), keywords, "Please enter a clearer product request."

        cleaned_query = " ".join(keywords)
        query_vector = self.vectorizer.transform([cleaned_query])
        scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

        results = self.df.copy()
        results["tfidf_score"] = scores
        results = results[results["tfidf_score"] > 0].copy()

        if results.empty:
            return pd.DataFrame(), keywords, "No related product was found."

        main_word = self.get_main_product_word(keywords)

        results["main_match"] = self.word_contains(
            results["title_clean"], main_word
        ).astype(int)

        # Apply product-specific rules first
        results = self.apply_product_rules(results, main_word)

        if results.empty:
            return (
                pd.DataFrame(),
                keywords,
                "No exact matching product was found. The system avoided unrelated results."
            )

        # Recalculate main_match after filtering
        results["main_match"] = self.word_contains(
            results["title_clean"], main_word
        ).astype(int)

        # Strong filter: prefer titles that contain the main product word
        main_filtered = results[results["main_match"] == 1].copy()
        if len(main_filtered) >= top_n:
            results = main_filtered

        results["keyword_matches"] = 0
        results["title_matches"] = 0

        for word in keywords:
            results["keyword_matches"] += self.word_contains(
                results["search_text"], word
            ).astype(int)

            results["title_matches"] += self.word_contains(
                results["title_clean"], word
            ).astype(int)

        results["keyword_ratio"] = results["keyword_matches"] / len(keywords)
        results["title_ratio"] = results["title_matches"] / len(keywords)

        if max_price is not None:
            price_results = results[results["price"] <= max_price].copy()
            if len(price_results) >= top_n:
                results = price_results

        results["final_score"] = (
            results["main_match"] * 0.40
            + results["tfidf_score"] * 0.30
            + results["title_ratio"] * 0.15
            + results["keyword_ratio"] * 0.08
            + (results["stars"] / 5) * 0.04
            + (np.log1p(results["reviews"]) / 15) * 0.02
            + results["isBestSeller"].astype(int) * 0.01
        )

        results = results.sort_values(
            by=["final_score", "main_match", "tfidf_score", "reviews"],
            ascending=False
        )

        return results.head(top_n), keywords, ""