import re
import difflib
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


STOPWORDS = {
    "i", "want", "need", "looking", "look", "find", "show", "give", "me",
    "please", "a", "an", "the", "for", "with", "and", "or", "to", "of",
    "in", "on", "by", "from", "best", "good", "cheap", "budget",
    "recommend", "recommended", "recommendation", "buy", "purchase", "get",
    "under", "below", "less", "than", "max", "maximum", "device", "product",
    "item", "items", "something", "thing"
}

BRAND_ALIASES = {
    "adidas": ["adidas"],
    "nike": ["nike"],
    "sony": ["sony"],
    "apple": ["apple"],
    "samsung": ["samsung", "galaxy"],
    "msi": ["msi"],
    "asus": ["asus", "rog"],
    "lenovo": ["lenovo", "legion"],
    "dell": ["dell", "alienware"],
    "hp": ["hp", "omen", "pavilion"],
    "acer": ["acer", "predator"],
    "logitech": ["logitech"],
    "razer": ["razer"],
    "gucci": ["gucci"],
}

PRODUCT_CATEGORIES: Dict[str, Dict[str, object]] = {
    "laptop": {
        "aliases": ["laptop", "notebook", "chromebook", "macbook"],
        "must": r"\b(laptop|notebook|chromebook|macbook)\b",
        "bad": r"backpack|bag|case|sleeve|cooling pad|cooler|stand|charger|adapter|mouse|keyboard|skin|cover|protector|screen protector|replacement|battery|power jack|cable|cord|monitor|portable monitor|touch monitor|webcam|microphone|capture card|dock|hub|external|accessory",
        "min_price": 250,
    },
    "iphone": {
        "aliases": ["iphone"],
        "must": r"(^|\b)(apple\s+iphone|iphone\s*(\d+|se|xr|xs|pro|max|plus))\b",
        "bad": r"case|cover|charger|charging|cable|cord|adapter|protector|screen protector|glass|film|stand|holder|mount|dock|storage|photo stick|flash drive|usb|memory|for iphone|compatible with iphone|works with iphone|iphone app|microphone|mic|camera|projector|toy|game|remote|keyboard|mouse|watch|smartwatch",
        "min_price": 90,
    },
    "android_phone": {
        "aliases": ["android", "android phone", "smartphone"],
        "must": r"\b(android|samsung|galaxy|pixel|oneplus|motorola|moto|xiaomi|redmi|oppo|vivo|realme|nokia|nothing phone|smartphone|cell phone|mobile phone)\b",
        "bad": r"iphone|apple|case|cover|charger|charging|cable|cord|adapter|protector|stand|holder|mount|dock|watch|smartwatch|camera|projector|toy|game|remote|keyboard|mouse",
        "min_price": 60,
    },
    "phone": {
        "aliases": ["phone", "phones", "smartphone", "mobile"],
        "must": r"\b(phone|smartphone|iphone|galaxy|pixel|oneplus|motorola|nothing phone|cell phone|mobile phone)\b",
        "bad": r"case|cover|charger|charging|cable|cord|adapter|protector|stand|holder|mount|dock|watch|smartwatch|camera|projector|toy|game|remote|keyboard|mouse",
        "min_price": 60,
    },
    "phone_case": {
        "aliases": ["case", "cover"],
        "must": r"\b(case|cover)\b",
        "bad": r"phone only|unlocked phone|smartphone only",
        "min_price": 0,
    },
    "phone_stand": {
        "aliases": ["stand", "holder", "mount"],
        "must": r"\b(stand|holder|mount)\b",
        "bad": r"laptop stand|monitor stand|microphone stand|speaker stand",
        "min_price": 0,
    },
    "speaker": {
        "aliases": ["speaker", "speakers"],
        "must": r"\b(speaker|speakers|bluetooth speaker|portable speaker)\b",
        "bad": r"stand|holder|mount|projector|monitor|case|cover|replacement|screen",
        "min_price": 5,
    },
    "headphones": {
        "aliases": ["headphone", "headphones", "headset", "headsets", "earbud", "earbuds", "earphone", "earphones"],
        "must": r"\b(headphone|headphones|headset|headsets|earbud|earbuds|earphone|earphones)\b",
        "bad": r"case|cover|stand|holder|replacement|protector|ear pads only|cable only",
        "min_price": 3,
    },
    "monitor": {
        "aliases": ["monitor", "display", "screen"],
        "must": r"\b(monitor|display|screen)\b",
        "bad": r"stand|mount|cable|adapter|case|cover|protector|screen protector",
        "min_price": 20,
    },
    "keyboard": {
        "aliases": ["keyboard", "keyboards"],
        "must": r"\b(keyboard|keyboards)\b",
        "bad": r"case|cover|protector|skin|stand|keycaps only|switches only",
        "min_price": 5,
    },
    "mouse": {
        "aliases": ["mouse", "mice"],
        "must": r"\b(mouse|mice)\b",
        "bad": r"mouse pad|mousepad|mat|case|cover|trap",
        "min_price": 3,
    },
    "camera": {
        "aliases": ["camera", "webcam"],
        "must": r"\b(camera|webcam)\b",
        "bad": r"case|cover|bag|strap|mount only|tripod only|charger|battery only",
        "min_price": 10,
    },
    "shoes": {
        "aliases": ["shoe", "shoes", "sneaker", "sneakers", "boots", "sandals"],
        "must": r"\b(shoe|shoes|sneaker|sneakers|boots|sandals)\b",
        "bad": r"cleaner|rack|organizer|lace|laces|insert|insole|brush",
        "min_price": 5,
    },
    "shoe_cleaner": {
        "aliases": ["cleaner", "cleaning", "brush"],
        "must": r"\b(shoe cleaner|sneaker cleaner|cleaning kit|shoe brush|sneaker brush)\b",
        "bad": r"\bshoes\b|\bsneakers\b",
        "min_price": 0,
    },
    "bag": {
        "aliases": ["bag", "backpack", "handbag", "purse", "tote"],
        "must": r"\b(bag|backpack|handbag|purse|tote)\b",
        "bad": r"organizer insert|strap replacement|charm",
        "min_price": 5,
    },
    "razor": {
        "aliases": ["razor", "shaver"],
        "must": r"\b(razor|shaver)\b",
        "bad": r"blade refill|case|cover|charger only|replacement head",
        "min_price": 3,
    },
    "knife": {
        "aliases": ["knife", "knives"],
        "must": r"\b(knife|knives)\b",
        "bad": r"sharpener only|holder only|block only|case",
        "min_price": 3,
    },
}


def safe_regex_contains(series: pd.Series, pattern: str) -> pd.Series:
    return series.str.contains(pattern, regex=True, na=False, case=False)


class ProductSearch:
    def __init__(self, data_path: str = "models/products_index.csv"):
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

        df["isBestSeller"] = (
            df["isBestSeller"].astype(str).str.lower().isin(["true", "1", "yes"])
        )

        df["title_clean"] = df["title"].apply(self.normalize_text)
        df["category_clean"] = df["category_name"].apply(self.normalize_text)
        df["search_text"] = df["title_clean"] + " " + df["category_clean"]

        self.df = df.reset_index(drop=True)

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            lowercase=True,
            ngram_range=(1, 3),
            max_features=180000,
            min_df=2,
            sublinear_tf=True
        )

        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["search_text"])

    def normalize_text(self, text: str) -> str:
        text = str(text).lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def extract_keywords(self, query: str) -> List[str]:
        q = self.normalize_text(query)
        words = re.findall(r"[a-z0-9]+", q)
        return [w for w in words if w not in STOPWORDS and len(w) > 1]

    def parse_price_limit(self, query: str):
        q = query.lower()
        match = re.search(r"(under|below|less than|max|maximum)\s*\$?\s*(\d+)", q)
        return float(match.group(2)) if match else None

    def word_contains(self, series: pd.Series, word: str) -> pd.Series:
        if not word:
            return pd.Series(False, index=series.index)
        return series.str.contains(rf"\b{re.escape(word)}s?\b", regex=True, na=False)

    def detect_brand(self, keywords: List[str]) -> str:
        for brand, aliases in BRAND_ALIASES.items():
            for alias in aliases:
                if alias in keywords:
                    return brand
        return ""

    def detect_product_type(self, keywords: List[str]) -> str:
        query_text = " ".join(keywords)

        if "iphone" in keywords and any(w in keywords for w in ["case", "cover"]):
            return "phone_case"

        if any(w in keywords for w in ["phone", "iphone", "android", "smartphone"]) and any(
            w in keywords for w in ["stand", "holder", "mount"]
        ):
            return "phone_stand"

        if any(w in keywords for w in ["shoe", "shoes", "sneaker", "sneakers"]) and any(
            w in keywords for w in ["cleaner", "cleaning", "brush"]
        ):
            return "shoe_cleaner"

        if "iphone" in keywords:
            return "iphone"

        if "android" in keywords:
            return "android_phone"

        for product_type, rule in PRODUCT_CATEGORIES.items():
            for alias in rule["aliases"]:
                if alias in keywords or alias in query_text:
                    return product_type

        return keywords[-1] if keywords else ""

    def did_you_mean(self, keywords: List[str]) -> List[str]:
        suggestions = []
        common_terms = set()
        for rule in PRODUCT_CATEGORIES.values():
            common_terms.update(rule["aliases"])
        for aliases in BRAND_ALIASES.values():
            common_terms.update(aliases)

        for word in keywords:
            if word in common_terms:
                continue
            match = difflib.get_close_matches(word, common_terms, n=1, cutoff=0.82)
            if match:
                suggestions.append(f"{word} → {match[0]}")

        return suggestions

    def apply_product_rules(self, results: pd.DataFrame, product_type: str, keywords: List[str]) -> pd.DataFrame:
        if product_type not in PRODUCT_CATEGORIES:
            return results

        rule = PRODUCT_CATEGORIES[product_type]
        title = results["title_clean"]

        filtered = results[
            safe_regex_contains(title, rule["must"])
            & ~safe_regex_contains(title, rule["bad"])
            & (results["price"] >= float(rule.get("min_price", 0)))
        ].copy()

        if product_type == "phone_case":
            for kw in keywords:
                if kw in ["iphone", "samsung", "google", "pixel"]:
                    matched = safe_regex_contains(title, kw)
                    if matched.any():
                        filtered = filtered[matched]

        return filtered

    def add_match_features(self, results: pd.DataFrame, keywords: List[str], product_type: str, brand: str):
        results = results.copy()
        title = results["title_clean"]
        text = results["search_text"]

        results["keyword_matches"] = 0
        results["title_matches"] = 0

        for word in keywords:
            results["keyword_matches"] += self.word_contains(text, word).astype(int)
            results["title_matches"] += self.word_contains(title, word).astype(int)

        results["keyword_ratio"] = results["keyword_matches"] / max(len(keywords), 1)
        results["title_ratio"] = results["title_matches"] / max(len(keywords), 1)

        if product_type in PRODUCT_CATEGORIES:
            must_pattern = PRODUCT_CATEGORIES[product_type]["must"]
            results["product_type_match"] = safe_regex_contains(title, must_pattern).astype(int)
        else:
            main_word = keywords[-1] if keywords else ""
            results["product_type_match"] = self.word_contains(title, main_word).astype(int)

        if brand:
            brand_aliases = BRAND_ALIASES.get(brand, [brand])
            brand_pattern = r"|".join([rf"\b{re.escape(b)}\b" for b in brand_aliases])
            results["brand_match"] = safe_regex_contains(title, brand_pattern).astype(int)
        else:
            results["brand_match"] = 0

        results["accessory_penalty"] = 0

        if product_type in ["laptop", "iphone", "android_phone", "phone"]:
            accessory_words = (
                r"case|cover|charger|charging|cable|cord|adapter|protector|"
                r"stand|holder|mount|dock|replacement|screen protector|skin|sleeve|bag|"
                r"monitor|webcam|microphone|capture card|power jack|cooling|cooler|hub"
            )
            results["accessory_penalty"] = safe_regex_contains(title, accessory_words).astype(int)

        return results

    def apply_extra_strict_rules(self, results: pd.DataFrame, keywords: List[str], product_type: str) -> pd.DataFrame:
        title = results["title_clean"]

        if product_type == "laptop" and "gaming" in keywords:
            strict = results[
                title.str.contains(r"\bgaming\b", regex=True, na=False)
                & title.str.contains(r"\b(laptop|notebook|chromebook|macbook)\b", regex=True, na=False)
                & ~title.str.contains(
                    r"monitor|backpack|bag|cooling|cooler|stand|charger|adapter|mouse|keyboard|webcam|microphone|capture card|power jack|dock|hub|accessory|sleeve|case|cover|battery",
                    regex=True,
                    na=False
                )
                & (results["price"] >= 300)
            ].copy()

            if not strict.empty:
                return strict

        return results

    def rerank(self, results: pd.DataFrame) -> pd.DataFrame:
        results = results.copy()

        popularity_score = np.log1p(results["reviews"]) / 12
        bought_score = np.log1p(results["boughtInLastMonth"]) / 12
        rating_score = results["stars"] / 3

        results["final_score"] = (
            results["product_type_match"] * 0.30
            + results["tfidf_score"] * 0.20
            + rating_score * 0.15
            + popularity_score * 0.12
            + bought_score * 0.10
            + results["title_ratio"] * 0.05
            + results["keyword_ratio"] * 0.03
            + results["brand_match"] * 0.04
            + results["isBestSeller"].astype(int) * 0.01
            - results["accessory_penalty"] * 0.45
        )

        return results.sort_values(
            by=[
                "final_score",
                "product_type_match",
                "brand_match",
                "tfidf_score",
                "reviews"
            ],
            ascending=False
        )

    def search(self, query: str, top_n: int = 5) -> Tuple[pd.DataFrame, List[str], str]:
        keywords = self.extract_keywords(query)
        max_price = self.parse_price_limit(query)

        if not keywords:
            return pd.DataFrame(), keywords, "Please enter a clearer product request."

        suggestions = self.did_you_mean(keywords)
        product_type = self.detect_product_type(keywords)
        brand = self.detect_brand(keywords)

        cleaned_query = " ".join(keywords)
        query_vector = self.vectorizer.transform([cleaned_query])
        scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

        results = self.df.copy()
        results["tfidf_score"] = scores
        results = results[results["tfidf_score"] > 0].copy()

        if results.empty:
            note = "No related product was found."
            if suggestions:
                note += " Did you mean: " + ", ".join(suggestions)
            return pd.DataFrame(), keywords, note

        strict_results = self.apply_product_rules(results, product_type, keywords)

        if strict_results.empty and product_type in PRODUCT_CATEGORIES:
            note = "No exact matching product was found. The system avoided unrelated results."
            if suggestions:
                note += " Did you mean: " + ", ".join(suggestions)
            return pd.DataFrame(), keywords, note

        if not strict_results.empty:
            results = strict_results

        results = self.apply_extra_strict_rules(results, keywords, product_type)

        if max_price is not None:
            price_results = results[results["price"] <= max_price].copy()
            if not price_results.empty:
                results = price_results

        results = self.add_match_features(results, keywords, product_type, brand)
        results = self.rerank(results)

        note = ""
        if suggestions:
            note = "Did you mean: " + ", ".join(suggestions)

        return results.head(top_n), keywords, note