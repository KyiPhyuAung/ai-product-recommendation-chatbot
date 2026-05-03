import re
import joblib
import numpy as np
import pandas as pd


STOP_WORDS = {
    "i", "want", "need", "looking", "look", "find", "search", "show", "give",
    "me", "please", "a", "an", "the", "for", "with", "and", "or", "to", "of",
    "in", "on", "by", "from", "that", "this", "is", "are", "can", "you",
    "recommend", "best", "good", "nice", "cheap", "budget", "under", "below",
    "less", "than", "max", "maximum"
}

ACCESSORY_WORDS = (
    r"case|cover|cable|charger|charging|adapter|protector|holder|stand|mount|"
    r"skin|strap|band|replacement|parts|dock|sleeve|organizer|sticker|remote|"
    r"scanner|transmitter|diagnostic|receiver|converter|connector|label|printer|"
    r"toy|kids|children|costume"
)

PHONE_ACCESSORY_WORDS = (
    r"case|cover|cable|charger|adapter|protector|holder|stand|mount|skin|"
    r"organizer|scanner|diagnostic|toy|kids|children"
)

REAL_PHONE_SIGNALS = (
    r"unlocked|smartphone|mobile phone|cell phone|renewed|128gb|256gb|512gb|"
    r"64gb|pro max|iphone|galaxy|pixel|oneplus|nothing phone"
)


class ProductRecommender:
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        self.df = None
        self.vectorizer = None
        self.tfidf_matrix = None
        self.nn_model = None

    def load(self):
        self.df = pd.read_csv(f"{self.model_dir}/products_index.csv")

        for col in ["title", "category_name", "productURL", "imgUrl"]:
            if col not in self.df.columns:
                self.df[col] = ""
            self.df[col] = self.df[col].fillna("").astype(str)

        for col in ["stars", "reviews", "price", "boughtInLastMonth"]:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce").fillna(0)

        self.df["isBestSeller"] = (
            self.df["isBestSeller"]
            .astype(str)
            .str.lower()
            .isin(["true", "1", "yes"])
        )

        self.df["search_text"] = (
            self.df["title"] + " " + self.df["category_name"]
        ).str.lower()

        self.vectorizer = joblib.load(f"{self.model_dir}/tfidf_vectorizer.joblib")
        self.tfidf_matrix = joblib.load(f"{self.model_dir}/tfidf_matrix.joblib")
        self.nn_model = joblib.load(f"{self.model_dir}/nearest_neighbors.joblib")

    def parse_price_limit(self, query):
        q = query.lower()
        patterns = [
            r"(under|below|less than|max|maximum)\s*\$?\s*(\d+)",
            r"\$?\s*(\d+)\s*(or less|and below|max|maximum)",
            r"under\s*(\d+)\s*\$",
        ]

        for pattern in patterns:
            match = re.search(pattern, q)
            if match:
                for group in match.groups():
                    if str(group).isdigit():
                        return float(group)

        return None

    def clean_query(self, query):
        q = query.lower()
        q = re.sub(r"\$?\d+\$?", " ", q)
        q = re.sub(r"[^\w\s]", " ", q)
        q = re.sub(r"\s+", " ", q).strip()
        return q

    def get_keywords(self, query):
        q = self.clean_query(query)
        words = re.findall(r"[a-zA-Z0-9]+", q)
        return [w for w in words if w not in STOP_WORDS and len(w) > 1]

    def detect_mode(self, query, keywords):
        q = query.lower()

        if "iphone case" in q or ("iphone" in keywords and "case" in keywords):
            return "iphone_case"

        if "phone case" in q or ("phone" in keywords and "case" in keywords):
            return "phone_case"

        if "nothing phone" in q:
            return "real_phone"

        if "iphone" in keywords and "case" not in keywords:
            return "real_phone"

        if "phone" in keywords and "case" not in keywords:
            return "real_phone"

        if any(k in keywords for k in ["headphone", "headphones", "headset", "headsets", "earbuds", "earbud", "earphones", "earphone", "airpods"]):
            return "audio"

        if any(k in keywords for k in ["laptop", "macbook", "notebook", "chromebook"]):
            return "laptop"

        if "keyboard" in keywords:
            return "keyboard"

        if "mouse" in keywords:
            return "mouse"

        if any(k in keywords for k in ["monitor", "display", "screen"]):
            return "monitor"

        if any(k in keywords for k in ["bag", "handbag", "backpack", "purse", "tote", "luggage", "suitcase"]):
            return "bag"

        if any(k in keywords for k in ["shoe", "shoes", "sneakers", "boots", "sandals"]):
            return "shoes"

        return "general"

    def contains_all_keywords(self, text, keywords):
        text = str(text).lower()
        return all(k in text for k in keywords)

    def keyword_score(self, text, keywords):
        text = str(text).lower()
        if not keywords:
            return 0
        return sum(1 for k in keywords if k in text) / len(keywords)

    def hard_filter(self, df, mode, keywords):
        title = df["title"].str.lower()
        text = df["search_text"].str.lower()

        if mode == "iphone_case":
            return df[
                title.str.contains("iphone", na=False)
                & title.str.contains("case|cover", na=False)
                & ~title.str.contains("charger|cable|adapter|headphone|earbud|scanner|organizer", na=False)
            ]

        if mode == "phone_case":
            return df[
                title.str.contains("phone", na=False)
                & title.str.contains("case|cover", na=False)
                & ~title.str.contains("charger|cable|adapter|headphone|earbud|scanner|organizer", na=False)
            ]

        if mode == "real_phone":
            filtered = df[
                title.str.contains(REAL_PHONE_SIGNALS, na=False)
                & ~title.str.contains(PHONE_ACCESSORY_WORDS, na=False)
                & (df["price"] >= 80)
            ]

            important = [k for k in keywords if k not in ["phone", "smartphone", "mobile", "cell"]]
            for token in important:
                token_filtered = filtered[filtered["title"].str.lower().str.contains(re.escape(token), na=False)]
                if len(token_filtered) > 0:
                    filtered = token_filtered

            return filtered

        if mode == "audio":
            filtered = df[
                title.str.contains("headphone|headphones|headset|headsets|earbud|earbuds|earphone|earphones|airpods|speaker", na=False)
            ]

            for token in keywords:
                if token not in ["headphone", "headphones", "headset", "headsets", "earbud", "earbuds"]:
                    token_filtered = filtered[filtered["title"].str.lower().str.contains(re.escape(token), na=False)]
                    if len(token_filtered) >= 3:
                        filtered = token_filtered

            return filtered

        if mode == "laptop":
            return df[
                title.str.contains("laptop|macbook|notebook|chromebook", na=False)
                & ~title.str.contains("case|cover|charger|adapter|stand|sleeve|bag|skin", na=False)
                & (df["price"] >= 150)
            ]

        mode_patterns = {
            "keyboard": "keyboard",
            "mouse": "mouse",
            "monitor": "monitor|display|screen",
            "bag": "bag|handbag|backpack|purse|tote|luggage|suitcase",
            "shoes": "shoe|shoes|sneakers|boots|sandals",
        }

        if mode in mode_patterns:
            return df[title.str.contains(mode_patterns[mode], na=False)]

        if keywords:
            strict = df[df["search_text"].apply(lambda x: self.contains_all_keywords(x, keywords))]
            if len(strict) > 0:
                return strict

        return df

    def recommend(self, query, top_n=5):
        query = query.strip()
        cleaned_query = self.clean_query(query)
        keywords = self.get_keywords(query)
        mode = self.detect_mode(query, keywords)
        max_price = self.parse_price_limit(query)

        query_vector = self.vectorizer.transform([cleaned_query])

        distances, indices = self.nn_model.kneighbors(
            query_vector,
            n_neighbors=min(12000, len(self.df))
        )

        candidates = self.df.iloc[indices.flatten()].copy()
        candidates["similarity"] = 1 - distances.flatten()

        filtered = self.hard_filter(candidates, mode, keywords)

        if len(filtered) < top_n:
            global_filtered = self.hard_filter(self.df.copy(), mode, keywords)
            if len(global_filtered) > 0:
                global_filtered = global_filtered.copy()
                global_filtered["similarity"] = 0.0
                filtered = pd.concat([filtered, global_filtered]).drop_duplicates(subset=["asin"])

        if max_price is not None:
            price_filtered = filtered[filtered["price"] <= max_price]
            if len(price_filtered) > 0:
                filtered = price_filtered

        if len(filtered) == 0:
            return pd.DataFrame(), "No related product found. Try a different keyword.", keywords, mode

        filtered = filtered.copy()

        filtered["keyword_score"] = filtered["search_text"].apply(
            lambda x: self.keyword_score(x, keywords)
        )

        filtered["title_keyword_score"] = filtered["title"].apply(
            lambda x: self.keyword_score(x, keywords)
        )

        filtered["exact_phrase"] = filtered["search_text"].str.contains(
            re.escape(cleaned_query), na=False
        ).astype(int)

        filtered["ranking_score"] = (
            filtered["keyword_score"] * 0.35
            + filtered["title_keyword_score"] * 0.25
            + filtered["similarity"] * 0.20
            + filtered["exact_phrase"] * 0.08
            + (filtered["stars"] / 5) * 0.05
            + (np.log1p(filtered["reviews"]) / 15) * 0.04
            + (np.log1p(filtered["boughtInLastMonth"]) / 15) * 0.02
            + filtered["isBestSeller"].astype(int) * 0.01
        )

        filtered = filtered.sort_values(
            by=["ranking_score", "keyword_score", "title_keyword_score", "similarity", "reviews"],
            ascending=False
        )

        return filtered.head(top_n), "", keywords, mode