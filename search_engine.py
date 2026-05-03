import re
import numpy as np
import pandas as pd


STOP_WORDS = {
    "i", "want", "need", "looking", "for", "find", "show", "give", "me",
    "please", "a", "an", "the", "with", "and", "or", "to", "of", "in", "on",
    "best", "good", "cheap", "budget", "under", "below", "less", "than",
    "max", "maximum", "recommend"
}

PHONE_DEVICE_WORDS = {"iphone", "phone", "smartphone", "mobile"}

PHONE_ACCESSORY_WORDS = (
    "case|cover|charger|charging|cable|adapter|stand|holder|mount|organizer|"
    "scanner|diagnostic|tool|protector|screen|dock|station|label|printer|toy|kids"
)

DEVICE_SIGNALS = (
    "unlocked|renewed|smartphone|mobile phone|cell phone|128gb|256gb|512gb|64gb|"
    "iphone|galaxy|pixel|oneplus|nothing phone"
)


class ProductSearchEngine:
    def __init__(self, data_path="models/products_index.csv"):
        self.data_path = data_path
        self.df = None

    def load(self):
        df = pd.read_csv(self.data_path)

        required_cols = [
            "asin", "title", "imgUrl", "productURL", "stars", "reviews",
            "price", "isBestSeller", "boughtInLastMonth", "category_name"
        ]

        for col in required_cols:
            if col not in df.columns:
                df[col] = ""

        df["title"] = df["title"].fillna("").astype(str)
        df["category_name"] = df["category_name"].fillna("").astype(str)
        df["imgUrl"] = df["imgUrl"].fillna("").astype(str)
        df["productURL"] = df["productURL"].fillna("").astype(str)

        df["stars"] = pd.to_numeric(df["stars"], errors="coerce").fillna(0)
        df["reviews"] = pd.to_numeric(df["reviews"], errors="coerce").fillna(0)
        df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0)
        df["boughtInLastMonth"] = pd.to_numeric(df["boughtInLastMonth"], errors="coerce").fillna(0)

        df["isBestSeller"] = (
            df["isBestSeller"].astype(str).str.lower().isin(["true", "1", "yes"])
        )

        df["search_text"] = (
            df["title"] + " " + df["category_name"]
        ).str.lower()

        self.df = df

    def parse_price(self, query):
        q = query.lower()
        match = re.search(r"(under|below|less than|max|maximum)\s*\$?\s*(\d+)", q)
        if match:
            return float(match.group(2))
        return None

    def normalize_word(self, word):
        word = word.lower()
        if len(word) > 4 and word.endswith("ies"):
            return word[:-3] + "y"
        if len(word) > 3 and word.endswith("s"):
            return word[:-1]
        return word

    def extract_keywords(self, query):
        q = query.lower()
        q = re.sub(r"\$?\d+\$?", " ", q)
        q = re.sub(r"[^a-zA-Z0-9\s]", " ", q)

        words = re.findall(r"[a-zA-Z0-9]+", q)
        keywords = []

        for word in words:
            word = self.normalize_word(word)
            if word not in STOP_WORDS and len(word) > 1:
                keywords.append(word)

        return keywords

    def word_mask(self, series, word):
        pattern = rf"\b{re.escape(word)}s?\b"
        return series.str.contains(pattern, regex=True, na=False)

    def all_keywords_mask(self, df, keywords):
        mask = pd.Series(True, index=df.index)
        for word in keywords:
            mask &= self.word_mask(df["search_text"], word)
        return mask

    def search(self, query, top_n=5):
        keywords = self.extract_keywords(query)
        max_price = self.parse_price(query)

        if not keywords:
            return pd.DataFrame(), "Please type a clearer product keyword.", keywords

        df = self.df.copy()

        # price filter
        if max_price is not None:
            df = df[df["price"] <= max_price]

        # strict search: every important keyword must appear
        strict = df[self.all_keywords_mask(df, keywords)].copy()

        # special case: user searches "iphone" or "phone" alone
        if len(keywords) == 1 and keywords[0] in PHONE_DEVICE_WORDS:
            title = df["title"].str.lower()

            strict = df[
                title.str.contains(DEVICE_SIGNALS, regex=True, na=False)
                & ~title.str.contains(PHONE_ACCESSORY_WORDS, regex=True, na=False)
                & (df["price"] >= 80)
            ].copy()

            if keywords[0] == "iphone":
                strict = strict[title.loc[strict.index].str.contains("iphone", na=False)]

        # if strict result is empty, use softer matching
        if strict.empty:
            soft_mask = pd.Series(False, index=df.index)
            for word in keywords:
                soft_mask |= self.word_mask(df["search_text"], word)
            strict = df[soft_mask].copy()

        if strict.empty:
            return pd.DataFrame(), "No related products were found.", keywords

        text = strict["search_text"]
        title = strict["title"].str.lower()

        strict["keyword_match_count"] = 0
        strict["title_match_count"] = 0

        for word in keywords:
            strict["keyword_match_count"] += self.word_mask(text, word).astype(int)
            strict["title_match_count"] += self.word_mask(title, word).astype(int)

        strict["keyword_score"] = strict["keyword_match_count"] / len(keywords)
        strict["title_score"] = strict["title_match_count"] / len(keywords)

        cleaned_query = " ".join(keywords)
        strict["phrase_score"] = title.str.contains(re.escape(cleaned_query), na=False).astype(int)

        strict["ranking_score"] = (
            strict["keyword_score"] * 0.35
            + strict["title_score"] * 0.25
            + strict["phrase_score"] * 0.15
            + (strict["stars"] / 5) * 0.08
            + (np.log1p(strict["reviews"]) / 15) * 0.07
            + (np.log1p(strict["boughtInLastMonth"]) / 15) * 0.05
            + strict["isBestSeller"].astype(int) * 0.05
        )

        strict = strict.sort_values(
            by=["ranking_score", "keyword_score", "title_score", "reviews"],
            ascending=False
        )

        return strict.head(top_n), "", keywords