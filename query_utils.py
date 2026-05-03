import re


STOP_WORDS = {
    "i", "want", "need", "looking", "look", "find", "search", "show", "give",
    "me", "please", "a", "an", "the", "for", "with", "and", "or", "to", "of",
    "in", "on", "by", "from", "that", "this", "is", "are", "can", "you",
    "recommend", "best", "good", "nice", "cheap", "budget"
}


def parse_price_limit(query):
    q = query.lower()

    patterns = [
        r"(under|below|less than|max|maximum|budget)\s*\$?\s*(\d+)",
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


def clean_query(query):
    q = query.lower()
    q = re.sub(r"\$?\d+\$?", " ", q)
    q = re.sub(r"[^\w\s]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q


def get_keywords(query):
    q = clean_query(query)
    words = re.findall(r"[a-zA-Z0-9]+", q)

    keywords = []
    for word in words:
        if word not in STOP_WORDS and len(word) > 1:
            keywords.append(word)

    return keywords


def keyword_coverage(text, keywords):
    text = str(text).lower()

    if not keywords:
        return 0

    matched = 0
    for keyword in keywords:
        if keyword in text:
            matched += 1

    return matched / len(keywords)


def phrase_match(text, query):
    text = str(text).lower()
    q = clean_query(query)

    if not q:
        return 0

    return int(q in text)