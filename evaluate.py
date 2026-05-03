from ai_recommender import AIProductRecommender


TEST_QUERIES = {
    "gaming laptop": "gaming_laptop",
    "iphone case": "iphone_case",
    "phone stand": "phone_stand",
    "mini speaker": "speaker",
    "portable mini speaker": "speaker",
    "wireless headphones": "headphones",
    "adidas shoes": "adidas_shoes",
    "shoe cleaner": "shoe_cleaner",
    "kitchen knife": "knife",
    "razor": "razor",
}


def is_relevant(query_type, title):
    title = title.lower()

    if query_type == "gaming_laptop":
        bad = [
            "monitor", "backpack", "bag", "cooling", "cooler", "stand",
            "charger", "adapter", "mouse", "keyboard", "webcam",
            "microphone", "capture card", "power jack", "dock", "hub",
            "case", "sleeve", "cover", "battery"
        ]
        return (
            ("laptop" in title or "notebook" in title or "chromebook" in title or "macbook" in title)
            and "gaming" in title
            and not any(word in title for word in bad)
        )

    if query_type == "iphone_case":
        return "iphone" in title and ("case" in title or "cover" in title)

    if query_type == "phone_stand":
        return (
            ("phone" in title or "iphone" in title or "smartphone" in title)
            and ("stand" in title or "holder" in title or "mount" in title)
        )

    if query_type == "speaker":
        bad = ["stand", "holder", "mount", "projector", "monitor", "case", "cover"]
        return ("speaker" in title or "speakers" in title) and not any(word in title for word in bad)

    if query_type == "headphones":
        return any(word in title for word in ["headphone", "headphones", "headset", "earbud", "earbuds"])

    if query_type == "adidas_shoes":
        return "adidas" in title and any(word in title for word in ["shoe", "shoes", "sneaker", "sneakers"])

    if query_type == "shoe_cleaner":
        return any(word in title for word in ["shoe cleaner", "sneaker cleaner", "cleaning kit", "shoe brush"])

    if query_type == "knife":
        return "knife" in title or "knives" in title

    if query_type == "razor":
        return "razor" in title or "shaver" in title

    return False


def main():
    recommender = AIProductRecommender()
    recommender.load_data()

    k = 5
    total_precision = 0
    total_hit_rate = 0

    print("\nEvaluation Results")
    print("=" * 70)

    for query, query_type in TEST_QUERIES.items():
        results, keywords, note = recommender.recommend(query, top_n=k)

        if results.empty:
            relevant_count = 0
            precision = 0
            hit_rate = 0
        else:
            relevant_count = 0

            for _, row in results.iterrows():
                if is_relevant(query_type, row["title"]):
                    relevant_count += 1

            precision = relevant_count / k
            hit_rate = 1 if relevant_count > 0 else 0

        total_precision += precision
        total_hit_rate += hit_rate

        print(f"Query: {query}")
        print(f"Detected keywords: {keywords}")
        print(f"Relevant in top {k}: {relevant_count}/{k}")
        print(f"Precision@{k}: {precision:.2f}")
        print(f"Hit Rate@{k}: {hit_rate}")
        if note:
            print(f"Note: {note}")
        print("-" * 70)

    print(f"Average Precision@{k}: {total_precision / len(TEST_QUERIES):.2f}")
    print(f"Average Hit Rate@{k}: {total_hit_rate / len(TEST_QUERIES):.2f}")


if __name__ == "__main__":
    main()