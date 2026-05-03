from ai_recommender import AIProductRecommender


TEST_QUERIES = {
    "mini speaker": ["speaker", "bluetooth"],
    "gaming laptop": ["laptop", "notebook", "chromebook", "macbook"],
    "iphone case": ["iphone", "case", "cover"],
    "phone stand": ["phone", "stand", "holder", "mount"],
    "adidas shoes": ["adidas", "shoe", "shoes", "sneaker"],
    "wireless headphones": ["wireless", "headphone", "headphones", "headset", "earbud"],
    "kitchen knife": ["kitchen", "knife"],
    "razor": ["razor", "shaver"],
}


def is_relevant(title, expected_words):
    title = title.lower()
    return any(word in title for word in expected_words)


def main():
    recommender = AIProductRecommender()
    recommender.load_data()

    total_precision = 0
    total_hit_rate = 0
    k = 5

    print("\nEvaluation Results")
    print("=" * 60)

    for query, expected_words in TEST_QUERIES.items():
        results, keywords, note = recommender.recommend(query, top_n=k)

        if results.empty:
            precision = 0
            hit = 0
        else:
            relevant_count = 0

            for _, row in results.iterrows():
                if is_relevant(row["title"], expected_words):
                    relevant_count += 1

            precision = relevant_count / k
            hit = 1 if relevant_count > 0 else 0

        total_precision += precision
        total_hit_rate += hit

        print(f"Query: {query}")
        print(f"Detected keywords: {keywords}")
        print(f"Precision@5: {precision:.2f}")
        print(f"Hit Rate@5: {hit}")
        print("-" * 60)

    print(f"Average Precision@5: {total_precision / len(TEST_QUERIES):.2f}")
    print(f"Average Hit Rate@5: {total_hit_rate / len(TEST_QUERIES):.2f}")


if __name__ == "__main__":
    main()