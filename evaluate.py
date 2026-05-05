from product_search import ProductSearch
import time

# Expanded test queries covering additional categories & attributes
TEST_QUERIES = {
    "gaming laptop": "laptop",
    "iphone case": "phone_case",
    "phone stand": "phone_stand",
    "mini speaker": "speaker",
    "portable mini speaker": "speaker",
    "wireless headphones": "headphones",
    "adidas shoes": "shoes",
    "shoe cleaner": "shoe_cleaner",
    "kitchen knife": "knife",
    "electric razor": "razor",
    "leather luggage": "bag"
}


def is_relevant(query_type, title):
    title = title.lower()

    if query_type == "laptop":
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

    if query_type == "phone_case":
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

    if query_type == "shoes":
        return any(word in title for word in ["shoe", "shoes", "sneaker", "sneakers"])

    if query_type == "shoe_cleaner":
        return any(word in title for word in ["shoe cleaner", "sneaker cleaner", "cleaning kit", "shoe brush"])

    if query_type == "knife":
        return "knife" in title or "knives" in title

    if query_type == "razor":
        return "razor" in title or "shaver" in title

    if query_type == "bag":
        return "bag" in title or "luggage" in title or "backpack" in title

    return False


def main():
    search_engine = ProductSearch()
    search_engine.load()

    k = 5
    total_precision = 0
    total_hit_rate = 0
    total_latency = 0

    print("\nSystem Performance Evaluation")
    print("=" * 70)

    for query, query_type in TEST_QUERIES.items():
        start_time = time.time()
        results, keywords, note = search_engine.search(query, top_n=k)
        latency = time.time() - start_time
        total_latency += latency

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
        print(f"Latency: {latency:.4f}s")
        print(f"Relevant in top {k}: {relevant_count}/{k}")
        print(f"Precision@{k}: {precision:.2f}")
        print(f"Hit Rate@{k}: {hit_rate}")
        if note:
            print(f"Note: {note}")
        print("-" * 70)

    print(f"Average Precision@{k}: {total_precision / len(TEST_QUERIES):.2f}")
    print(f"Average Hit Rate@{k}: {total_hit_rate / len(TEST_QUERIES):.2f}")
    print(f"Average System Latency: {total_latency / len(TEST_QUERIES):.4f}s")


if __name__ == "__main__":
    main()