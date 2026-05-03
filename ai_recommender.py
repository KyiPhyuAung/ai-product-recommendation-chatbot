from product_search import ProductSearch


class AIProductRecommender:
    def __init__(self):
        self.search_engine = ProductSearch()

    def load_data(self):
        self.search_engine.load()

    def recommend(self, query, top_n=5):
        results, keywords, note = self.search_engine.search(query, top_n)
        return results, keywords, note