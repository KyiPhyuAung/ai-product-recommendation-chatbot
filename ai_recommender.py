from product_search import ProductSearch


class AIProductRecommender:
    def __init__(self):
        self.search_engine = ProductSearch()

    def load_data(self):
        self.search_engine.load()

    def load_model(self):
        pass

    def build_or_load_embeddings(self):
        pass

    def recommend(self, query, top_n=5):
        return self.search_engine.search(query, top_n)