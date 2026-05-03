import streamlit as st
from ai_recommender import AIProductRecommender

st.set_page_config(
    page_title="AI Product Recommendation Chatbot",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 AI Product Recommendation Chatbot")
st.write(
    "Search naturally for online shopping products. "
    "The system uses strict product matching and ranking to avoid unrelated results."
)

@st.cache_resource
def load_ai():
    recommender = AIProductRecommender()
    recommender.load_data()
    recommender.load_model()
    recommender.build_or_load_embeddings()
    return recommender

with st.spinner("Loading product recommendation system..."):
    recommender = load_ai()

query = st.text_input(
    "What are you looking for?",
    placeholder="Example: iphone case, iphone 13, phone stand, adidas shoes, shoe cleaner"
)

top_n = st.selectbox("Number of results", [5, 10, 15, 20], index=0)

st.divider()

if query:
    results, keywords, note = recommender.recommend(query, top_n)

    st.subheader(f"Recommendations for: {query}")
    st.caption(f"Detected keywords: {', '.join(keywords)}")

    if note:
        st.warning(note)

    if not results.empty:
        for _, row in results.iterrows():
            with st.container(border=True):
                col_img, col_info = st.columns([1, 4])

                with col_img:
                    img_url = row.get("imgUrl", "")
                    if isinstance(img_url, str) and img_url.startswith("http"):
                        st.image(img_url, width=150)

                with col_info:
                    title = row.get("title", "No title")
                    url = row.get("productURL", "#")

                    st.markdown(f"### [{title}]({url})")
                    st.write(
                        f"⭐ Rating: {row.get('stars', 0)} | "
                        f"💬 Reviews: {int(row.get('reviews', 0))} | "
                        f"💵 Price: ${row.get('price', 0)}"
                    )

                    category = row.get("category_name", "")
                    if category:
                        st.write(f"📦 Category: {category}")

                    st.write(f"🔥 Bought last month: {int(row.get('boughtInLastMonth', 0))}")

                    if bool(row.get("isBestSeller", False)):
                        st.success("Best Seller")

                    st.info(
                        f"Why recommended: keyword match = {row.get('keyword_ratio', 0):.2f}, "
                        f"title match = {row.get('title_ratio', 0):.2f}, "
                        f"score = {row.get('final_score', 0):.2f}."
                    )
else:
    st.info("Type a product request above.")