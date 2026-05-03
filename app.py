import streamlit as st
from ai_recommender import AIProductRecommender

st.set_page_config(
    page_title="AI Product Recommendation Chatbot",
    layout="wide"
)

st.title("🛒 AI Product Recommendation Chatbot")
st.write("Search naturally for products. The system uses AI + strict filtering to return accurate results.")

# Load model
@st.cache_resource
def load_ai():
    recommender = AIProductRecommender()
    recommender.load_data()
    return recommender

with st.spinner("Loading AI system..."):
    recommender = load_ai()

# Input
query = st.text_input(
    "What are you looking for?",
    placeholder="Example: gaming laptop, iphone, wireless headphones, mini speaker"
)

top_n = st.selectbox("Number of results", [5, 10, 15, 20], index=0)

# Search
if query:
    results, keywords, note = recommender.recommend(query, top_n)

    st.markdown(f"## Results for: **{query}**")

    # Keywords
    if keywords:
        st.write(f"🔎 Detected keywords: {', '.join(keywords)}")

    # Did you mean
    if note:
        st.warning(note)

    # No results
    if results.empty:
        st.error("❌ No suitable products found.")
    else:
        for _, row in results.iterrows():
            with st.container():
                col1, col2 = st.columns([1, 3])

                # Image
                with col1:
                    if row.get("imgUrl"):
                        st.image(row["imgUrl"], use_container_width=True)

                # Info
                with col2:
                    title = row.get("title", "No Title")
                    url = row.get("productURL", "#")

                    st.markdown(f"### [{title}]({url})")

                    st.write(
                        f"⭐ {row.get('stars', 0)} | "
                        f"🗨️ {int(row.get('reviews', 0))} reviews | "
                        f"💰 ${row.get('price', 0)}"
                    )

                    st.write(f"🔥 Bought last month: {int(row.get('boughtInLastMonth', 0))}")

                    if row.get("isBestSeller", False):
                        st.success("Best Seller")

                    st.info(
                        f"Score: {row.get('final_score', 0):.2f} | "
                        f"Type match: {row.get('product_type_match', 0)} | "
                        f"TF-IDF: {row.get('tfidf_score', 0):.2f}"
                    )

                st.markdown("---")