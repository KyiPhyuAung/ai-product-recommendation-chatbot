import streamlit as st
import pandas as pd
from product_search import ProductSearch

st.set_page_config(
    page_title="AI Product Recommendation Chatbot",
    layout="wide"
)

st.title("🛒 AI Product Recommendation Chatbot")
st.write("Search naturally for products. The system uses AI + strict filtering to return accurate results.")

@st.cache_resource
def load_engine():
    search_engine = ProductSearch()
    search_engine.load()
    return search_engine

with st.spinner("Loading AI system..."):
    search_engine = load_engine()

# Sidebar Interactive Filters
st.sidebar.header("Product Filters")
min_price = st.sidebar.slider("Minimum Price ($)", 0, 5000, 0)
max_price = st.sidebar.slider("Maximum Price ($)", 0, 5000, 5000)
min_stars = st.sidebar.slider("Minimum Rating (Stars)", 0.0, 5.0, 0.0)

query = st.text_input(
    "What are you looking for?",
    placeholder="Example: gaming laptop, iphone, wireless headphones, mini speaker"
)

top_n = st.selectbox("Number of results", [5, 10, 15, 20], index=0)

if query:
    results, keywords, note = search_engine.search(query, top_n)

    # Filter based on user configuration
    if not results.empty:
        results = results[
            (results["price"] >= min_price) & 
            (results["price"] <= max_price) & 
            (results["stars"] >= min_stars)
        ]

    st.markdown(f"## Results for: **{query}**")

    if keywords:
        st.write(f"🔎 Detected keywords: {', '.join(keywords)}")

    if note:
        st.warning(note)

    if results.empty:
        st.error("❌ No suitable products found matching the criteria.")
    else:
        # Pagination - 10 items per page
        page_size = 10
        total_pages = (len(results) + page_size - 1) // page_size
        
        if total_pages > 1:
            page = st.selectbox("Page", list(range(1, total_pages + 1))) - 1
            sliced_results = results.iloc[page * page_size : (page + 1) * page_size]
        else:
            sliced_results = results

        for _, row in sliced_results.iterrows():
            with st.container():
                col1, col2 = st.columns([1, 3])

                with col1:
                    if row.get("imgUrl"):
                        st.image(row["imgUrl"], use_container_width=True)

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