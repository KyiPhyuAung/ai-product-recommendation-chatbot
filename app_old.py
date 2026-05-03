import streamlit as st
from recommender import ProductRecommender

st.set_page_config(
    page_title="AI Product Recommendation Chatbot",
    page_icon="🛒",
    layout="wide"
)

st.markdown("""
# 🛒 AI Product Recommendation Chatbot

Search naturally for online shopping products.  
Example: **iphone**, **iphone case**, **sony headphones**, **wireless headset**, **gucci bag**, **kitchen knife**
""")

@st.cache_resource
def load_recommender():
    recommender = ProductRecommender()
    recommender.load()
    return recommender

with st.spinner("Loading recommendation engine..."):
    recommender = load_recommender()

col1, col2 = st.columns([4, 1])

with col1:
    query = st.text_input(
        "What product are you looking for?",
        placeholder="Type product keyword here..."
    )

with col2:
    top_n = st.selectbox("Results", [5, 10, 15, 20], index=0)

st.divider()

if query:
    results, note, keywords, mode = recommender.recommend(query, top_n=top_n)

    st.subheader(f"Recommended products for: {query}")
    st.caption(f"Search mode: {mode} | Keywords: {', '.join(keywords)}")

    if note:
        st.warning(note)

    if results.empty:
        st.error("No suitable products found.")
    else:
        for _, row in results.iterrows():
            with st.container(border=True):
                col_img, col_text = st.columns([1, 4])

                with col_img:
                    img_url = row.get("imgUrl", "")
                    if isinstance(img_url, str) and img_url.startswith("http"):
                        st.image(img_url, width=150)

                with col_text:
                    title = row.get("title", "No title")
                    url = row.get("productURL", "#")

                    st.markdown(f"### [{title}]({url})")
                    st.write(
                        f"⭐ **Rating:** {row['stars']} | "
                        f"💬 **Reviews:** {int(row['reviews'])} | "
                        f"💵 **Price:** ${row['price']}"
                    )

                    category = row.get("category_name", "")
                    if category:
                        st.write(f"📦 **Category:** {category}")

                    st.write(f"🔥 **Bought last month:** {int(row['boughtInLastMonth'])}")

                    if bool(row["isBestSeller"]):
                        st.success("Best Seller")

                    st.info(
                        f"Why recommended: keyword match {row['keyword_score']:.2f}, "
                        f"title match {row['title_keyword_score']:.2f}, "
                        f"similarity {row['similarity']:.2f}."
                    )
else:
    st.info("Enter a product keyword to start.")