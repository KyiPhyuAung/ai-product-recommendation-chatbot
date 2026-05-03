# 🛒 AI Product Recommendation Chatbot

This project is an AI-powered product recommendation chatbot built using Python and scikit-learn.  
It allows users to search for products using natural language queries and returns relevant results from an Amazon-style dataset.

The system combines **TF-IDF vectorisation**, **cosine similarity**, and **rule-based filtering** to provide accurate and meaningful recommendations.

---

## 🚀 Features

- 🔍 Natural language product search (e.g., "gaming laptop", "iphone case")
- 🧠 TF-IDF based text processing (scikit-learn)
- 🎯 Cosine similarity for ranking relevance
- 🧹 Strict filtering to remove unrelated products
- ⭐ Ranking based on:
  - TF-IDF similarity
  - Keyword matching
  - Product type detection
  - Brand matching
  - Ratings and popularity
- 💡 "Did you mean" suggestions
- 📊 Evaluation using Precision@5 and Hit Rate@5
- 🖥️ Interactive UI with Streamlit

---

## 🧠 System Pipeline

```

User Query
→ Text Preprocessing
→ TF-IDF Vectorisation
→ Cosine Similarity
→ Product Filtering
→ Ranking
→ Top-N Results

```

---

## 🛠️ Technologies

- Python
- pandas
- numpy
- scikit-learn
- Streamlit

---

## 📂 Project Structure

```

product_recommendation_chatbot/
│
├── app.py
├── ai_recommender.py
├── product_search.py
├── evaluate.py
├── requirements.txt
├── README.md
│
└── models/
└── products_index.csv   (not included)

```

---

## ▶️ How to Run

Install dependencies:

```

pip install -r requirements.txt

```

Run the app:

```

streamlit run app.py

```

Run evaluation:

```

python evaluate.py

```

---

## 📊 Evaluation Results

- Precision@5: 1.00  
- Hit Rate@5: 1.00  

Test queries include:
- gaming laptop  
- iphone case  
- phone stand  
- mini speaker  
- wireless headphones  
- adidas shoes  

---

## ⚠️ Notes

- The dataset is not included due to file size limits.
- Filtering rules are used to remove accessories and irrelevant items.

---

## 📌 Limitations

- TF-IDF does not fully capture semantic meaning
- Results depend on dataset quality
- Evaluation is based on predefined relevance rules

---

## 🔮 Future Improvements

- Add semantic models (e.g., BERT)
- Improve synonym handling
- Add personalization
- Use real-time product APIs

---

## 📄 Ko Kyi Phyu Aung

Developed as part of a Data Mining / Information Retrieval project.
```
