# Product Review Sentiment Analysis

An end-to-end NLP project that classifies product reviews as **Positive** or **Negative** using TF-IDF features and a Linear SVM model, deployed as an interactive **Streamlit** web application.

---

## 📌 Overview

This project demonstrates a complete sentiment analysis workflow:

- Data ingestion from the **Amazon Fine Food Reviews** dataset (Kaggle).
- Text preprocessing using **NLTK** (tokenization, stopword removal, lemmatization).
- Feature extraction using **TF-IDF** (unigrams + bigrams).
- Model training with **Linear SVM** (and Logistic Regression benchmark).
- Model persistence (`sentiment_model.pkl`, `vectorizer.pkl`) with **joblib**.
- Web UI for single review and bulk CSV predictions using **Streamlit**.
- Ready for deployment on **Streamlit Community Cloud**.

---

## 🧠 Architecture

1. **Data Layer**
   - Raw reviews from Kaggle (`data/Reviews.csv`).
   - Columns `Text` and `Score` used for sentiment modeling.

2. **Preprocessing & Feature Layer**
   - Custom text cleaner:
     - lowercase
     - HTML & URL removal
     - digits & punctuation removal
     - stopword removal
     - lemmatization
   - TF-IDF vectorization (`max_features=50,000`, `ngram_range=(1,2)`).

3. **Model Layer**
   - Binary labels:
     - `Score` 4–5 → positive
     - `Score` 1–2 → negative
   - Neutral (`Score = 3`) reviews removed.
   - Models evaluated:
     - Logistic Regression
     - Linear SVM (chosen as final model).

4. **Deployment Layer**
   - `app.py` (Streamlit app) for:
     - Single review prediction
     - CSV upload (`review` column) for bulk prediction
   - Hosted locally or on **Streamlit Community Cloud**.

---

## 📂 Project Structure

```bash
product-review-sentiment-analysis/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── Reviews.csv
├── models/
│   ├── sentiment_model.pkl
│   └── vectorizer.pkl
└── notebooks/
    └── product_review_sentiment.ipynb
