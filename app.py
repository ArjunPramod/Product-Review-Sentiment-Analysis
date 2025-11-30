import os
import joblib
import pandas as pd
import streamlit as st
import nltk
import re
import string

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure NLTK resources (you may comment these out after first run if they slow startup)
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab') 
nltk.download('wordnet')

# ------------------------
# Load model and vectorizer
# ------------------------

MODEL_PATH = os.path.join("models", "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join("models", "vectorizer.pkl")

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer

model, vectorizer = load_artifacts()

# ------------------------
# Text preprocessing
# ------------------------

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    tokens = [
        lemmatizer.lemmatize(tok)
        for tok in tokens
        if tok not in stop_words and len(tok) > 2
    ]
    return " ".join(tokens)

def predict_sentiment(texts):
    cleaned = [clean_text(t) for t in texts]
    X_vec = vectorizer.transform(cleaned)
    preds = model.predict(X_vec)
    return preds

# ------------------------
# Streamlit UI
# ------------------------

st.set_page_config(
    page_title="Product Review Sentiment Analysis",
    page_icon="⭐",
    layout="wide"
)

st.title("🛒 Product Review Sentiment Analysis")
st.write(
    "Classify product reviews as **Positive** or **Negative** using a TF-IDF + Linear SVM model."
)

# Tabs: Single Prediction / Bulk Prediction
tab1, tab2 = st.tabs(["🔍 Single Review", "📂 CSV Upload"])

with tab1:
    st.subheader("Single Review Prediction")
    user_review = st.text_area(
        "Enter a product review:",
        placeholder="Type or paste a review here..."
    )
    if st.button("Predict Sentiment", type="primary"):
        if not user_review.strip():
            st.warning("Please enter a review.")
        else:
            prediction = predict_sentiment([user_review])[0]
            if prediction == "positive":
                st.success("✅ Predicted Sentiment: **Positive**")
            else:
                st.error("❌ Predicted Sentiment: **Negative**")

with tab2:
    st.subheader("Bulk Prediction from CSV")

    st.write(
        """
        **Instructions:**  
        - Upload a CSV file with a column named **`review`** containing the review text.  
        """
    )

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            df = None

        if df is not None:
            if "review" not in df.columns:
                st.error("CSV must contain a column named `review`.")
            else:
                if st.button("Run Bulk Prediction"):
                    reviews = df["review"].astype(str).tolist()
                    preds = predict_sentiment(reviews)
                    df["predicted_sentiment"] = preds

                    st.write("### Preview of Results")
                    st.dataframe(df.head())

                    # Download link
                    csv_out = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv_out,
                        file_name="sentiment_predictions.csv",
                        mime="text/csv",
                    )
