import streamlit as st
import pickle
import numpy as np

# Setting Page
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

# App title
st.title("📰 Fake News Detector")
st.write("This app predicts whether a news article is **Fake or Real** using a machine learning model trained on a Fake/True news dataset.")

# Load model + vectorizer (cached for speed)
@st.cache_resource
def load_artifacts():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_artifacts()

# Text input
st.markdown("### ✍️ Paste a News Article or Headline Below:")
news_text = st.text_area("", height=200, placeholder="Paste news article here...")

# Predict button
if st.button("🔍 Check News Authenticity"):
    if news_text.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        input_vector = vectorizer.transform([news_text])
        prediction = model.predict(input_vector)[0]
        prob = model.predict_proba(input_vector)[0]
        confidence = round(np.max(prob) * 100, 2)

        if prediction == 1:
            st.error(f"🚨 Fake News Detected — Confidence: {confidence}%")
        else:
            st.success(f"✅ Real News Detected — Confidence: {confidence}%")

st.markdown("---")
st.markdown("<p style='text-align:center;color:gray;'>Made with ❤️ by Aayush</p>", unsafe_allow_html=True)

