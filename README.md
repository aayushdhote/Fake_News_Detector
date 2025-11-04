# Fake_News_Detector
# 📰 Fake News Detector using Machine Learning

A **Fake News Detection Web App** built using **Python, Scikit-learn, and Streamlit**.  
It classifies news articles as **Real** or **Fake** based on their textual content.

---

## 🚀 Live Demo
👉 [Click here to try the app](https://share.streamlit.io/)

---

## 🧠 Overview

This project uses **Natural Language Processing (NLP)** and a **Machine Learning model** trained on real-world news datasets to detect whether a piece of news is genuine or fake.

Users can paste any headline or article text into the app, and it will instantly predict the authenticity with a confidence score.

---

## ⚙️ How It Works

1. The dataset contains two files:
   - `Fake.csv`
   - `True.csv`

2. Data was preprocessed and combined into one dataframe.
3. Text features were extracted using **TF-IDF Vectorization**.
4. A **PassiveAggressiveClassifier** was trained to distinguish fake vs. real news.
5. The final trained model and vectorizer were saved as:
   - `model.pkl`
   - `vectorizer.pkl`
6. These are used in the **Streamlit web app** for real-time prediction.

---

## 🧰 Tech Stack

| Technology | Purpose |
|-------------|----------|
| **Python** | Programming language |
| **Scikit-learn** | Machine learning algorithms |
| **Pandas / NumPy** | Data processing |
| **TF-IDF Vectorizer** | Feature extraction |
| **Streamlit** | Web app interface |

---

## 🧾 Installation & Usage

### 1️⃣ Clone this repository
```bash
git clone https://github.com/<your-username>/fake-news-detector.git
cd fake-news-detector
