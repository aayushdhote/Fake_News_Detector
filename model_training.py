import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load data
fake = pd.read_csv("/content/Fake.csv")
true = pd.read_csv("/content/True.csv")

fake["label"] = 1
true["label"] = 0

data = pd.concat([fake, true], axis=0)
data = data.sample(frac=1).reset_index(drop=True)

X = data["text"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorization
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Logistic Regression that supports predict_proba()
model = LogisticRegression(max_iter=300, solver="lbfgs")
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)
print(f" Accuracy: {acc*100:.2f}%")

# Save artifacts
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print(" Model retrained and saved successfully!")
