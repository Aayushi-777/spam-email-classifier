import pandas as pd
import re
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = "data/spam.csv"
MODEL_DIR = "saved_model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
VEC_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")

# Text Cleaning Function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Train Model
def train_model():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH, encoding="latin-1")
    df = df[["v1", "v2"]]
    df.columns = ["label", "message"]
    df["label"] = df["label"].str.strip().str.lower()
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})
    df["clean_message"] = df["message"].apply(clean_text)

    print("Splitting dataset...")
    X_train, X_test, Y_train, Y_test = train_test_split(
        df["clean_message"],
        df["label_num"],
        test_size=0.2,
        random_state=42,
        stratify=df["label_num"]
    )
    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("Training Model...")
    model = MultinomialNB()
    model.fit(X_train_vec, Y_train)

    print("Evaluating model...")
    Y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(Y_test, Y_pred)
    print(f"\nModelAccuracy: {accuracy*100:.2f}%\n")
    print("Classification Report:\n")
    print(classification_report(Y_test, Y_pred))

    # Save Model
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VEC_PATH)
    print("\nModel and vectorizer saved successfully!")

if __name__ == "__main__":
    train_model()