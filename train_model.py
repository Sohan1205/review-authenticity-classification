import pandas as pd
import random
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ---------- Generate Synthetic Dataset ----------
genuine_templates = [
    "This product is amazing and works perfectly",
    "Very satisfied with the quality",
    "Worth every penny",
    "Delivery was fast and product is authentic",
    "Excellent build quality and great performance",
    "Highly recommend this seller",
    "Product matches description exactly",
    "Very happy with my purchase",
    "Five stars, will buy again",
    "Original product and great packaging"
]

fake_templates = [
    "Worst product ever, fake seller",
    "Completely scam, do not buy",
    "Terrible quality and not original",
    "Very disappointed, waste of money",
    "Fake item received",
    "Seller is fraud",
    "Not as described, very bad",
    "Cheap copy product",
    "Horrible experience",
    "This is totally fake"
]

data = []

for _ in range(1000):
    data.append((random.choice(genuine_templates), 1))
    data.append((random.choice(fake_templates), 0))

df = pd.DataFrame(data, columns=["review", "label"])
df.to_csv("dataset/reviews.csv", index=False)

print("Dataset created with", len(df), "rows")

# ---------- Train Model ----------
X = df["review"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ---------- Save Model ----------
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model and vectorizer saved successfully.")
