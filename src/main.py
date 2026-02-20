import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = {
    "review": [
        "This product is amazing and works perfectly",
        "Worst product ever, total waste of money",
        "I love this item, highly recommended",
        "Buy now limited offer click this link",
        "Terrible quality, not worth the price",
        "Best purchase I have made this year"
    ],
    "label": [1, 0, 1, 0, 0, 1]
}

df = pd.DataFrame(data)
X = df["review"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Report:\n", classification_report(y_test, y_pred))

while True:
    text = input("Enter a review (or 'exit'): ")
    if text.lower() == "exit":
        break
    pred = model.predict(vectorizer.transform([text]))[0]
    print("Prediction:", "Genuine" if pred == 1 else "Fake")
