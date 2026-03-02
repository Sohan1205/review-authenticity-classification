# Review Authenticity Classification

## 📌 Project Overview
This project detects whether an e-commerce product review is Genuine or Fake using Machine Learning techniques.

The system uses Natural Language Processing (NLP) and supervised classification algorithms to analyze review text and classify authenticity.

---

## ⚙️ Technologies Used
- Python
- Scikit-learn
- Pandas
- TF-IDF Vectorization
- Naive Bayes
- Logistic Regression
- Linear SVM

---

## 📊 Dataset
The dataset contains labeled product reviews:

- 1 = Genuine Review
- 0 = Fake Review

The dataset is stored inside the `dataset/` folder.

---

## 🧠 Machine Learning Models
The following models were implemented and compared:

- Multinomial Naive Bayes
- Logistic Regression
- Linear Support Vector Machine (SVM)

---

## 📈 Evaluation Metrics
Model performance was evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## 🔍 Text Processing Steps
- Convert text to lowercase
- Remove stopwords
- Apply TF-IDF vectorization
- Use unigrams and bigrams (ngram_range=(1,2))

---

## ▶️ How to Run the Project

1. Install dependencies:
