import streamlit as st
import joblib

st.title("Review Authenticity Checker")

model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

review = st.text_area("Enter product review:")

if st.button("Check Review"):
    if review:
        review_vec = vectorizer.transform([review])
        prediction = model.predict(review_vec)[0]

        if prediction == 1:
            st.success("Genuine Review")
        else:
            st.error("Fake Review")
    else:
        st.warning("Please enter a review")
