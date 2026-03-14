import streamlit as st
import joblib
import pandas as pd
ingredient_db = pd.read_csv("dataset/ingredient_db.csv")
st.set_page_config(page_title="Smart Review Trust Assistant", page_icon="🛒", layout="wide")

model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

positive_words = {"good", "great", "amazing", "excellent", "perfect", "love", "best", "happy", "satisfied"}
negative_words = {"bad", "worst", "fake", "terrible", "waste", "fraud", "scam", "poor", "disappointed"}

def sentiment(text):
    words = text.lower().split()
    pos = sum(1 for w in words if w in positive_words)
    neg = sum(1 for w in words if w in negative_words)

    if pos > neg:
        return "Positive"
    if neg > pos:
        return "Negative"
    return "Neutral"

st.title("🛒 Smart Review Trust Assistant")
st.caption("Check reviews with prediction, confidence, sentiment, and bulk CSV analysis.")

tabs = st.tabs(["Single Review Check", "Bulk CSV Analysis"])

with tabs[0]:
    review = st.text_area("Enter product review", height=180)

    if st.button("Check Review", use_container_width=True):
        if review.strip() == "":
            st.warning("Enter a review first")
        else:
            vec = vectorizer.transform([review])
            pred = model.predict(vec)[0]

            if hasattr(model, "predict_proba"):
                confidence = model.predict_proba(vec).max() * 100
            else:
                confidence = 80.0

            sent = sentiment(review)
            trust_score = confidence if pred == 1 else 100 - confidence

            c1, c2, c3 = st.columns(3)

            with c1:
                if pred == 1:
                    st.success("✅ Genuine Review")
                else:
                    st.error("❌ Fake Review")

            with c2:
                st.metric("Confidence", f"{confidence:.2f}%")

            with c3:
                st.metric("Sentiment", sent)

            st.metric("Trust Score", f"{trust_score:.2f}/100")

with tabs[1]:
    file = st.file_uploader("Upload CSV with column name 'review'", type=["csv"])

    if file:
        df = pd.read_csv(file)

        if "review" not in df.columns:
            st.error("CSV must contain column 'review'")
        else:
            reviews = df["review"].astype(str)
            vec = vectorizer.transform(reviews)
            preds = model.predict(vec)

            if hasattr(model, "predict_proba"):
                conf = model.predict_proba(vec).max(axis=1) * 100
            else:
                conf = [80.0] * len(reviews)

            df["prediction"] = ["Genuine" if p == 1 else "Fake" for p in preds]
            df["confidence"] = conf
            df["sentiment"] = [sentiment(r) for r in reviews]

            st.dataframe(df, use_container_width=True)
            fake_count = (df["prediction"] == "Fake").sum()
            total = len(df)
            fake_ratio = fake_count / total * 100 if total else 0

            if fake_ratio < 20:
                st.success("Most reviews look trustworthy")
            elif fake_ratio < 40:
                st.warning("Some reviews look suspicious")
            else:
                st.error("High number of fake reviews detected")
st.header("Ingredient Safety Check")

ingredient_text = st.text_area("Enter ingredients separated by commas")

if st.button("Check Ingredient"):

    ingredients = [i.strip().lower() for i in ingredient_text.split(",") if i.strip()]

    if not ingredients:
        st.warning("Please enter at least one ingredient")

    for ing in ingredients:

        result = ingredient_db[ingredient_db["ingredient"].str.lower() == ing]

        if result.empty:
            st.warning(f"{ing} not found in database")

        else:
            data = result.iloc[0]

            st.success(f"Ingredient: {data['ingredient']}")
      st.header("Ingredient Safety Check")

ingredient_text = st.text_area("Enter ingredients separated by commas")

if st.button("Check Ingredient"):

    ingredients = [i.strip().lower() for i in ingredient_text.split(",") if i.strip()]

    if not ingredients:
        st.warning("Please enter at least one ingredient")

    for ing in ingredients:

        result = ingredient_db[ingredient_db["ingredient"].str.lower() == ing]

        if result.empty:
            st.warning(f"{ing} not found in database")

        else:
            data = result.iloc[0]

            st.success(f"Ingredient: {data['ingredient']}")

            st.write("Category:", data["category"])
            st.write("Safety Level:", data["safety_level"])
            st.write("Good Side:", data["good_side"])

            st.write(
                "Bad Side:",
                data["bad_side"] if pd.notna(data["bad_side"]) else "None"
            )

            st.write(
                "Best For:",
                data["best_for"] if pd.notna(data["best_for"]) else "General use"
            )

            st.write(
                "Avoid For:",
                data["avoid_for"] if pd.notna(data["avoid_for"]) else "None"
            )

            st.write(
                "Notes:",
                data["notes"] if pd.notna(data["notes"]) else "None"
            )


       
    
