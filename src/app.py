import streamlit as st
import joblib
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="TrustLens AI",
    page_icon="🛡️",
    layout="wide"
)

# ---------------- LOAD FILES ----------------
ingredient_db = pd.read_csv("dataset/ingredient_db.csv")
model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# ---------------- SIMPLE SENTIMENT ----------------
positive_words = {
    "good", "great", "amazing", "excellent", "perfect",
    "love", "best", "happy", "satisfied", "awesome", "authentic"
}
negative_words = {
    "bad", "worst", "fake", "terrible", "waste",
    "fraud", "scam", "poor", "disappointed", "awful"
}

def sentiment(text: str) -> str:
    words = text.lower().split()
    pos = sum(1 for w in words if w in positive_words)
    neg = sum(1 for w in words if w in negative_words)

    if pos > neg:
        return "Positive"
    elif neg > pos:
        return "Negative"
    return "Neutral"

def safety_color(level: str):
    level = str(level).strip().lower()
    if level == "safe":
        st.success(f"Safety Level: {level.title()}")
    elif level == "caution":
        st.warning(f"Safety Level: {level.title()}")
    else:
        st.error(f"Safety Level: {level.title()}")

def ingredient_score(level: str):
    level = str(level).strip().lower()
    if level == "safe":
        return 10
    elif level == "caution":
        return -5
    elif level == "risky":
        return -15
    return 0

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main-title {
    font-size: 40px;
    font-weight: 800;
    margin-bottom: 0;
}
.sub-title {
    font-size: 17px;
    color: #94a3b8;
    margin-top: 0;
    margin-bottom: 18px;
}
.feature-card {
    padding: 16px;
    border-radius: 14px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 10px;
}
.small-note {
    font-size: 13px;
    color: #94a3b8;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("🛡️ TrustLens AI")
st.sidebar.caption("Commercial AI Product Trust Platform")
st.sidebar.markdown("""
### Core Modules
- Home Dashboard
- Review Authenticity Check
- Bulk Review Analysis
- Ingredient Safety Check
- Camera Scanner

### Best Use Cases
- Fake review detection
- Product trust evaluation
- Ingredient risk analysis
- Consumer safety support
""")
st.sidebar.info("Tagline: See the truth behind every product.")

# ---------------- HEADER ----------------
st.markdown('<p class="main-title">🛡️ TrustLens AI</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">AI-powered review authenticity, ingredient safety, and product trust intelligence.</p>',
    unsafe_allow_html=True
)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown('<div class="feature-card"><b>Review Detection</b><br>Detect fake and genuine product reviews instantly.</div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="feature-card"><b>Ingredient Safety</b><br>Analyze ingredients and identify safety risks.</div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="feature-card"><b>Bulk Intelligence</b><br>Upload CSV reviews for large-scale analytics.</div>', unsafe_allow_html=True)

tabs = st.tabs([
    "Home",
    "Single Review Check",
    "Bulk Review Analysis",
    "Ingredient Safety",
    "Camera Scanner"
])

# ---------------- HOME / DASHBOARD ----------------
with tabs[0]:
    st.subheader("Commercial Product Dashboard")

    a, b, c, d = st.columns(4)
    with a:
        st.metric("AI Modules", "5")
    with b:
        st.metric("Review Detection", "Enabled")
    with c:
        st.metric("Ingredient Safety", "Enabled")
    with d:
        st.metric("Deployment", "Live")

    st.markdown("### Platform Overview")
    st.write(
        "TrustLens AI helps users and businesses detect fake reviews, analyze ingredient safety, "
        "evaluate product trust, and make safer buying decisions."
    )

    x, y = st.columns(2)

    with x:
        st.markdown("#### What users can do")
        st.write("- Analyze a single product review")
        st.write("- Upload CSV files for bulk review insights")
        st.write("- Check ingredients for safety and caution")
        st.write("- Use camera input to scan product labels")
        st.write("- See trust metrics and safety signals")

    with y:
        st.markdown("#### Why this matters")
        st.write("- Reduces fake review manipulation")
        st.write("- Improves consumer trust")
        st.write("- Helps identify risky ingredients")
        st.write("- Supports smarter product decisions")
        st.write("- Feels like a real commercial AI tool")

# ---------------- SINGLE REVIEW CHECK ----------------
with tabs[1]:
    st.subheader("Single Review Authenticity Check")

    review = st.text_area(
        "Enter product review",
        height=180,
        placeholder="Paste a customer review here..."
    )

    if st.button("Analyze Review", use_container_width=True):
        if review.strip() == "":
            st.warning("Enter a review first")
        else:
            vec = vectorizer.transform([review])
            pred = model.predict(vec)[0]

            if hasattr(model, "predict_proba"):
                confidence = float(model.predict_proba(vec).max() * 100)
            else:
                confidence = 80.0

            sent = sentiment(review)
            trust_score = confidence if pred == 1 else 100 - confidence

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                if pred == 1:
                    st.success("✅ Genuine")
                else:
                    st.error("❌ Fake")

            with col2:
                st.metric("Confidence", f"{confidence:.2f}%")

            with col3:
                st.metric("Sentiment", sent)

            with col4:
                st.metric("Trust Score", f"{trust_score:.2f}/100")

            st.progress(max(0, min(int(trust_score), 100)))

            if trust_score >= 75:
                st.success("This review appears highly trustworthy.")
            elif trust_score >= 45:
                st.warning("This review has mixed trust signals.")
            else:
                st.error("This review appears suspicious.")

# ---------------- BULK REVIEW ANALYSIS ----------------
with tabs[2]:
    st.subheader("Bulk Review Analysis")

    file = st.file_uploader(
        "Upload CSV with column name 'review'",
        type=["csv"]
    )

    if file is not None:
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

            fake_count = int((df["prediction"] == "Fake").sum())
            genuine_count = int((df["prediction"] == "Genuine").sum())
            total = len(df)
            fake_ratio = (fake_count / total * 100) if total else 0
            trust_score = 100 - fake_ratio

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Total Reviews", total)
            with m2:
                st.metric("Fake Reviews", fake_count)
            with m3:
                st.metric("Genuine Reviews", genuine_count)
            with m4:
                st.metric("Product Trust Score", f"{trust_score:.2f}/100")

            chart_df = pd.DataFrame(
                {
                    "Type": ["Genuine", "Fake"],
                    "Count": [genuine_count, fake_count]
                }
            ).set_index("Type")
            st.bar_chart(chart_df)

            sentiment_counts = df["sentiment"].value_counts()
            st.markdown("### Sentiment Distribution")
            st.bar_chart(sentiment_counts)

            if fake_ratio < 20:
                st.success("Most reviews look trustworthy.")
            elif fake_ratio < 40:
                st.warning("Some reviews look suspicious.")
            else:
                st.error("High number of fake reviews detected.")

# ---------------- INGREDIENT SAFETY ----------------
with tabs[3]:
    st.subheader("Ingredient Safety Check")

    ingredient_text = st.text_area(
        "Enter ingredients separated by commas",
        placeholder="Example: water, glycerin, citric acid"
    )

    if st.button("Check Ingredients", use_container_width=True):
        ingredients = [i.strip().lower() for i in ingredient_text.split(",") if i.strip()]

        if not ingredients:
            st.warning("Please enter at least one ingredient")
        else:
            total_score = 50

            for ing in ingredients:
                result = ingredient_db[ingredient_db["ingredient"].str.lower() == ing]

                if result.empty:
                    st.warning(f"{ing} not found in database")
                else:
                    data = result.iloc[0]

                    total_score += ingredient_score(data["safety_level"])

                    st.markdown(f"### {data['ingredient']}")
                    st.write("**Category:**", data["category"])
                    safety_color(data["safety_level"])
                    st.write("**Good Side:**", data["good_side"])
                    st.write("**Bad Side:**", data["bad_side"] if pd.notna(data["bad_side"]) else "None")
                    st.write("**Best For:**", data["best_for"] if pd.notna(data["best_for"]) else "General use")
                    st.write("**Avoid For:**", data["avoid_for"] if pd.notna(data["avoid_for"]) else "None")
                    st.write("**Notes:**", data["notes"] if pd.notna(data["notes"]) else "None")
                    st.divider()

            total_score = max(0, min(100, total_score))
            st.markdown("### Overall Ingredient Safety Score")
            st.metric("Safety Score", f"{total_score}/100")
            st.progress(total_score)

            if total_score >= 75:
                st.success("Overall ingredient profile looks safe.")
            elif total_score >= 45:
                st.warning("Overall ingredient profile needs caution.")
            else:
                st.error("Overall ingredient profile appears risky.")

# ---------------- CAMERA SCANNER ----------------
with tabs[4]:
    st.subheader("Camera Scanner")

    st.write("Capture a product label image and use it for manual ingredient reading.")
    st.caption("This is the first commercial-ready scanner step. OCR can be added later.")

    camera_image = st.camera_input("Scan product label")

    if camera_image is not None:
        st.image(camera_image, caption="Captured Product Label", use_container_width=True)
        st.success("Image captured successfully.")
        st.info("Next step: read the label text and paste the ingredients into the Ingredient Safety tab.")
