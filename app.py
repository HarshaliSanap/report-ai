import streamlit as st
import joblib
import os
import sys
import pdfplumber

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from preprocess import clean_text
from translations import SPECIALTY_TRANSLATIONS, MESSAGES

st.set_page_config(page_title="Doctor Recommendation AI", page_icon="🩺")

st.title("🩺 Medical Report → Doctor Recommendation")
st.write("Report upload करा, AI तुम्हाला योग्य specialist सुचवेल.")

@st.cache_resource
def load_model():
    model = joblib.load("models/specialty_model.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

uploaded_file = st.file_uploader("Report upload करा (PDF)", type=["pdf"])
manual_text = st.text_area("किंवा थेट symptoms/report text टाका:")

if st.button("Analyze करा"):
    if uploaded_file is not None:
        raw_text = extract_text_from_pdf(uploaded_file)
    elif manual_text.strip():
        raw_text = manual_text
    else:
        st.warning("कृपया PDF upload करा किंवा text टाका.")
        st.stop()

    cleaned = clean_text(raw_text)
    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0]
    confidence = max(proba)

    trans = SPECIALTY_TRANSLATIONS.get(
        prediction, {"en": prediction, "hi": prediction, "mr": prediction}
    )

    st.markdown(
        "### 🩺 "
        + MESSAGES["result_heading"]["en"] + " / "
        + MESSAGES["result_heading"]["hi"] + " / "
        + MESSAGES["result_heading"]["mr"]
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**English**")
        st.success(trans["en"])
    with col2:
        st.markdown("**हिंदी**")
        st.success(trans["hi"])
    with col3:
        st.markdown("**मराठी**")
        st.success(trans["mr"])

    st.write(
        f"{MESSAGES['confidence']['en']} / "
        f"{MESSAGES['confidence']['hi']} / "
        f"{MESSAGES['confidence']['mr']}: **{confidence:.2%}**"
    )

    with st.expander("Extracted Text बघा"):
        st.write(raw_text)

    st.info(
        f"⚠️ {MESSAGES['disclaimer']['en']}\n\n"
        f"{MESSAGES['disclaimer']['hi']}\n\n"
        f"{MESSAGES['disclaimer']['mr']}"
    )