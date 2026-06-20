import joblib
import os
import sys
import pdfplumber

sys.path.append(os.path.dirname(__file__))
from preprocess import clean_text

def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def predict_specialty(input_text_or_path, is_pdf=False):
    if is_pdf:
        raw_text = extract_text_from_pdf(input_text_or_path)
    else:
        raw_text = input_text_or_path

    cleaned = clean_text(raw_text)

    model = joblib.load("models/specialty_model.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")

    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0]
    confidence = max(proba)

    return prediction, confidence

if __name__ == "__main__":
    
    sample_text = "patient has joint pain and swelling in knees, difficulty walking"
    specialty, confidence = predict_specialty(sample_text)
    print(f"Recommended Specialty: {specialty}")
    print(f"Confidence: {confidence:.2f}")