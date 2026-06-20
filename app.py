from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import sys
import os
import io
import re
import pdfplumber

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from preprocess import clean_text
from translations import SPECIALTY_TRANSLATIONS

app = FastAPI(title="Doctor AI - Report & Symptom Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("models/specialty_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

RECOMMENDATION_TEMPLATES = {
    "en": "Based on the information provided, we recommend consulting a {specialty}.",
    "hi": "दी गई जानकारी के आधार पर, हम {specialty} से परामर्श करने की सलाह देते हैं।",
    "mr": "दिलेल्या माहितीनुसार, आम्ही {specialty} चा सल्ला घेण्याची शिफारस करतो."
}

OVERALL_MESSAGE = {
    "en": "This is an AI-generated suggestion. Please consult a doctor for an accurate diagnosis.",
    "hi": "यह AI द्वारा दिया गया सुझाव है। सटीक निदान के लिए कृपया डॉक्टर से सलाह लें।",
    "mr": "हे AI द्वारे दिलेले सूचना आहे. अचूक निदानासाठी कृपया डॉक्टरांचा सल्ला घ्या."
}

DOCTOR_EXPLANATION = {
    "en": ["The AI analyzed the provided text to identify likely symptoms or conditions.",
           "This suggestion does not replace professional medical advice."],
    "hi": ["AI ने दिए गए टेक्स्ट का विश्लेषण कर संभावित लक्षणों या स्थितियों की पहचान की है।",
           "यह सुझाव पेशेवर चिकित्सा सलाह का विकल्प नहीं है।"],
    "mr": ["AI ने दिलेल्या मजकुराचे विश्लेषण करून संभाव्य लक्षणे ओळखली आहेत.",
           "हे सूचना व्यावसायिक वैद्यकीय सल्ल्याला पर्याय नाही."]
}

ABNORMAL_PATTERNS = [
    (r"blood pressure\D{0,10}(\d{2,3})\s*/\s*(\d{2,3})", "Blood Pressure"),
    (r"blood sugar\D{0,10}(\d{2,3})", "Blood Sugar"),
    (r"cholesterol\D{0,10}(\d{2,3})", "Cholesterol"),
    (r"heart rate\D{0,10}(\d{2,3})", "Heart Rate"),
]


# ---------- Request schema for JSON-based symptom input ----------
class SymptomsRequest(BaseModel):
    symptoms: str


def extract_text_from_pdf_bytes(file_bytes):
    text = ""
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def find_abnormal_values(raw_text):
    found = []
    lower_text = raw_text.lower()
    for pattern, label in ABNORMAL_PATTERNS:
        match = re.search(pattern, lower_text)
        if match:
            found.append(f"{label}: {match.group(0)}")
    return found


def build_language_block(specialty, lang, abnormal_values):
    trans = SPECIALTY_TRANSLATIONS.get(specialty, {"en": specialty, "hi": specialty, "mr": specialty})
    return {
        "specialty": trans[lang],
        "recommendation": RECOMMENDATION_TEMPLATES[lang].format(specialty=trans[lang]),
        "overall_message": OVERALL_MESSAGE[lang],
        "abnormal_values": abnormal_values,
        "doctor_explanation": DOCTOR_EXPLANATION[lang],
    }


def run_prediction(raw_text: str):
    cleaned = clean_text(raw_text)
    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0]
    confidence = float(max(proba))
    abnormal_values = find_abnormal_values(raw_text)

    result = {
        "en": build_language_block(prediction, "en", abnormal_values),
        "hi": build_language_block(prediction, "hi", abnormal_values),
        "mr": build_language_block(prediction, "mr", abnormal_values),
    }

    return {
        "result": result,
        "confidence": round(confidence, 4),
        "extracted_text": raw_text,
    }


@app.get("/")
def home():
    return {"status": "Doctor AI - Report & Symptom Analysis API is running"}


# ✅ PDF report upload साठी (upload_documents.dart वापरतं)
# multipart/form-data, field name: "image" (PDF bytes यात पाठवले जातात)
@app.post("/predict-report")
async def predict_report(image: UploadFile = File(...)):
    file_bytes = await image.read()
    raw_text = extract_text_from_pdf_bytes(file_bytes)
    return run_prediction(raw_text)


# ✅ थेट symptoms text साठी (ai_symptoms_screen.dart वापरतं)
# application/json, body: {"symptoms": "..."}
@app.post("/predict")
async def predict_symptoms(payload: SymptomsRequest):
    return run_prediction(payload.symptoms)