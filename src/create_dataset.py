import pandas as pd
import os

data = {
    "report_text": [
        # Cardiologist
        "patient complains of chest pain, high blood pressure, irregular heartbeat",
        "shortness of breath, chest tightness, palpitations during exertion",
        "history of hypertension, swelling in legs, fatigue, rapid heart rate",
        "chest pain radiating to left arm, sweating, nausea",
        "high cholesterol levels, occasional chest discomfort, family history of heart disease",

        # Dermatologist
        "skin rash, itching, red patches on arms and legs",
        "acne breakout on face, oily skin, blackheads",
        "dry scaly patches on skin, severe itching, psoriasis like symptoms",
        "hair fall, dandruff, scalp irritation",
        "mole changing color and size, skin growth concern",

        # Neurologist
        "frequent headaches, blurred vision, dizziness",
        "numbness in hands and feet, tingling sensation",
        "memory loss, confusion, difficulty concentrating",
        "seizures, loss of consciousness, muscle twitching",
        "severe migraine, sensitivity to light, nausea",

        # Orthopedist
        "joint pain, swelling in knees, difficulty walking",
        "lower back pain, stiffness, difficulty bending",
        "fracture in arm after fall, severe pain and swelling",
        "shoulder pain, limited range of motion, sports injury",
        "chronic neck pain, muscle stiffness, posture related",

        # Endocrinologist
        "high blood sugar levels, frequent urination, fatigue",
        "unexplained weight gain, thyroid swelling, tiredness",
        "excessive thirst, weight loss, blurred vision, diabetes symptoms",
        "irregular periods, hormonal imbalance, weight fluctuation",
        "low energy levels, cold intolerance, hair thinning, thyroid issue",

        # Gastroenterologist
        "abdominal pain, bloating, irregular bowel movements",
        "acid reflux, heartburn, difficulty swallowing",
        "nausea, vomiting, loss of appetite, stomach pain",
        "blood in stool, chronic constipation, abdominal cramps",
        "liver function abnormal, jaundice, fatigue",

        # General Physician
        "mild fever, body ache, common cold symptoms",
        "general weakness, fatigue, no specific symptoms",
        "routine checkup, no major complaints, slight cough",
        "seasonal flu, sore throat, mild fever",
        "general fatigue, low immunity, frequent minor illnesses",

        # Pulmonologist
        "persistent cough, difficulty breathing, wheezing",
        "chest congestion, shortness of breath, asthma symptoms",
        "chronic cough, history of smoking, breathlessness",
        "lung infection, fever, productive cough with phlegm",
        "sleep apnea symptoms, snoring, daytime fatigue",

        # ENT Specialist
        "ear pain, hearing difficulty, ringing in ears",
        "sore throat, difficulty swallowing, hoarseness",
        "nasal congestion, sinus pain, frequent sneezing",
        "recurrent ear infections, fluid discharge from ear",
        "loss of smell, nasal blockage, post nasal drip",
    ],
    "specialty": (
        ["Cardiologist"] * 5 +
        ["Dermatologist"] * 5 +
        ["Neurologist"] * 5 +
        ["Orthopedist"] * 5 +
        ["Endocrinologist"] * 5 +
        ["Gastroenterologist"] * 5 +
        ["General Physician"] * 5 +
        ["Pulmonologist"] * 5 +
        ["ENT Specialist"] * 5
    )
}

df = pd.DataFrame(data)

os.makedirs("data", exist_ok=True)
df.to_csv("data/reports_dataset.csv", index=False)

print("Dataset created successfully!")
print(df.shape)
print(df['specialty'].value_counts())