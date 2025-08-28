import os
import io
import torch
import nltk
import speech_recognition as sr
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from transformers import pipeline, BertForSequenceClassification, BertTokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# ---------------- INIT ----------------



NLTK_DATA_PATH = os.path.join(os.path.dirname(__file__), "nltk_data")
os.makedirs(NLTK_DATA_PATH, exist_ok=True)

# Ensure VADER is available
try:
    nltk.data.find("sentiment/vader_lexicon")
except LookupError:
    nltk.download("vader_lexicon", download_dir=NLTK_DATA_PATH)

# Add path manually so nltk can find it
nltk.data.path.append(NLTK_DATA_PATH)


vader = SentimentIntensityAnalyzer()

# Emotion model
emotion_model = pipeline("sentiment-analysis", model="tabularisai/multilingual-sentiment-analysis")

# FinBERT Tone
finbert = BertForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone", num_labels=3)
finbert_tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
tone_labels = ["Neutral", "Positive", "Negative"]

# FastAPI
app = FastAPI(title="Sentiment • Emotion • Tone API", version="2.0.0")


# ---------------- HELPERS ----------------
def _label3(label_str: str) -> str:
    l = label_str.lower()
    if "pos" in l:
        return "Positive"
    if "neg" in l:
        return "Negative"
    return "Neutral"

def _signed_score(label: str, score01: float) -> float:
    if label == "Positive":
        return +abs(float(score01))
    if label == "Negative":
        return -abs(float(score01))
    return 0.0

def score_sentiment(text: str) -> float:
    c = vader.polarity_scores(text)["compound"]
    if c >= 0.05:
        return _signed_score("Positive", abs(c))
    elif c <= -0.05:
        return _signed_score("Negative", abs(c))
    else:
        return 0.0

def score_emotion(text: str) -> float:
    out = emotion_model(text)[0]
    lab = _label3(out["label"])
    return _signed_score(lab, float(out["score"]))

def score_tone(text: str) -> float:
    inputs = finbert_tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = finbert(**inputs).logits
        probs = torch.softmax(logits, dim=1).squeeze()
        idx = torch.argmax(probs).item()
        lab = tone_labels[idx]
        scr = float(probs[idx].item())
    return _signed_score(lab, scr)

def analyze_text_core(text: str):
    return [{
        "sentiment": round(score_sentiment(text), 4),
        "emotion":   round(score_emotion(text),   4),
        "tone":      round(score_tone(text),      4),
    }]


# ---------------- SCHEMAS ----------------
class TextIn(BaseModel):
    text: str


# ---------------- ROUTES ----------------
@app.get("/")
def root():
    return {"ok": True, "endpoints": ["/analyze-text", "/analyze-voice"]}

@app.post("/analyze-text")
def analyze_text(payload: TextIn):
    text = (payload.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    return analyze_text_core(text)

@app.post("/analyze-voice")
async def analyze_voice(file: UploadFile = File(...)):
    # Save uploaded audio temporarily
    fname = (file.filename or "audio").lower()
    if not any(fname.endswith(ext) for ext in (".wav", ".aiff", ".aif")):
        raise HTTPException(status_code=400, detail="Please upload WAV/AIFF file (MP3 not supported by speech_recognition without ffmpeg).")

    data = await file.read()
    tmp_path = f"/tmp/{fname}"
    with open(tmp_path, "wb") as f:
        f.write(data)

    # SpeechRecognition with Google Web Speech API (free, no key)
    recognizer = sr.Recognizer()
    with sr.AudioFile(tmp_path) as source:
        audio = recognizer.record(source)

    try:
        transcript = recognizer.recognize_google(audio, language="en-US")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")

    return analyze_text_core(transcript)
