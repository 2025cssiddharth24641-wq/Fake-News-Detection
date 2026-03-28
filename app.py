import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from starlette.requests import Request
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

from utils.text_processing import clean_text, cleaned_word_count, is_meaningful_input

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=50000)


class PredictResponse(BaseModel):
    prediction: str
    confidence: float
    reliability: str
    status: str
    guidance: str


app = FastAPI(title="Fake News Detection", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


class ModelBundle:
    def __init__(self) -> None:
        self.model: Optional[tf.keras.Model] = None
        self.tokenizer = None
        self.max_len: int = 300
        self.threshold: float = 0.5
        self.uncertainty_threshold: float = 0.7
        self.ready: bool = False
        self.message: str = "Model not initialized"

    def load(self) -> None:
        model_path = MODEL_DIR / "fake_news_model.keras"
        tokenizer_path = MODEL_DIR / "tokenizer.json"
        metadata_path = MODEL_DIR / "metadata.json"

        if not model_path.exists() or not tokenizer_path.exists() or not metadata_path.exists():
            self.ready = False
            self.message = "Model artifacts missing. Train the model first with train_model.py"
            return

        try:
            self.model = tf.keras.models.load_model(model_path)
            with open(tokenizer_path, "r", encoding="utf-8") as f:
                self.tokenizer = tokenizer_from_json(f.read())
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            self.max_len = int(metadata.get("max_len", 300))
            self.threshold = float(metadata.get("threshold", 0.5))
            self.uncertainty_threshold = float(metadata.get("uncertainty_threshold", 0.7))
            self.ready = True
            self.message = "Model loaded"
        except Exception as exc:
            self.ready = False
            self.message = f"Failed to load model: {exc}"

    def infer_real_probability(self, text: str) -> tuple[float, str]:
        if not self.ready or self.model is None or self.tokenizer is None:
            raise RuntimeError(self.message)

        cleaned = clean_text(text)
        if len(cleaned.split()) < 6:
            raise ValueError("Input is too short after cleaning")

        seq = self.tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=self.max_len, padding="post", truncating="post")
        prob_real = float(self.model.predict(np.array(padded), verbose=0)[0][0])
        return prob_real, cleaned


_EXTREME_PATTERNS = [
    r"\bcure all diseases\b",
    r"\bcure\b.*\bdisease",
    r"\bbreaking shocking\b",
    r"\b100\s*%\s*guaranteed\b",
    r"\bmiracle cure\b",
    r"\bno side effects\b",
    r"\bsecret they don't want you to know\b",
    r"\bthis changes everything\b",
    r"\byou won't believe\b",
    r"\bshocking truth\b",
]

_FACTUAL_CUES = {
    "according",
    "official",
    "officials",
    "government",
    "ministry",
    "agency",
    "authorities",
    "reported",
    "report",
    "confirmed",
    "announced",
    "statement",
    "inaugurated",
    "election",
    "court",
    "policy",
    "regulator",
    "national",
    "international",
}


def rule_based_risk_score(raw_text: str, cleaned_text: str) -> float:
    text = raw_text.lower()
    score = 0.0

    for pattern in _EXTREME_PATTERNS:
        if re.search(pattern, text):
            score += 0.14

    exclamation_count = raw_text.count("!")
    if exclamation_count >= 3:
        score += min(0.18, exclamation_count * 0.02)

    all_caps_words = re.findall(r"\b[A-Z]{4,}\b", raw_text)
    if len(all_caps_words) >= 3:
        score += min(0.16, len(all_caps_words) * 0.03)

    cleaned_tokens = cleaned_text.split()
    unique_ratio = len(set(cleaned_tokens)) / max(len(cleaned_tokens), 1)
    if unique_ratio < 0.45:
        score += 0.08

    if len(cleaned_tokens) < 9:
        score += 0.12

    return float(np.clip(score, 0.0, 0.75))


def has_extreme_claim_language(raw_text: str) -> bool:
    lowered = raw_text.lower()
    return any(re.search(pattern, lowered) for pattern in _EXTREME_PATTERNS)


def factual_context_score(raw_text: str, cleaned_text: str) -> float:
    score = 0.0
    lower_raw = raw_text.lower()
    cleaned_tokens = cleaned_text.split()

    sentence_like_chunks = [part.strip() for part in re.split(r"[.!?]+", raw_text) if part.strip()]
    if len(sentence_like_chunks) >= 2:
        score += 0.2
    if len(sentence_like_chunks) >= 3:
        score += 0.1

    number_count = len(re.findall(r"\b\d{1,4}(?:,\d{3})*(?:\.\d+)?\b", raw_text))
    if number_count >= 2:
        score += 0.15

    uppercase_acronyms = len(re.findall(r"\b[A-Z]{2,5}\b", raw_text))
    if uppercase_acronyms >= 2:
        score += 0.12

    factual_hits = sum(1 for token in cleaned_tokens if token in _FACTUAL_CUES)
    if factual_hits >= 2:
        score += 0.18

    if len(cleaned_tokens) >= 25:
        score += 0.12

    if re.search(r"\b(on|in|at)\s+[A-Z][a-z]+\s+\d{1,2},\s+\d{4}\b", raw_text):
        score += 0.08

    return float(np.clip(score, 0.0, 1.0))


def reliability_from_confidence(confidence: float) -> str:
    if confidence >= 0.85:
        return "HIGH"
    if confidence >= 0.7:
        return "MEDIUM"
    return "LOW"


def combine_scores(prob_real_model: float, risk_score: float) -> float:
    adjusted_real_from_rules = 1.0 - risk_score
    combined_real = (0.82 * prob_real_model) + (0.18 * adjusted_real_from_rules)
    return float(np.clip(combined_real, 0.02, 0.98))


bundle = ModelBundle()


@app.on_event("startup")
def startup_event() -> None:
    bundle.load()


@app.get("/", response_class=HTMLResponse)
def read_home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
def health() -> dict:
    return {"status": "ok" if bundle.ready else "degraded", "model": bundle.message}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    text = payload.text.strip()

    if not text:
        raise HTTPException(status_code=400, detail="Input text cannot be empty")

    if not is_meaningful_input(text):
        raise HTTPException(status_code=400, detail="Input is too short or not meaningful. Enter a full news paragraph.")

    if cleaned_word_count(text) < 6:
        raise HTTPException(status_code=400, detail="Input is too short after preprocessing. Enter a longer news paragraph.")

    try:
        prob_real_model, cleaned = bundle.infer_real_probability(text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception:
        raise HTTPException(status_code=500, detail="Prediction failed due to an internal error")

    risk_score = rule_based_risk_score(text, cleaned)
    factual_score = factual_context_score(text, cleaned)
    extreme_language = has_extreme_claim_language(text)
    prob_real_combined = combine_scores(prob_real_model, risk_score)

    if prob_real_combined >= bundle.threshold:
        base_prediction = "REAL"
        base_confidence = prob_real_combined
    else:
        base_prediction = "FAKE"
        base_confidence = 1.0 - prob_real_combined

    confidence = float(np.clip(base_confidence - (risk_score * 0.25), 0.0, 0.95))

    unclear_margin = abs(prob_real_combined - 0.5)
    if unclear_margin < 0.08:
        prediction = "Needs Verification"
        confidence = min(confidence, 0.69)
        guidance = "Signals are mixed. Verify with trusted sources before accepting this claim."
    elif base_prediction == "FAKE" and prob_real_model < 0.2 and risk_score <= 0.16 and not extreme_language and (factual_score >= 0.28 or len(cleaned.split()) >= 7):
        prediction = "Needs Verification"
        confidence = min(confidence, 0.64)
        guidance = "This text appears factual, but the model strongly disagrees. Likely out-of-domain input: verify with multiple trusted sources."
    elif base_prediction == "FAKE" and extreme_language and confidence >= 0.65:
        prediction = "FAKE"
        guidance = "Likely fake. The text includes extreme-claim language. Cross-check with reliable outlets before sharing."
    elif confidence < bundle.uncertainty_threshold:
        prediction = "Uncertain"
        guidance = "Low confidence, verify with trusted sources."
    elif base_prediction == "FAKE":
        prediction = "FAKE"
        guidance = "Likely fake. Cross-check with reliable outlets before sharing."
    else:
        prediction = "REAL"
        guidance = "Likely real. Continue verifying with additional trusted sources."

    reliability = reliability_from_confidence(confidence)

    return PredictResponse(
        prediction=prediction,
        confidence=confidence,
        reliability=reliability,
        status="success",
        guidance=guidance,
    )
