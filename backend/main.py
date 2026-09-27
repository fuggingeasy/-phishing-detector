"""
main.py
FastAPI backend for the Phishing Email Detector, now with user accounts.

Run:
    uvicorn main:app --reload --port 8000

Requires model.pkl and vectorizer.pkl in the same directory
(generate them first with: python train_model.py --generate 4000)

Endpoints:
    POST /auth/register   -> create account, returns access token
    POST /auth/login      -> log in, returns access token
    GET  /auth/me         -> current user info (requires token)
    POST /predict         -> analyze an email (requires token)
    GET  /                -> health check
"""

import os

import joblib
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session

import auth
import db_models
from database import Base, engine, get_db
from preprocessing import clean_text, extract_signals

Base.metadata.create_all(bind=engine)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")

app = FastAPI(
    title="Phishing Email Detection API",
    description="Context-Aware Phishing Email Detection Using Machine Learning and NLP",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None
_vectorizer = None


def _load_artifacts():
    global _model, _vectorizer
    if _model is None or _vectorizer is None:
        if not (os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH)):
            raise HTTPException(
                status_code=503,
                detail=(
                    "Model artifacts not found. Run `python train_model.py "
                    "--generate 4000` in the backend/ directory first."
                ),
            )
        _model = joblib.load(MODEL_PATH)
        _vectorizer = joblib.load(VECTORIZER_PATH)
    return _model, _vectorizer


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------
class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6, description="At least 6 characters")


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    email: str


class UserResponse(BaseModel):
    id: int
    email: str


class EmailRequest(BaseModel):
    email: str = Field(..., min_length=1, description="Raw email content to analyze")


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    risk: str
    signals: list


def _risk_level(confidence: float, label: str) -> str:
    if label == "legitimate":
        return "LOW"
    if confidence >= 85:
        return "HIGH"
    if confidence >= 60:
        return "MEDIUM"
    return "LOW"


# --------------------------------------------------------------------------
# Auth routes
# --------------------------------------------------------------------------
@app.post("/auth/register", response_model=TokenResponse)
def register(payload: RegisterRequest, db: Session = Depends(get_db)):
    existing = db.query(db_models.User).filter(db_models.User.email == payload.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="An account with this email already exists.")

    user = db_models.User(
        email=payload.email,
        hashed_password=auth.hash_password(payload.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = auth.create_access_token({"sub": user.email})
    return TokenResponse(access_token=token, email=user.email)


@app.post("/auth/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(db_models.User).filter(db_models.User.email == payload.email).first()
    if not user or not auth.verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect email or password.")

    token = auth.create_access_token({"sub": user.email})
    return TokenResponse(access_token=token, email=user.email)


@app.get("/auth/me", response_model=UserResponse)
def me(current_user: db_models.User = Depends(auth.get_current_user)):
    return UserResponse(id=current_user.id, email=current_user.email)


# --------------------------------------------------------------------------
# Core routes
# --------------------------------------------------------------------------
@app.get("/")
def root():
    return {"message": "Phishing Detection API is running"}


@app.post("/predict", response_model=PredictionResponse)
def predict(
    payload: EmailRequest,
    current_user: db_models.User = Depends(auth.get_current_user),
):
    model, vectorizer = _load_artifacts()

    cleaned = clean_text(payload.email)
    if not cleaned:
        raise HTTPException(status_code=400, detail="Email content is empty after cleaning.")

    vec = vectorizer.transform([cleaned])
    label = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    class_index = list(model.classes_).index(label)
    confidence = round(float(proba[class_index]) * 100, 2)

    prediction = "PHISHING" if label == "phishing" else "LEGITIMATE"
    risk = _risk_level(confidence, label)
    signals = extract_signals(payload.email)

    return PredictionResponse(
        prediction=prediction,
        confidence=confidence,
        risk=risk,
        signals=signals,
    )
