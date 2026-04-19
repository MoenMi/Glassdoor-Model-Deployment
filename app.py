from contextlib import asynccontextmanager
from pathlib import Path
import re

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from nltk.corpus import stopwords

from models import PredictRequest, PredictResponse


APP_DIR = Path(__file__).resolve().parent
LOGREG_MODEL_PATH = APP_DIR / "models" / "final_logistic_regression_model.joblib"
DATA_PATH = APP_DIR / "data" / "2023_jan1_data.parquet"

LOGREG_MODEL = None
STOPWORDS_SET = set(stopwords.words("english"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global LOGREG_MODEL
    if not LOGREG_MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found at {LOGREG_MODEL_PATH}")
    LOGREG_MODEL = joblib.load(LOGREG_MODEL_PATH)
    yield
    # Shutdown
    model = None


def clean_text(text: str) -> str:
    """Apply lowercase, strip non-alphanumeric characters, strip whitespace, and remove stopwords"""
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [w for w in text.split() if w not in STOPWORDS_SET]
    return " ".join(tokens)


def prefix_field(text: str, prefix: str) -> str:
    """Append original fields to tokens"""
    cleaned = clean_text(text)
    return " ".join(f"{prefix}_{token}" for token in cleaned.split())


def build_text(title: str, pros: str, cons: str) -> str:
    """Merge tokens from each category into one string"""
    title_text = prefix_field(title, "title")
    pros_text = prefix_field(pros, "pro")
    cons_text = prefix_field(cons, "con")
    return " ".join([title_text, pros_text, cons_text])


app = FastAPI(
    title="Glassdoor Review Inference API",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)


@app.post("/predict", response_model=PredictResponse)
def predict_review(payload: PredictRequest):
    global model

    if LOGREG_MODEL is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")

    try:
        # Preprocess the input text using the same pipeline as training
        preprocessed_text = build_text(payload.title, payload.pros, payload.cons)

        # Model pipeline expects a list of strings, not a DataFrame
        pred = LOGREG_MODEL.predict([preprocessed_text])[0]

        prob_recommend = None
        prob_not_recommend = None

        if hasattr(LOGREG_MODEL, "predict_proba"):
            probs = LOGREG_MODEL.predict_proba([preprocessed_text])[0]

            # assumes binary classes ordered as model.classes_
            if len(probs) == 2 and hasattr(LOGREG_MODEL, "classes_"):
                class_to_prob = dict(zip(LOGREG_MODEL.classes_, probs))
                prob_not_recommend = float(class_to_prob.get(False, class_to_prob.get(0, 0.0)))
                prob_recommend = float(class_to_prob.get(True, class_to_prob.get(1, 0.0)))

        label = "recommend" if pred in [1, True] else "not_recommend"

        return PredictResponse(
            prediction=int(pred) if isinstance(pred, (bool, int)) else 1 if label == "recommend" else 0,
            label=label,
            probability_recommend=prob_recommend,
            probability_not_recommend=prob_not_recommend,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.get("/sample-data")
def get_sample_data():
    """
    Get a random row from the 2023 January data
    
    Note that the model is trained on 2022 data, so this data is out of time and was not encountered in the model's training.
    """
    try:
        if not DATA_PATH.exists():
            raise HTTPException(status_code=500, detail=f"Data file not found at {DATA_PATH}")

        df = pd.read_parquet(DATA_PATH)

        if df.empty:
            raise HTTPException(status_code=500, detail="Data file is empty")

        random_row = df.sample(n=1).iloc[0]
        # Replace NaN with None for JSON serialization
        return random_row.where(pd.notna(random_row), None).to_dict()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load sample data: {str(e)}")
