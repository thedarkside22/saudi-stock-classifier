from fastapi import FastAPI,HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import data
import joblib
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

model = joblib.load(MODELS_DIR / "stock_price_model.joblib")


app = FastAPI()

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")



@app.get("/")
def root():
    return FileResponse(BASE_DIR / "static/index.html")



@app.get("/health")
def get_health():
    if model:
        return {"Status": "Active"}
    else:
        return {"Status": "Inactive"}


@app.get("/predict")
def get_prediction(ticker):
    try:
        features = data.load_inference_features(ticker)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    prob = float(model.predict_proba(features)[:, 1][0])
    return {"ticker":ticker, "probability": prob, "as_of":str(features.index[0].date())}


@app.get("/tickers")
def get_tickers():
    return {"Tickers": data.TICKERS}
