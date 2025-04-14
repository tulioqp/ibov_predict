from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
import pandas as pd
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Modelo para validar parâmetros do /update
class UpdateRequest(BaseModel):
    start_date: str
    end_date: str

# Carregar modelo inicial
try:
    model = joblib.load("ibov_model.pkl")
except FileNotFoundError:
    model = None

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/predict", response_class=HTMLResponse)
def predict_form(request: Request):
    return templates.TemplateResponse("predict.html", {"request": request})

@app.get("/predict/api")
def predict_api(days: int = 1):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not trained")
    forecast = model.forecast(steps=days)
    return {"date": pd.Timestamp.now().strftime("%Y-%m-%d"), "prediction": float(forecast.iloc[-1])}

@app.get("/update", response_class=HTMLResponse)
def update_form(request: Request):
    return templates.TemplateResponse("update.html", {"request": request})

@app.post("/update")
def update_data(request: UpdateRequest):
    global model
    try:
        start = pd.to_datetime(request.start_date)
        end = pd.to_datetime(request.end_date)
        if start >= end:
            raise ValueError("start_date must be before end_date")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    ibov = yf.download("^BVSP", start=request.start_date, end=request.end_date)
    if ibov.empty:
        raise HTTPException(status_code=500, detail="Failed to fetch data")
    df = ibov[["Close"]]
    new_model = ARIMA(df["Close"], order=(1, 1, 1))
    new_model_fit = new_model.fit()
    joblib.dump(new_model_fit, "ibov_model.pkl")
    model = new_model_fit
    return {"message": f"Model updated with data from {request.start_date} to {request.end_date}"}