from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import pandas as pd
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="templates"), name="static")

# Modelo para validar parâmetros do /update
class UpdateRequest(BaseModel):
    start_date: str
    end_date: str

# Carregar modelo inicial
try:
    model = joblib.load("ibov_model.pkl")
except FileNotFoundError:
    model = None

# Armazenar a última data do treinamento
last_training_date = None

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/predict", response_class=HTMLResponse)
def predict_form(request: Request):
    return templates.TemplateResponse("predict.html", {"request": request})

@app.get("/predict/api")
def predict_api(days: int = 1):
    if model is None or last_training_date is None:
        raise HTTPException(
            status_code=400,
            detail="Model not trained. Please return to the homepage and update the model in 'Update Model'."
        )
    forecast = model.forecast(steps=days)
    # Usar a última data do treinamento armazenada
    last_date = last_training_date
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days, freq='B')
    return {"date": future_dates[-1].strftime("%Y-%m-%d"), "prediction": float(forecast.iloc[-1])}

@app.get("/update", response_class=HTMLResponse)
def update_form(request: Request):
    return templates.TemplateResponse("update.html", {"request": request})

@app.post("/update")
def update_data(request: UpdateRequest):
    global model, last_training_date
    try:
        start = pd.to_datetime(request.start_date)
        end = pd.to_datetime(request.end_date)
        if start >= end:
            raise ValueError("start_date must be before end_date")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    ibov = yf.download("^BVSP", start=request.start_date, end=request.end_date)
    if ibov.empty:
        raise HTTPException(
            status_code=400,
            detail="Unable to fetch data from Yahoo Finance. Please check the date range and try again."
        )
    df = ibov[["Close"]]
    # Definir frequência do índice como dias úteis
    df.index = pd.to_datetime(df.index)
    df = df.asfreq('B')
    new_model = ARIMA(df["Close"], order=(1, 1, 1))
    new_model_fit = new_model.fit()
    joblib.dump(new_model_fit, "ibov_model.pkl")
    model = new_model_fit
    # Armazenar a última data do treinamento
    last_training_date = end
    return {"message": f"Model updated with data from {request.start_date} to {request.end_date}"}