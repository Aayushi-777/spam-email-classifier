from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import os
from app.utils import clean_text
from fastapi.staticfiles import StaticFiles

app = FastAPI()

templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

MODEL_PATH = "saved_model/model.pkl"
VEC_PATH = "saved_model/vectorizer.pkl"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VEC_PATH)

@app.get("/", response_class = HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class = HTMLResponse)
async def predict(request: Request, message: str = Form(...)):
    cleaned = clean_text(message)
    vec = vectorizer.transform([cleaned])
    prob = model.predict_proba(vec)[0]
    label = "Spam" if prob[1] > prob[0] else "Real"
    confidence = round(max(prob)*100, 2)
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": label,
            "confidence": confidence,
            "message": message
        }
    )