from api.schemas import Prediction, TextRequest
from fastapi import FastAPI, HTTPException
from model.predict import Predictor

app = FastAPI(title="Abandonment-Reason Classifier API", version="0.1.0")

predictor = Predictor()  # loads on app start


@app.post("/predict", response_model=Prediction)
def predict(req: TextRequest):
    try:
        return predictor(req.text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
