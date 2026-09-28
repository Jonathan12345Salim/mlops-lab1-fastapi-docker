# from fastapi import FastAPI
# from pydantic import BaseModel
# import joblib

# app = FastAPI()
# model = joblib.load("model/model.joblib")

# class Features(BaseModel):
#     sepal_length: float
#     sepal_width: float

# @app.post("/predict")
# def predict(f: Features):
#     y = model.predict([[f.sepal_length,
#                         f.sepal_width]])[0]
#     return {"prediction": int(y)}


from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()
model = joblib.load("model/model.joblib")

class Features(BaseModel):
    sepal_length: float
    sepal_width: float

# Tambahkan endpoint health check ini
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(f: Features):
    y = model.predict([[f.sepal_length, f.sepal_width]])[0]
    return {"prediction": int(y)}