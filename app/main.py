from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os

app = FastAPI(title="Phishing Email Detector API")

# using absolute path here so it works no matter where i run uvicorn from
# (learned this the hard way with the relative path issues earlier)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "svm_model.pkl")
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "tfidf_vectorizer.pkl")

# loading model and vectorizer ONCE outside the endpoint
# if i load it inside predict(), it'll reload from disk every single request which is dumb and slow
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# this is basically telling fastapi "expect a json with a text field, string type"
# so if someone sends garbage it gets rejected automatically before hitting my code
class EmailRequest(BaseModel):
    text: str

# what i'm sending back - keeping both a readable label and the raw 0/1
# in case whoever's using this api wants the number instead of the string
class PredictionResponse(BaseModel):
    prediction: str
    label: int

# just a basic root route to check if the server's even alive
@app.get("/")
def read_root():
    return {"message": "Phishing Email Detector API is running"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: EmailRequest):
    # important: using .transform() here NOT .fit_transform()
    # fit_transform would build a whole new vocabulary from just this one email which makes no sense
    # gotta reuse the exact same vectorizer that was fit on the training data
    text_vector = vectorizer.transform([request.text])

    # predict() returns an array even though we only gave it one email, so grabbing index 0
    prediction = model.predict(text_vector)[0]

    return PredictionResponse(
        prediction="phishing" if prediction == 1 else "legitimate",
        # sklearn gives back numpy int64 not a normal python int
        # fastapi can be weird about serializing that to json so just casting it to be safe
        label=int(prediction)
    )