from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import os

API_KEY = os.environ.get("API_KEY") # Pobranie klucza API z zmiennych środowiskowych
if not API_KEY:
    raise ValueError("Brak zdefiniowanej zmiennej API_KEY")

app = FastAPI(title="Lab 03")

# sztuczne dane: [wiek, zarobki w tys.], kategoria (0 lub 1) 
X_all = np.array([
    [22, 2.5], [35, 8.0], [45, 12.0], [20, 1.5], [50, 20.0], 
    [30, 5.0], [40, 9.0], [25, 3.0], [55, 15.0], [28, 4.0]
])
y_all = np.array([0, 1, 1, 0, 1, 0, 1, 0, 1, 0])

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
model_accuracy = accuracy_score(y_test, y_pred)

class PredictionInput(BaseModel):
    age: float = Field(..., description="Wiek osoby")
    salary: float = Field(..., description="Zarobki w tysiącach")

# główny endpoint 
@app.get("/")
def read_root():

    masked_key = f"{API_KEY[:3]}***" if API_KEY else "Brak klucza"

    return {"message": "Witaj w API predykcji!", 
             "api_key_status": masked_key}

# endpoint predykcji z walidacją 
@app.post("/predict")
def predict(data: PredictionInput):
    try:
        features = np.array([[data.age, data.salary]])
        prediction = model.predict(features)
        
        return {"prediction": int(prediction[0]), "input_data": data.model_dump()} 
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Błąd przetwarzania: {str(e)}") 

# endpoint info
@app.get("/info")
def get_info():
    return {
        "model_type": "LogisticRegression",
        "features": ["age", "salary"],
        "accuracy": f"{model_accuracy * 100}%", 
        "description": "Model klasyfikacji na podstawie wieku i zarobków."
    }

# endpoint health
@app.get("/health")
def get_health():
    return {"status": "ok"}