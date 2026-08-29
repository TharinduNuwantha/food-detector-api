import io
import os
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field
from ultralytics import YOLO

app = FastAPI(
    title="Maathru Care AI API",
    description="Maternal Food Detection & Gestational Diabetes Risk Prediction Engine",
    version="2.0.0",
)

# 1. CORS Configuration (Allow Mobile App Requests)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Global Model Loading on Server Startup
print("Loading YOLO Model...")
yolo_model = YOLO("best.pt")

print("Loading GDM Diabetes Models & Scalers...")
gdm_imputer = joblib.load("gdm_imputer.pkl")
gdm_scaler = joblib.load("gdm_scaler.pkl")
gdm_model = joblib.load("gdm_prediction_model.pkl")


# 3. Input Schema for Gestational Diabetes Assessment (15 Features)
class DiabetesPredictionInput(BaseModel):
  age: float = Field(..., example=28.0, description="Mother's Age")
  no_of_pregnancy: float = Field(
      ..., example=2.0, description="Total number of pregnancies"
  )
  gestation_in_previous_pregnancy: float = Field(
      ..., example=38.0, description="Gestation weeks in previous pregnancy"
  )
  bmi: float = Field(..., example=24.5, description="Body Mass Index")
  hdl: float = Field(
      ..., example=52.0, description="High-Density Lipoprotein level"
  )
  family_history: int = Field(
      ..., example=0, description="Family history of diabetes (1: Yes, 0: No)"
  )
  unexplained_prenatal_loss: int = Field(
      ...,
      example=0,
      description="Past unexplained prenatal loss (1: Yes, 0: No)",
  )
  large_child_or_birth_default: int = Field(
      ...,
      example=0,
      description="Previous macrosomic baby or birth default (1: Yes, 0: No)",
  )
  pcos: int = Field(
      ..., example=0, description="Polycystic Ovary Syndrome (1: Yes, 0: No)"
  )
  sys_bp: float = Field(
      ..., example=118.0, description="Systolic Blood Pressure"
  )
  dia_bp: float = Field(
      ..., example=78.0, description="Diastolic Blood Pressure"
  )
  ogtt: float = Field(
      ..., example=120.0, description="Oral Glucose Tolerance Test result"
  )
  hemoglobin: float = Field(..., example=12.2, description="Hemoglobin level")
  sedentary_lifestyle: int = Field(
      ..., example=0, description="Sedentary Lifestyle (1: Yes, 0: No)"
  )
  prediabetes: int = Field(
      ..., example=0, description="History of Prediabetes (1: Yes, 0: No)"
  )


# --- Endpoints ---


@app.get("/health")
def health_check():
  return {
      "status": "alive",
      "service": "maathru-care-ai-backend",
      "models_loaded": {
          "yolo_food_detector": True,
          "gdm_diabetes_predictor": True,
      },
  }


@app.post("/predict")
async def predict_food(
    file: UploadFile = File(...), conf_threshold: float = 0.25
):
  try:
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Fast Inference with imgsz=416
    results = yolo_model.predict(
        source=image, conf=conf_threshold, imgsz=416, verbose=False
    )

    detections = []
    for box in results[0].boxes:
      cls_id = int(box.cls[0].item())
      cls_name = yolo_model.names[cls_id]
      confidence = float(box.conf[0].item())
      xyxy = box.xyxy[0].tolist()

      detections.append({
          "food_id": cls_name,
          "confidence": round(confidence, 3),
          "box": [round(coord, 1) for coord in xyxy],
      })

    return {
        "success": True,
        "count": len(detections),
        "detections": detections,
    }
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-diabetes")
async def predict_diabetes(data: DiabetesPredictionInput):
  try:
    # 1. Feature vector alignment
    raw_features = np.array([[
        data.age,
        data.no_of_pregnancy,
        data.gestation_in_previous_pregnancy,
        data.bmi,
        data.hdl,
        data.family_history,
        data.unexplained_prenatal_loss,
        data.large_child_or_birth_default,
        data.pcos,
        data.sys_bp,
        data.dia_bp,
        data.ogtt,
        data.hemoglobin,
        data.sedentary_lifestyle,
        data.prediabetes,
    ]])

    # 2. Preprocessing pipeline: Imputation -> Scaling
    imputed_data = gdm_imputer.transform(raw_features)
    scaled_data = gdm_scaler.transform(imputed_data)

    # 3. Model Inference
    prediction = int(gdm_model.predict(scaled_data)[0])
    probabilities = gdm_model.predict_proba(scaled_data)[0]
    risk_percentage = round(float(probabilities[1] * 100), 2)

    # Risk level classification
    if risk_percentage >= 65:
      risk_level = "High Risk"
      recommendation = (
          "Immediate medical consultation and strict glycemic diet plan"
          " recommended."
      )
    elif risk_percentage >= 35:
      risk_level = "Moderate Risk"
      recommendation = (
          "Monitor daily carbohydrate intake and maintain regular light"
          " maternal exercise."
      )
    else:
      risk_level = "Low Risk"
      recommendation = (
          "Maternal glucose indicators are within normal healthy ranges."
      )

    return {
        "success": True,
        "diabetes_risk": bool(prediction == 1),
        "risk_percentage": risk_percentage,
        "risk_level": risk_level,
        "recommendation": recommendation,
    }
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))