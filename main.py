import io
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from ultralytics import YOLO

app = FastAPI(title="Maternal Food Detection API")

# Allow Mobile App Requests (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model
model = YOLO("best.pt")

# Ping Bot / Health Check Endpoint (Sleep වීම වැළැක්වීමට)
@app.get("/health")
def health_check():
    return {"status": "alive", "service": "maternal-food-detector"}

@app.post("/predict")
async def predict_food(file: UploadFile = File(...), conf_threshold: float = 0.25):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # YOLO Inference
        results = model.predict(source=image, conf=conf_threshold)
        
        detections = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0].item())
            cls_name = model.names[cls_id]
            confidence = float(box.conf[0].item())
            
            # Bounding box coordinates [x1, y1, x2, y2]
            xyxy = box.xyxy[0].tolist()
            
            detections.append({
                "food_id": cls_name,
                "confidence": round(confidence, 3),
                "box": [round(coord, 1) for coord in xyxy]
            })
            
        return {
            "success": True,
            "count": len(detections),
            "detections": detections
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))