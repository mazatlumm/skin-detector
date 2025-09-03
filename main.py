import io
import os
import requests
import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

app = FastAPI()

# ===== CORS Middleware =====
origins = [
    "http://localhost:5600", 
    "https://sehat.kediriku.id",      
    "https://skin-disease.alicestech.com", 
    "*"                                  
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== Download & load model =====
MODEL_URL = "https://boardingpas.sgp1.cdn.digitaloceanspaces.com/ai-model/pytorch_model.bin"
MODEL_DIR = "./model"
MODEL_FILE = os.path.join(MODEL_DIR, "pytorch_model.bin")

os.makedirs(MODEL_DIR, exist_ok=True)
if not os.path.exists(MODEL_FILE):
    print("📥 Downloading model from DigitalOcean...")
    r = requests.get(MODEL_URL, stream=True)
    with open(MODEL_FILE, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print("✅ Model downloaded!")

model = ViTForImageClassification.from_pretrained(MODEL_DIR)
processor = ViTImageProcessor.from_pretrained(MODEL_DIR)
model.eval()

id2label = model.config.id2label

# ===== Predict endpoint =====
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)[0]

    confidence, predicted_class = torch.max(probs, dim=0)
    result = {
        "class": id2label[predicted_class.item()],
        "confidence": float(confidence.item())
    }

    return result

# ===== Run server =====
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5700, reload=False)
