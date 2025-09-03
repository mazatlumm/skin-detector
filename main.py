import io
import os
import requests
import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

app = FastAPI()

# URL DigitalOcean tempat file model disimpan
MODEL_URL = "https://boardingpas.sgp1.cdn.digitaloceanspaces.com/ai-model/pytorch_model.bin"
MODEL_DIR = "./model"
MODEL_FILE = os.path.join(MODEL_DIR, "pytorch_model.bin")

# Download model jika belum ada di lokal
os.makedirs(MODEL_DIR, exist_ok=True)
if not os.path.exists(MODEL_FILE):
    print("📥 Downloading model from DigitalOcean...")
    r = requests.get(MODEL_URL, stream=True)
    with open(MODEL_FILE, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print("✅ Model downloaded!")

# Load model & processor
model = ViTForImageClassification.from_pretrained(MODEL_DIR)
processor = ViTImageProcessor.from_pretrained(MODEL_DIR)
model.eval()

# Ambil label dari config.json (id2label)
id2label = model.config.id2label


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Baca file gambar
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Preprocess
    inputs = processor(images=image, return_tensors="pt")

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)[0]

    # Cari prediksi terbaik
    confidence, predicted_class = torch.max(probs, dim=0)
    result = {
        "class": id2label[predicted_class.item()],
        "confidence": float(confidence.item())
    }

    return result


# Supaya otomatis jalan di port 5700
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5700, reload=False)
