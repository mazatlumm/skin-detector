import io
import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

app = FastAPI()

# Load model & processor
model_path = "./model"
model = ViTForImageClassification.from_pretrained(model_path)
processor = ViTImageProcessor.from_pretrained("./model")
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


# Tambahkan ini supaya otomatis jalan di port 5700
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5700, reload=False)
