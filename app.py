from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image

app = FastAPI(
    title="Distracted Driver Detector API",
    description="API to classify driver behaviour from an image (5-class model)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model = keras.models.load_model("distracted_driver_detector.keras")

class_names = [
    "Normal Driving",
    "Talking on the Phone - Right",
    "Texting",
    "Talking on the Phone - Left",
    "Operating the Radio",
]

CONFIDENCE_THRESHOLD = 0.7
IMG_SIZE = (224, 224)


def predict_image_file(file_obj, top_k: int = 2):
    """Predict from a file-like object using your existing approach."""
    img = Image.open(file_obj).convert("RGB").resize(IMG_SIZE)

    x = np.array(img, dtype=np.float32)
    x = tf.convert_to_tensor(x)[None, ...] 

    probs = model(x, training=False).numpy()[0]

    top_idxs = np.argsort(-probs)[:top_k]

    results = []
    for i in top_idxs:
        conf = float(probs[i])
        results.append(
            {
                "label": class_names[i],
                "confidence": conf,
                "confidence_percent": round(conf * 100, 2),
                "above_threshold": conf >= CONFIDENCE_THRESHOLD,
            }
        )

    return results



@app.get("/")
def index():
    return {"message": "Distracted Driver Detector API (5-class) is running."}


@app.post("/predict")
async def predict(image: UploadFile = File(...), top_k: int = 2):
    if image.filename == "":
        raise HTTPException(status_code=400, detail="Empty filename.")

    try:
        preds = predict_image_file(image.file, top_k=top_k)
        return JSONResponse(content={"predictions": preds})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
