
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image


app = FastAPI(
    title="Distracted Driver Detector API",
    description="API to classify driver behaviour from an image",
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
    "Safe Driving",
    "Texting - Right",
    "Talking on the Phone - Right",
    "Texting - Left",
    "Talking on the Phone - Left",
    "Operating the Radio",
    "Drinking",
    "Reaching Behind",
    "Hair and Makeup",
    "Talking to Passenger",
]


def predict_image_file(file_obj, top_k: int = 3):
    """
    file_obj: file-like object (UploadFile.file)
    """
    img = Image.open(file_obj).convert("RGB").resize((224, 224))

    x = np.array(img, dtype=np.float32)
    x = tf.convert_to_tensor(x)[None, ...] 

    logits = model(x, training=False)
    probs = tf.nn.softmax(logits, axis=-1).numpy()[0]

    top_idxs = np.argsort(-probs)[:top_k]
    results = [(class_names[i], float(probs[i])) for i in top_idxs]
    return results

@app.get("/")
def index():
    return {"message": "Distracted driver API is running"}

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if image.filename == "":
        raise HTTPException(status_code=400, detail="Empty filename.")

    try:
        preds = predict_image_file(image.file, top_k=3)

        preds_json = [
            {"label": label, "confidence": conf}
            for (label, conf) in preds
        ]

        return JSONResponse(content={"predictions": preds_json})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
