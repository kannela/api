# api_pneumonia_fastapi.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn, io, os
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf

# ---- Configuración -----------------------------------------------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "modelo_neumonia.h5")   # cambia si usas otra ruta
IMG_SIZE   = (150, 150)
THRESHOLD  = 0.5
LABELS     = {0: "NORMAL", 1: "PNEUMONIA"}

# ---- Carga del modelo al arrancar la app -------------------------------------------
model = tf.keras.models.load_model("modelo_neumonia")

# ---- FastAPI -----------------------------------------------------------------------
app = FastAPI(title="Pneumonia X‑ray Classifier API")

# CORS (permite peticiones desde tu Streamlit local)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # production: limita el dominio
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Helpers -----------------------------------------------------------------------
def preprocess(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(IMG_SIZE)
    array = img_to_array(image) / 255.0
    return np.expand_dims(array, axis=0)       # (1,150,150,3)

# ---- Endpoint ----------------------------------------------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    tensor    = preprocess(img_bytes)
    prob      = model.predict(tensor)[0][0]
    idx       = int(prob > THRESHOLD)
    return {"prediction": LABELS[idx], "probability": float(prob)}

# ---- Ejecución local (útil para depurar) -------------------------------------------
if __name__ == "__main__":
    uvicorn.run("api_pneumonia_fastapi:app", host="0.0.0.0", port=8000, reload=True)
