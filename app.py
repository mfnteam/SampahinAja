from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import tensorflow as tf
import numpy as np
import io

app = FastAPI()

# Mengizinkan React (Frontend) mengakses API ini
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Di produksi, ganti dengan URL React Anda
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model h5 yang sudah Anda latih sebelumnya
MODEL = tf.keras.models.load_model('model_klasifikasi_sampah.h5')
LABELS = ['Anorganik', 'Limbah B3', 'Organik', 'Residu']


def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    img = img.resize((224, 224))  # Sesuaikan dengan ukuran input model Anda
    img_array = np.array(img)

    # Jika gambar memiliki channel RGBA (4 channel), ubah ke RGB (3 channel)
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]

    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalisasi
    return img_array


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read file gambar
    image_bytes = await file.read()

    # Preprocessing
    input_data = preprocess_image(image_bytes)

    # Prediksi
    predictions = MODEL.predict(input_data)
    score = predictions[0]

    # Ambil hasil tertinggi
    highest_idx = np.argmax(score)
    hasil_kelas = LABELS[highest_idx]
    persentase = float(score[highest_idx]) * 100

    return {
        "status": "success",
        "prediction": hasil_kelas,
        "confidence": round(persentase, 2)
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)