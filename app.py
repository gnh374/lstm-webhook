import numpy as np
import tensorflow as tf
import asyncio
import aiohttp  # Tambah aiohttp untuk request async
from flask import Flask, jsonify
from fetch_cpu import get_all_cpu_usage  # Import dari fetch_cpu.py

WEBHOOK_ENDPOINTS = [
    "http://44.215.167.230:30080",
    "http://52.0.214.121:30080",
    "http://52.73.210.243:30080",
]

# Load Model LSTM
model = tf.keras.models.load_model("cpu_model.h5", custom_objects={"mse": tf.keras.losses.MeanSquaredError()})

app = Flask(__name__)

async def predict_cpu(cluster_name, data):
    X_input = np.array(data).reshape(1, 3, 1)  # Sesuaikan input
    prediction = model.predict(X_input)[0][0]  # Prediksi CPU Usage
    return cluster_name, prediction

async def send_webhook_request(cluster_index):
    url = WEBHOOK_ENDPOINTS[cluster_index]  # Ambil URL webhook sesuai index
    payload = {"selected_cluster": cluster_index}

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as response:
            return await response.text()  # Response dari webhook

@app.route("/predict", methods=["POST"])
async def predict():
    cpu_data = await get_all_cpu_usage()  # Ambil data CPU dari semua cluster
    
    # Prediksi secara paralel
    tasks = [predict_cpu(name, data) for name, data in cpu_data.items()]
    predictions = await asyncio.gather(*tasks)
    
    # Cari cluster dengan CPU usage prediksi terendah
    best_cluster = min(predictions, key=lambda x: x[1])

    # Kirim request ke webhook
    print("best cluster index ", best_cluster[0])
    webhook_response = await send_webhook_request(best_cluster[0])

    return jsonify({
        "predictions": dict(predictions),
        "best_cluster": best_cluster[0],
        "webhook_response": webhook_response  # Tambahkan response dari webhook
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
