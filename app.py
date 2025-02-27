import numpy as np
import tensorflow as tf
import asyncio
from flask import Flask, jsonify
from fetch_cpu import get_all_cpu_usage  # Import dari fetch_cpu.py

# Load Model LSTM
model = tf.keras.models.load_model("cpu_model.h5", custom_objects={"mse": tf.keras.losses.MeanSquaredError()})

app = Flask(__name__)

async def predict_cpu(cluster_name, data):
    X_input = np.array(data).reshape(1, 3, 1)  # Sesuaikan input
    prediction = model.predict(X_input)[0][0]  # Prediksi CPU Usage
    return cluster_name, prediction

@app.route("/predict", methods=["GET"])
async def predict():
    cpu_data = await get_all_cpu_usage()  # Ambil data CPU dari semua cluster
    
    # Prediksi secara paralel
    tasks = [predict_cpu(name, data) for name, data in cpu_data.items()]
    predictions = await asyncio.gather(*tasks)
    
    # Cari cluster dengan CPU usage prediksi terendah
    best_cluster = min(predictions, key=lambda x: x[1])

    return jsonify({
        "predictions": dict(predictions),
        "best_cluster": best_cluster[0]
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
