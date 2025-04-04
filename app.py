from asyncio import subprocess
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn  # Import nn module from torch
import asyncio
import aiohttp  # Tambah aiohttp untuk request async
from flask import Flask, jsonify
from fetch_cpu import get_all_cpu_usage
from scripts.dataloader import create_dataloader
from scripts.model import Predictor
from scripts.preprocessing import scale , preprocess # Import dari fetch_cpu.py

WEBHOOK_ENDPOINTS = [
    "http://44.215.167.230:30080/api/trigger",
    "http://52.0.214.121:30080/api/trigger",
    "http://52.73.210.243:30080/api/trigger",
]
lookback_window = 5 
features = [
    'rolling_std_CPU_usage',
    'rolling_mean_CPU_usage',
    'transformed_mean_CPU_usage_rate',
    *[f'mean_CPU_usage_rate_-{i*2}min' for i in range(1, lookback_window + 1)]
]



app = Flask(__name__)

async def predict_cpu(cluster_name, data):
    model = Predictor(len(features), 64, 1)
    model.load_state_dict(torch.load("./best_model.pth"))

    with torch.inference_mode():
        model.eval()
        df = pd.DataFrame(data, columns=["mean_CPU_usage_rate"])
        df = preprocess(df, 5,2)
        feature_scaler = MinMaxScaler()
    
        # cpu_usage_data = scale(df.copy(), features, feature_scaler)
        
        # cpu_usage_data = create_dataloader(np.array(cpu_usage_data), 64, lookback_window, 1)
        seq =  df[features].tail(lookback_window).values

        input_data = torch.tensor(seq).to(torch.float32)
        prediction = model(input_data.unsqueeze(0))

        return cluster_name, prediction

async def send_webhook_request(cluster_index):
    url = WEBHOOK_ENDPOINTS[cluster_index]  
    payload = {"selected_cluster": cluster_index}

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as response:
            return await response.text()  
        
@app.route("/predict", methods=["POST"])
async def predict():
    cpu_data = await get_all_cpu_usage()  # Ambil data CPU dari semua cluster

    tasks = [predict_cpu(name, data) for name, data in cpu_data.items()]
    predictions = await asyncio.gather(*tasks)

    # Cari cluster dengan CPU usage prediksi terendah
    best_cluster = min(predictions, key=lambda x: x[1])

    # Kirim request ke webhook
    print("best cluster index ", best_cluster[0])
    webhook_response = await send_webhook_request(best_cluster[0])


    return jsonify({
    "predictions": {key: value.item() for key, value in predictions},  # Convert tensor to float
    "best_cluster": best_cluster[0],
    "webhook_response": webhook_response  # Include webhook response
})

@app.route('/run-terraform', methods=['POST'])
def run_terraform():
    try:
        # Change directory to where the Terraform files are located
        terraform_dir = "./"  # Adjust this path if needed
        os.chdir(terraform_dir)

        # Run `terraform init`
        init_process = subprocess.run(["terraform", "init"], capture_output=True, text=True)
        if init_process.returncode != 0:
            return jsonify({"error": "Terraform init failed", "details": init_process.stderr}), 500

        # Run `terraform apply` with auto-approve
        apply_process = subprocess.run(["terraform", "apply", "-auto-approve"], capture_output=True, text=True)
        if apply_process.returncode != 0:
            return jsonify({"error": "Terraform apply failed", "details": apply_process.stderr}), 500

        return jsonify({"message": "Terraform applied successfully", "output": apply_process.stdout})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


 

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
