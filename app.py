import subprocess
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn  # Import nn module from torch
import asyncio
import aiohttp  # Tambah aiohttp untuk request async
from flask import Flask, jsonify
from fetch_cpu import get_all_cpu_usage, get_all_cpu_usage_new
from scripts.dataloader import create_dataloader
from scripts.model import Predictor
from scripts.preprocessing import scale , preprocess # Import dari fetch_cpu.py

WEBHOOK_ENDPOINTS = [
    "http://54.227.47.149:30080/api/trigger",
    "http://54.162.198.32:30080/api/trigger",
    "http://98.85.113.75:30080/api/trigger",
]

CPU_MAX = [
    3,3,3
]
lookback_window = 5 
lag_features = []

for i in range(1, lookback_window + 1):
    lag_features.append(f'mean_CPU_usage_rate_-{i*2}min')

features = [
    'transformed_mean_CPU_usage_rate',
    'rolling_std_CPU_usage',
    'rolling_mean_CPU_usage',
    *lag_features,
]



app = Flask(__name__)

async def predict_cpu_new(cluster_name, data):
    model = Predictor(len(features), 128, 1)
    model.load_state_dict(torch.load("./best_model.pt"))

    with torch.inference_mode():
        model.eval()
        # Create DataFrame from input data
        df = pd.DataFrame(data, columns=["mean_CPU_usage_rate"])
        
        # Preprocess data (this already performs some transformations)
        df = preprocess(df, 5, 2)
        
        # Apply scaling before prediction
        feature_scaler = MinMaxScaler()
        
        # Fit scaler on the features we're using
        feature_values = df[features].values
        feature_scaler.fit(feature_values)
        
        # Scale the features
        scaled_features = feature_scaler.transform(feature_values)
        
        # Get the last lookback_window rows for model input
        seq = scaled_features[-lookback_window:]
        
        # Convert to tensor for model
        input_data = torch.tensor(seq).to(torch.float32)
        
        # Make prediction
        raw_prediction = model(input_data.unsqueeze(0))
        
        # Scale back the prediction to original range
        # Assuming the prediction is a single value representing CPU usage
        # and should be in the same scale as the original data
        
        # Get min/max from the original data to scale back
        original_min = df["mean_CPU_usage_rate"].min()
        original_max = df["mean_CPU_usage_rate"].max()
        
        # Scale back (this is a simple approach - adjust based on your exact scaling method)
        scaled_back_prediction = raw_prediction * (original_max - original_min) + original_min

        return cluster_name, scaled_back_prediction
async def predict_cpu(cluster_name, data):
    model = Predictor(len(features), 128, 1)
    model.load_state_dict(torch.load("./best_model.pt"))

    with torch.inference_mode():
        model.eval()
        df = pd.DataFrame(data, columns=["mean_CPU_usage_rate"])
        df = preprocess(df, 5,2)
        # feature_scaler = MinMaxScaler()
    
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
    print(cpu_data)
    tasks = [predict_cpu_new(name, data) for name, data in cpu_data.items()]
    predictions = await asyncio.gather(*tasks)
    print(predictions)
    # Normalize predictions by dividing by CPU_MAX
    cpu_utilization = []
    for name, prediction in predictions:
        # Convert name to integer if it's a string index
        idx = int(name) if isinstance(name, str) and name.isdigit() else name
        # Normalize by dividing by corresponding CPU_MAX value
        normalized_value = prediction / CPU_MAX[idx]
        cpu_utilization.append((name, normalized_value))

    # Cari cluster dengan CPU usage prediksi terendah (normalized)
    best_cluster = min(cpu_utilization, key=lambda x: x[1])
  
    print("best cluster index ", best_cluster[0])
    webhook_response = await send_webhook_request(best_cluster[0])


    return jsonify({
    "predictions": {key: value.item() for key, value in cpu_utilization},  # Convert tensor to float
    "best_cluster": best_cluster[0],
    "webhook_response": webhook_response  # Include webhook response
})

# @app.route("/predict-combined", methods=["POST"])
# async def predict_combined():
#     # Fetch CPU data from all clusters
#     cpu_data = await get_all_cpu_usage_new()
    
#     # Extract the down cluster data but keep a copy
#     down_cluster_data = cpu_data.pop("down_cluster", None)
    
#     # First predict CPU usage for all clusters individually
#     all_tasks = []
    
#     # Add tasks for active clusters
#     for name, data in cpu_data.items():
#         all_tasks.append(predict_cpu_new(name, data))
    
#     # Add task for down cluster if it exists
#     down_prediction = None
#     if down_cluster_data and "values" in down_cluster_data:
#         all_tasks.append(predict_cpu_new("down_cluster", down_cluster_data["values"]))
    
#     # Wait for all predictions
#     all_predictions = await asyncio.gather(*all_tasks)
    
#     # Separate down cluster prediction
#     active_predictions = []
#     for pred in all_predictions:
#         if pred[0] == "down_cluster":
#             down_prediction = pred[1]
#         else:
#             active_predictions.append(pred)
    
#     # Add down cluster prediction to all other clusters' predictions
#     combined_predictions = []
#     if down_prediction is not None:
#         for name, prediction in active_predictions:
#             # Add down cluster prediction to this cluster's prediction
#             combined_predictions.append((name, prediction + down_prediction))
#     else:
#         combined_predictions = active_predictions
#     # Normalize predictions based on CPU_MAX values before finding minimum
#     cpu_utilization = []
#     for name, prediction in combined_predictions:
#         # Convert name to integer if it's a string index
#         idx = int(name) if isinstance(name, str) and name.isdigit() else name
#         # Normalize by dividing by corresponding CPU_MAX value
#         normalized_value = prediction / CPU_MAX[idx]
#         cpu_utilization.append((name, normalized_value))

    
#     # Find cluster with lowest normalized predicted CPU usage
#     best_cluster = min(cpu_utilization, key=lambda x: x[1])
#     best_cluster_value = best_cluster[1].item()  # Get the value and convert tensor to float
    
#     # Check if even the best cluster is highly utilized (>80%)
#     terraform_response = None
#     webhook_response = None

#     if best_cluster_value > 0.8:
#         # If all clusters are heavily loaded, create new resources with Terraform
#         terraform_response = run_terraform()
#         print("Terraform applied successfully")
#     else:
#         # Otherwise, send webhook request to the best cluster
#         webhook_response = await send_webhook_request(best_cluster[0])
#         print("best cluster: ", best_cluster[0])
#     return jsonify({
#         "predictions": {key: value.item() for key, value in combined_predictions},
#         "cpu_utilization": {key: value.item() for key, value in cpu_utilization},
#         "original_predictions": {key: value.item() for key, value in active_predictions},
#         "down_cluster_prediction": down_prediction.item() if down_prediction is not None else None,
#         "best_cluster": best_cluster[0],
#         "best_cluster_utilization": best_cluster_value,
#         "terraform_applied": terraform_response is not None,
#         "terraform_response": terraform_response,
#         "webhook_response": webhook_response,
#         "down_cluster_status": down_cluster_data.get("status") if down_cluster_data else "unknown"
#     })

@app.route("/predict-combined", methods=["POST"])
async def predict_combined():
    # Fetch CPU data from all clusters
    cpu_data = await get_all_cpu_usage_new()
    
    # Extract the down cluster data
    down_cluster_data = cpu_data.pop("down_cluster", None)
    
    # Predict recursively for all clusters
    all_tasks = []
    for name, data in cpu_data.items():
        all_tasks.append(predict_recursive(name, data))
    
    # Predict recursively for the down cluster if it exists
    down_predictions = None
    if down_cluster_data and "values" in down_cluster_data:
        down_predictions = await predict_recursive("down_cluster", down_cluster_data["values"])
    
    # Wait for all predictions
    all_predictions = await asyncio.gather(*all_tasks)
    
    # Combine predictions with the down cluster predictions
    combined_predictions = []
    for name, predictions in all_predictions:
        # Add the down cluster predictions to the current cluster's predictions
        if down_predictions:
            predictions = [p + down_predictions[i] for i, p in enumerate(predictions)]
        # Get the maximum prediction for the cluster
        max_prediction = max(predictions)
        combined_predictions.append((name, max_prediction))
    
    # Normalize predictions based on CPU_MAX
    normalized_predictions = []
    for name, max_prediction in combined_predictions:
        idx = int(name) if isinstance(name, str) and name.isdigit() else name
        normalized_value = max_prediction / CPU_MAX[idx]
        normalized_predictions.append((name, normalized_value))
    
    # Find the cluster with the minimum utilization
    best_cluster = min(normalized_predictions, key=lambda x: x[1])
    best_cluster_value = best_cluster[1]  # Get the normalized value
    
    # Check if even the best cluster is highly utilized (>80%)
    terraform_response = None
    webhook_response = None
    if best_cluster_value >= 0.8:
        # If all clusters are heavily loaded, create new resources with Terraform
        terraform_response = run_terraform()
        print("Terraform applied successfully")
    else:
        # Otherwise, send webhook request to the best cluster
        webhook_response = await send_webhook_request(best_cluster[0])
        print("Best cluster:", best_cluster[0])
    
    return jsonify({
        "predictions": {key: value for key, value in combined_predictions},
        "cpu_utilization": {key: value for key, value in normalized_predictions},
        "best_cluster": best_cluster[0],
        "best_cluster_utilization": best_cluster_value,
        "terraform_applied": terraform_response is not None,
        "terraform_response": terraform_response,
        "webhook_response": webhook_response,
        "down_cluster_status": down_cluster_data.get("status") if down_cluster_data else "unknown"
    })

async def predict_recursive(cluster_name, data):
    model = Predictor(len(features), 128, 1)
    model.load_state_dict(torch.load("./best_model.pt"))

    with torch.inference_mode():
        model.eval()
        df = pd.DataFrame(data, columns=["mean_CPU_usage_rate"])
        df = preprocess(df, 5, 2)

        # Apply scaling
        feature_scaler = MinMaxScaler()
        feature_values = df[features].values
        feature_scaler.fit(feature_values)
        scaled_features = feature_scaler.transform(feature_values)

        # Recursive prediction for 6 minutes (3 intervals of 2 minutes)
        predictions = []
        for _ in range(3):
            seq = scaled_features[-lookback_window:]
            input_data = torch.tensor(seq).to(torch.float32)
            raw_prediction = model(input_data.unsqueeze(0))
            predictions.append(raw_prediction.item())

            # Append the prediction to the input series and move the window forward
            new_row = np.zeros((1, len(features)))
            new_row[0, 0] = raw_prediction.item()  # Assuming the first feature is the target
            scaled_features = np.vstack([scaled_features, new_row])

        return cluster_name, predictions
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
