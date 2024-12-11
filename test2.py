import serial
import joblib
import numpy as np
from datetime import datetime
import threading
import time
import streamlit as st
import pandas as pd
from queue import Queue
import Levenshtein  # Ensure this library is installed
from collections import Counter

# Load the saved model and scaler
model = joblib.load('evil_twin_rf_model.pkl')
scaler = joblib.load('evil_twin_scaler.pkl')

# Parameters
SCAN_THRESHOLD = 10  # Number of scans before classification
RSSI_VARIATION_THRESHOLD = 20  # RSSI fluctuation threshold to label as suspicious

# Shared resources
accumulated_scan_data = []
scan_counter = 0
network_history = {}
network_classification = {}
shared_results = Queue()  # Thread-safe queue for sharing data between threads

# Debug logging function
def debug_log(message):
    print(f"[DEBUG] {datetime.now().strftime('%H:%M:%S')}: {message}")

# Function to parse ESP data
def parse_esp_data(raw_data):
    debug_log(f"Raw data received: {raw_data}")
    try:
        if not raw_data or raw_data.startswith("Number of networks"):
            return []

        lines = raw_data.strip().split("\n")
        parsed_networks = []

        for line in lines:
            fields = line.split(',')
            if len(fields) < 12:
                debug_log(f"Incomplete data ignored: {line}")
                continue

            parsed_data = {
                "SSID": fields[1],
                "BSSID": fields[3],
                "RSSI": int(fields[5]),
                "Channel": int(fields[7]),
                "Encryption": int(fields[9]),
                "Time": fields[11]
            }
            parsed_networks.append(parsed_data)

        debug_log(f"Parsed {len(parsed_networks)} networks.")
        return parsed_networks
    except Exception as e:
        debug_log(f"Error parsing data: {e}")
        return []

# Function to calculate additional features for classification
def calculate_features(network_data):
    ssid_bssid_key = f"{network_data['SSID']}_{network_data['BSSID']}"
    current_time = datetime.strptime(network_data["Time"], "%H:%M:%S")
    rssi = network_data["RSSI"]

    if ssid_bssid_key not in network_history:
        network_history[ssid_bssid_key] = {
            "SSID": network_data["SSID"],
            "BSSID": network_data["BSSID"],
            "RSSI_values": [],
            "Timestamps": [],
            "Channel": network_data["Channel"],
            "Encryption": network_data["Encryption"]
        }

    history = network_history[ssid_bssid_key]
    history["RSSI_values"].append(rssi)
    history["Timestamps"].append(current_time)

    levenshtein_distance = 0  # Placeholder
    detection_frequency = len(history["Timestamps"])
    rssi_variance = np.var(history["RSSI_values"])

    if len(history["Timestamps"]) > 1:
        time_diff = (history["Timestamps"][-1] - history["Timestamps"][-2]).total_seconds()
        rssi_diff = history["RSSI_values"][-1] - history["RSSI_values"][-2]
        signal_gradient = rssi_diff / time_diff if time_diff > 0 else 0
    else:
        signal_gradient = 0

    signal_fluctuation = max(history["RSSI_values"]) - min(history["RSSI_values"])
    encryption_hierarchy = history["Encryption"]
    congested_channel = False

    feature_vector = [
        rssi,
        history["Channel"],
        levenshtein_distance,
        detection_frequency,
        rssi_variance,
        signal_gradient,
        signal_fluctuation,
        encryption_hierarchy,
        int(congested_channel)
    ]
    return feature_vector

# Function to classify network using the trained model
def classify_network(feature_vector):
    feature_vector_scaled = scaler.transform([feature_vector])
    prediction = model.predict(feature_vector_scaled)
    return int(prediction[0])  # 0: Legitimate, 1: Evil Twin

# Function to process and classify after accumulating data for SCAN_THRESHOLD scans
def classify_after_scans():
    global scan_counter, accumulated_scan_data
    scan_counter += 1

    debug_log(f"Scan counter: {scan_counter}")
    if scan_counter >= SCAN_THRESHOLD:
        # Track classifications for each network (SSID_BSSID as key)
        network_classifications = {}

        # Classify each network in the accumulated scan data
        for network in accumulated_scan_data:
            feature_vector = calculate_features(network)
            prediction = classify_network(feature_vector)
            classification = 'Evil Twin' if prediction == 1 else 'Legitimate'

            ssid_bssid_key = f"{network['SSID']}_{network['BSSID']}"
            if ssid_bssid_key not in network_classifications:
                network_classifications[ssid_bssid_key] = []

            network_classifications[ssid_bssid_key].append(classification)

        # Determine majority vote for each network
        results = []
        for ssid_bssid_key, classifications in network_classifications.items():
            # Get the majority classification (most common)
            majority_class = Counter(classifications).most_common(1)[0][0]

            # Find the corresponding network and append the result
            network = next(network for network in accumulated_scan_data if f"{network['SSID']}_{network['BSSID']}" == ssid_bssid_key)
            results.append({**network, "Classification": majority_class})

        # Send results to shared queue for Streamlit
        shared_results.put(results)

        # Reset for next round
        accumulated_scan_data.clear()
        scan_counter = 0
        debug_log("Classification complete and shared.")

# Serial reading function
def read_from_esp(port, baudrate=9600):
    try:
        with serial.Serial(port, baudrate, timeout=1) as ser:
            current_scan_data = []
            while True:
                if ser.in_waiting > 0:
                    raw_data = ser.readline().decode('utf-8', errors='replace').strip()
                    debug_log(f"ESP Output: {raw_data}")

                    if raw_data.startswith("Number of networks"):
                        if current_scan_data:
                            accumulated_scan_data.extend(current_scan_data)
                            classify_after_scans()
                            current_scan_data = []
                        continue

                    networks = parse_esp_data(raw_data)
                    if networks:
                        current_scan_data.extend(networks)
    except serial.SerialException as e:
        debug_log(f"Serial Error: {e}")
    except Exception as e:
        debug_log(f"Unexpected Error: {e}")

# Streamlit app function
def main():
    st.title("Evil Twin Detection")
    st.markdown("### Real-Time Network Classification")

    placeholder = st.empty()

    while True:
        if not shared_results.empty():
            results = shared_results.get()
            df = pd.DataFrame(results)
            with placeholder.container():
                st.table(df)
        time.sleep(1)

# Start serial reading thread
esp_thread = threading.Thread(target=read_from_esp, args=('COM5', 115200), daemon=True)
esp_thread.start()

if __name__ == "__main__":
    main()
