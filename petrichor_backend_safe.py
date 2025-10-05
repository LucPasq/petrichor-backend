#!/usr/bin/env python3
"""
Petrichor Backend with graceful import handling
"""

import re
import io
import csv
import threading
import time
import uuid
import json
import os
import sys
from datetime import datetime, timedelta

# Try to import required packages with graceful fallbacks
try:
    from flask import Flask, request, jsonify, send_from_directory, Response
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError as e:
    print(f"Flask not available: {e}")
    print("Please install Flask: pip install Flask Flask-CORS")
    FLASK_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError as e:
    print(f"Requests not available: {e}")
    print("Please install requests: pip install requests")
    REQUESTS_AVAILABLE = False

try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError as e:
    print(f"Pandas/NumPy not available: {e}")
    print("Please install: pip install pandas numpy")
    PANDAS_AVAILABLE = False

# Optional AI components
try:
    from weather_data_collector import WeatherDataCollector
    AI_DATA_AVAILABLE = True
except ImportError as e:
    print(f"AI data collector not available: {e}")
    AI_DATA_AVAILABLE = False

try:
    from weather_prediction_model import WeatherPredictionModel
    AI_MODEL_AVAILABLE = True
except ImportError as e:
    print(f"AI prediction model not available: {e}")
    AI_MODEL_AVAILABLE = False

try:
    from notebook_integration import NotebookExecutor
    NOTEBOOK_AVAILABLE = True
except ImportError as e:
    print(f"Notebook integration not available: {e}")
    NOTEBOOK_AVAILABLE = False

# Exit if critical components are missing
if not FLASK_AVAILABLE or not REQUESTS_AVAILABLE:
    print("\\nCritical components missing. Please install required packages:")
    print("pip install Flask Flask-CORS requests")
    sys.exit(1)

# Initialize Flask app
app = Flask(__name__)
CORS(app, origins="*", methods=["GET", "POST", "OPTIONS"], allow_headers=["Content-Type", "Authorization"])

progress_dict = {}  # {task_id: {progress, total, status, result}}
RESULTS_HISTORY_FILE = "results_history.json"

# Initialize components that are available
weather_collector = WeatherDataCollector() if AI_DATA_AVAILABLE else None
weather_model = WeatherPredictionModel() if AI_MODEL_AVAILABLE else None
notebook_executor = NotebookExecutor() if NOTEBOOK_AVAILABLE else None

def classify_rainfall(mean_mm):
    if mean_mm == 0:
        return "Sunny"
    elif mean_mm <= 2:
        return "Drizzle"
    elif mean_mm <= 10:
        return "Showers"
    else:
        return "Heavy Rain"

def build_api_url(year, month, day, lat, lon):
    start_date = f"{year}-{month:02d}-{day:02d}T00"
    day_dt = datetime(year, month, day)
    end_dt = day_dt + timedelta(days=1)
    end_date = end_dt.strftime("%Y-%m-%dT00")
    return (
        "https://hydro1.gesdisc.eosdis.nasa.gov/daac-bin/access/timeseries.cgi"
        "?variable=NLDAS2%3ANLDAS_FORA0125_H_v2.0%3ARainf"
        "&type=asc2"
        f"&location=GEOM%3APOINT({lon}%2C%20{lat})"
        f"&startDate={start_date}"
        f"&endDate={end_date}"
    )

def parse_daily_total(raw_text):
    lines = raw_text.strip().split('\\n')
    first_data_idx = None
    for idx, line in enumerate(lines):
        if re.match(r"\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}", line):
            first_data_idx = idx
            break
    
    if first_data_idx is None:
        return None, 0.0
    date = lines[first_data_idx][:10]
    total = 0.0
    for line in lines[first_data_idx:]:
        parts = line.strip().split('\\t')
        if len(parts) == 2 and re.match(r"\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}", parts[0]):
            try:
                total += float(parts[1])
            except ValueError:
                pass
    return date, total

def save_result_to_history(new_entry):
    history = []
    if os.path.exists(RESULTS_HISTORY_FILE):
        try:
            with open(RESULTS_HISTORY_FILE, "r") as f:
                history = json.load(f)
        except Exception:
            history = []
    history.insert(0, new_entry)
    history = history[:10]
    with open(RESULTS_HISTORY_FILE, "w") as f:
        json.dump(history, f)

def get_results_history():
    if os.path.exists(RESULTS_HISTORY_FILE):
        try:
            with open(RESULTS_HISTORY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/health')
def health_check():
    status = {
        "status": "healthy",
        "message": "Backend is running",
        "components": {
            "flask": FLASK_AVAILABLE,
            "requests": REQUESTS_AVAILABLE,
            "pandas": PANDAS_AVAILABLE,
            "ai_data_collector": AI_DATA_AVAILABLE,
            "ai_model": AI_MODEL_AVAILABLE,
            "notebook_integration": NOTEBOOK_AVAILABLE
        }
    }
    return jsonify(status)

@app.route('/rainfall-summary', methods=['POST'])
def rainfall_summary():
    req = request.get_json()
    target_date = req.get("date")
    lat = req.get("lat")
    lon = req.get("lon")
    location_name = req.get("location_name", "")
    if not target_date or lat is None or lon is None:
        return jsonify({'error': 'Missing date or location.'}), 400

    try:
        dt = datetime.strptime(target_date, '%Y-%m-%d')
    except ValueError:
        return jsonify({'error': 'Incorrect date format. Use YYYY-MM-DD.'}), 400

    # Generate a unique task ID
    task_id = str(uuid.uuid4())
    progress_dict[task_id] = {
        "progress": 0,
        "total": 10,
        "status": "running",
        "result": None
    }
    threading.Thread(target=rainfall_worker, args=(task_id, dt, lat, lon, location_name)).start()
    return jsonify({"task_id": task_id})

def rainfall_worker(task_id, dt, lat, lon, location_name):
    month = dt.month
    day = dt.day
    year = dt.year

    daily_results = []
    try:
        for i in range(1, 11):
            this_year = year - i
            api_url = build_api_url(this_year, month, day, lat, lon)
            try:
                api_resp = requests.get(api_url, timeout=30)
                api_resp.raise_for_status()
                raw = api_resp.text
                date, total = parse_daily_total(raw)
                if date:
                    daily_results.append({"date": date, "rainfall_mm": round(total, 4)})
                else:
                    daily_results.append({"date": f"{this_year}-{month:02d}-{day:02d}", "rainfall_mm": None})
            except requests.exceptions.Timeout:
                daily_results.append({"date": f"{this_year}-{month:02d}-{day:02d}", "rainfall_mm": None, "error": "Request timeout"})
            except requests.exceptions.RequestException as e:
                daily_results.append({"date": f"{this_year}-{month:02d}-{day:02d}", "rainfall_mm": None, "error": f"Request failed: {str(e)}"})
            except Exception as e:
                daily_results.append({"date": f"{this_year}-{month:02d}-{day:02d}", "rainfall_mm": None, "error": str(e)})
            
            progress_dict[task_id]["progress"] = i

        values = [entry["rainfall_mm"] for entry in daily_results if entry["rainfall_mm"] is not None]
        avg = round(sum(values)/len(values), 4) if values else 0.0
        category = classify_rainfall(avg)

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["date", "rainfall_mm"])
        for entry in daily_results:
            writer.writerow([entry["date"], entry["rainfall_mm"]])
        output.seek(0)
        csv_data = output.read()

        result_obj = {
            "average_rainfall_mm": avg,
            "category": category,
            "history": daily_results,
            "csv": csv_data
        }
        progress_dict[task_id]["result"] = result_obj
        progress_dict[task_id]["status"] = "done"

        entry = {
            "date": dt.strftime("%Y-%m-%d"),
            "location": {
                "lat": lat,
                "lon": lon,
                "name": location_name
            },
            "average_rainfall_mm": avg,
            "category": category
        }
        save_result_to_history(entry)
        
    except Exception as e:
        progress_dict[task_id]["status"] = "error"
        progress_dict[task_id]["error"] = str(e)
        print(f"Worker error for task {task_id}: {str(e)}")

@app.route('/progress/<task_id>')
def get_progress(task_id):
    p = progress_dict.get(task_id)
    if not p:
        return jsonify({"error": "Invalid task ID"}), 404
    resp = {
        "progress": p["progress"],
        "total": p["total"],
        "status": p["status"]
    }
    if p["status"] == "done":
        resp["result"] = p["result"]
    elif p["status"] == "error":
        resp["error"] = p.get("error", "Unknown error occurred")
    return jsonify(resp)

@app.route('/results-history')
def results_history():
    return jsonify(get_results_history())

# AI-related endpoints (only if components are available)
if AI_DATA_AVAILABLE and weather_collector:
    @app.route('/collect-training-data', methods=['POST'])
    def collect_training_data():
        if not PANDAS_AVAILABLE:
            return jsonify({'error': 'Pandas not available. Please install: pip install pandas numpy'}), 500
        
        req = request.get_json()
        start_date = req.get("start_date", "2024-08-01T00")
        end_date = req.get("end_date", "2024-08-01T23")
        address = req.get("address", "CN Tower")
        years_back = req.get("years_back", 10)
        
        task_id = str(uuid.uuid4())
        progress_dict[task_id] = {
            "progress": 0,
            "total": years_back,
            "status": "running",
            "result": None
        }
        
        threading.Thread(target=collect_training_data_worker, 
                        args=(task_id, start_date, end_date, address, years_back)).start()
        
        return jsonify({"task_id": task_id, "message": "Data collection started"})

def collect_training_data_worker(task_id, start_date, end_date, address, years_back):
    try:
        progress_dict[task_id]["status"] = "collecting data"
        df = weather_collector.collect_weather_data(start_date, end_date, address, years_back)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"weather_training_data_{timestamp}.csv"
        df.to_csv(csv_filename, index=False)
        
        progress_dict[task_id]["progress"] = years_back
        progress_dict[task_id]["status"] = "done"
        progress_dict[task_id]["result"] = {
            "csv_file": csv_filename,
            "data_points": len(df),
            "columns": df.columns.tolist(),
            "weather_distribution": df['Weather'].value_counts().to_dict(),
        }
        
    except Exception as e:
        progress_dict[task_id]["status"] = "error"
        progress_dict[task_id]["error"] = str(e)

if AI_MODEL_AVAILABLE and weather_model:
    @app.route('/predict-weather', methods=['POST'])
    def predict_weather():
        req = request.get_json()
        
        weather_data = {
            'Rainf': req.get('rainfall', 0.0),
            'Humidity': req.get('humidity', 0.0),
            'Air Temperature': req.get('air_temperature', 0.0),
            'Wind_N': req.get('wind_north', 0.0),
            'Wind_E': req.get('wind_east', 0.0)
        }
        
        try:
            prediction = weather_model.predict_weather(weather_data)
            return jsonify({
                'success': True,
                'prediction': prediction,
                'input_data': weather_data
            })
        except Exception as e:
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

    @app.route('/model-info')
    def model_info():
        try:
            info = weather_model.get_model_info()
            return jsonify(info)
        except Exception as e:
            return jsonify({'error': f'Failed to get model info: {str(e)}'}), 500

if __name__ == "__main__":
    print("Starting Petrichor Backend...")
    print(f"Flask: {'✓' if FLASK_AVAILABLE else '✗'}")
    print(f"Requests: {'✓' if REQUESTS_AVAILABLE else '✗'}")
    print(f"Pandas/NumPy: {'✓' if PANDAS_AVAILABLE else '✗'}")
    print(f"AI Data Collector: {'✓' if AI_DATA_AVAILABLE else '✗'}")
    print(f"AI Model: {'✓' if AI_MODEL_AVAILABLE else '✗'}")
    print(f"Notebook Integration: {'✓' if NOTEBOOK_AVAILABLE else '✗'}")
    print("\\n" + "="*50)
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)