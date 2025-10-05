import re
import io
import csv
import threading
import time
import uuid
import json
import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from datetime import datetime, timedelta
import requests

app = Flask(__name__)
CORS(app)

progress_dict = {}  # {task_id: {progress, total, status, result}}
RESULTS_HISTORY_FILE = "results_history.json"

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
    lines = raw_text.strip().split('\n')
    first_data_idx = None
    for idx, line in enumerate(lines):
        if re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", line):
            first_data_idx = idx
            break
    if first_data_idx is None:
        return None, 0.0
    date = lines[first_data_idx][:10]
    total = 0.0
    for line in lines[first_data_idx:]:
        parts = line.strip().split('\t')
        if len(parts) == 2 and re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", parts[0]):
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
    for i in range(1, 11):
        this_year = year - i
        api_url = build_api_url(this_year, month, day, lat, lon)
        try:
            api_resp = requests.get(api_url)
            raw = api_resp.text
            date, total = parse_daily_total(raw)
            if date:
                daily_results.append({"date": date, "rainfall_mm": round(total, 4)})
            else:
                daily_results.append({"date": f"{this_year}-{month:02d}-{day:02d}", "rainfall_mm": None})
        except Exception as e:
            daily_results.append({"date": f"{this_year}-{month:02d}-{day:02d}", "rainfall_mm": None, "error": str(e)})
        progress_dict[task_id]["progress"] = i
        time.sleep(0.1)  # Optional: simulate delay for demo

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

    # Save to history (with location name if available)
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
    return jsonify(resp)

@app.route('/results-history')
def results_history():
    return jsonify(get_results_history())

if __name__ == "__main__":
    app.run(debug=True)