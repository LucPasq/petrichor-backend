import re
import io
import csv
import threading
import time
import uuid
import json
import os
from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
from datetime import datetime, timedelta
import requests
from notebook_integration import NotebookExecutor
from weather_data_collector import WeatherDataCollector
from weather_prediction_model import WeatherPredictionModel

app = Flask(__name__)
CORS(app, origins="*", methods=["GET", "POST", "OPTIONS"], allow_headers=["Content-Type", "Authorization"])

progress_dict = {}  # {task_id: {progress, total, status, result}}
RESULTS_HISTORY_FILE = "results_history.json"

# Initialize components
notebook_executor = NotebookExecutor()
weather_collector = WeatherDataCollector()
weather_model = WeatherPredictionModel()

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

@app.route('/health')
def health_check():
    return jsonify({"status": "healthy", "message": "Backend is running"})

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
                # Add timeout to prevent hanging requests
                api_resp = requests.get(api_url, timeout=30)
                api_resp.raise_for_status()  # Raise an exception for bad status codes
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
            
            # Update progress
            progress_dict[task_id]["progress"] = i
            # time.sleep(0.1)  # Removed artificial delay for faster processing

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
        
    except Exception as e:
        # Handle any unexpected errors
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

@app.route('/generate-notebook', methods=['POST'])
def generate_notebook():
    """Generate and execute a Jupyter notebook for rainfall analysis"""
    req = request.get_json()
    task_id = req.get("task_id")
    location_name = req.get("location_name", "Unknown Location")
    
    if not task_id:
        return jsonify({'error': 'Missing task_id'}), 400
    
    # Get the rainfall data from progress_dict
    task_data = progress_dict.get(task_id)
    if not task_data or task_data.get("status") != "done":
        return jsonify({'error': 'Task not found or not completed'}), 404
    
    rainfall_data = task_data.get("result")
    if not rainfall_data:
        return jsonify({'error': 'No rainfall data available'}), 404
    
    try:
        # Create notebook
        notebook = notebook_executor.create_rainfall_analysis_notebook(
            rainfall_data, location_name
        )
        
        # Execute notebook
        executed_notebook, error = notebook_executor.execute_notebook(notebook)
        if error:
            return jsonify({'error': f'Notebook execution failed: {error}'}), 500
        
        # Save notebook
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"rainfall_analysis_{timestamp}.ipynb"
        notebook_executor.save_notebook(executed_notebook, filename)
        
        # Convert to HTML for preview
        html_content = notebook_executor.notebook_to_html(executed_notebook)
        
        return jsonify({
            'success': True,
            'notebook_file': filename,
            'preview_html': html_content,
            'message': 'Notebook generated and executed successfully'
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to generate notebook: {str(e)}'}), 500

@app.route('/download-notebook/<filename>')
def download_notebook(filename):
    """Download a generated notebook file"""
    try:
        return send_from_directory('.', filename, as_attachment=True)
    except FileNotFoundError:
        return jsonify({'error': 'Notebook file not found'}), 404

@app.route('/notebook-preview/<task_id>')
def notebook_preview(task_id):
    """Get HTML preview of the notebook analysis"""
    # Get the rainfall data from progress_dict
    task_data = progress_dict.get(task_id)
    if not task_data or task_data.get("status") != "done":
        return jsonify({'error': 'Task not found or not completed'}), 404
    
    rainfall_data = task_data.get("result")
    if not rainfall_data:
        return jsonify({'error': 'No rainfall data available'}), 404
    
    try:
        # Create and execute notebook
        notebook = notebook_executor.create_rainfall_analysis_notebook(rainfall_data)
        executed_notebook, error = notebook_executor.execute_notebook(notebook)
        
        if error:
            return f"<html><body><h1>Error</h1><p>{error}</p></body></html>", 500
        
        # Convert to HTML
        html_content = notebook_executor.notebook_to_html(executed_notebook)
        return Response(html_content, mimetype='text/html')
        
    except Exception as e:
        return f"<html><body><h1>Error</h1><p>Failed to generate preview: {str(e)}</p></body></html>", 500

@app.route('/collect-training-data', methods=['POST'])
def collect_training_data():
    """Collect comprehensive weather data for AI model training"""
    req = request.get_json()
    start_date = req.get("start_date", "2024-08-01T00")
    end_date = req.get("end_date", "2024-08-01T23")
    address = req.get("address", "CN Tower")
    years_back = req.get("years_back", 10)
    
    # Generate a unique task ID
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
    """Worker function to collect training data"""
    try:
        # Update progress
        progress_dict[task_id]["status"] = "collecting data"
        
        # Collect data
        df = weather_collector.collect_weather_data(start_date, end_date, address, years_back)
        
        # Save to CSV
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
            "date_range": {
                "start": df['time'].min().isoformat() if not df.empty else None,
                "end": df['time'].max().isoformat() if not df.empty else None
            }
        }
        
    except Exception as e:
        progress_dict[task_id]["status"] = "error"
        progress_dict[task_id]["error"] = str(e)
        print(f"Training data collection error for task {task_id}: {str(e)}")

@app.route('/train-ai-model', methods=['POST'])
def train_ai_model():
    """Train the weather prediction AI model"""
    req = request.get_json()
    csv_file = req.get("csv_file")
    epochs = req.get("epochs", 100)
    
    if not csv_file or not os.path.exists(csv_file):
        return jsonify({'error': 'CSV file not found'}), 400
    
    # Generate a unique task ID
    task_id = str(uuid.uuid4())
    progress_dict[task_id] = {
        "progress": 0,
        "total": epochs,
        "status": "running",
        "result": None
    }
    
    threading.Thread(target=train_model_worker, 
                    args=(task_id, csv_file, epochs)).start()
    
    return jsonify({"task_id": task_id, "message": "Model training started"})

def train_model_worker(task_id, csv_file, epochs):
    """Worker function to train the AI model"""
    try:
        import pandas as pd
        
        progress_dict[task_id]["status"] = "loading data"
        
        # Load data
        df = pd.read_csv(csv_file)
        
        progress_dict[task_id]["status"] = "training model"
        
        # Train model
        history = weather_model.train_model(df, epochs=epochs)
        
        progress_dict[task_id]["progress"] = epochs
        progress_dict[task_id]["status"] = "done"
        progress_dict[task_id]["result"] = {
            "model_file": weather_model.model_path,
            "scaler_file": weather_model.scaler_path,
            "training_samples": len(df),
            "final_accuracy": float(history.history['accuracy'][-1]),
            "final_val_accuracy": float(history.history['val_accuracy'][-1]),
            "epochs_completed": len(history.history['accuracy'])
        }
        
    except Exception as e:
        progress_dict[task_id]["status"] = "error"
        progress_dict[task_id]["error"] = str(e)
        print(f"Model training error for task {task_id}: {str(e)}")

@app.route('/predict-weather', methods=['POST'])
def predict_weather():
    """Predict weather using the trained AI model"""
    req = request.get_json()
    
    # Extract weather parameters
    weather_data = {
        'Rainf': req.get('rainfall', 0.0),
        'Humidity': req.get('humidity', 0.0),
        'Air Temperature': req.get('air_temperature', 0.0),
        'Wind_N': req.get('wind_north', 0.0),
        'Wind_E': req.get('wind_east', 0.0)
    }
    
    try:
        # Make prediction
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
    """Get information about the current AI model"""
    try:
        info = weather_model.get_model_info()
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': f'Failed to get model info: {str(e)}'}), 500

@app.route('/enhanced-rainfall-analysis', methods=['POST'])
def enhanced_rainfall_analysis():
    """Enhanced rainfall analysis with AI predictions"""
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
    
    threading.Thread(target=enhanced_rainfall_worker, 
                    args=(task_id, dt, lat, lon, location_name)).start()
    
    return jsonify({"task_id": task_id})

def enhanced_rainfall_worker(task_id, dt, lat, lon, location_name):
    """Enhanced worker with AI predictions"""
    try:
        # Use the original rainfall analysis
        month = dt.month
        day = dt.day
        year = dt.year

        daily_results = []
        ai_predictions = []
        
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
                    
                    # Try to get AI prediction if model is available
                    try:
                        # Create sample weather data for prediction
                        sample_data = {
                            'Rainf': total,
                            'Humidity': 0.5,  # Default values - in real scenario, collect these
                            'Air Temperature': 20.0,
                            'Wind_N': 0.0,
                            'Wind_E': 0.0
                        }
                        
                        prediction = weather_model.predict_weather(sample_data)
                        ai_predictions.append({
                            "date": date,
                            "ai_prediction": prediction
                        })
                    except Exception as ai_error:
                        print(f"AI prediction error for {date}: {ai_error}")
                        ai_predictions.append({
                            "date": date,
                            "ai_prediction": {"error": "Model not available"}
                        })
                else:
                    daily_results.append({"date": f"{this_year}-{month:02d}-{day:02d}", "rainfall_mm": None})
                    
            except Exception as e:
                daily_results.append({"date": f"{this_year}-{month:02d}-{day:02d}", "rainfall_mm": None, "error": str(e)})
            
            progress_dict[task_id]["progress"] = i

        # Calculate statistics
        values = [entry["rainfall_mm"] for entry in daily_results if entry["rainfall_mm"] is not None]
        avg = round(sum(values)/len(values), 4) if values else 0.0
        category = classify_rainfall(avg)

        # Generate CSV
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
            "ai_predictions": ai_predictions,
            "csv": csv_data
        }
        
        progress_dict[task_id]["result"] = result_obj
        progress_dict[task_id]["status"] = "done"

        # Save to history
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
        print(f"Enhanced worker error for task {task_id}: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)