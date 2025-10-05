# Petrichor Backend API Documentation

The Petrichor backend now includes comprehensive AI functionality integrated from the Jupyter notebook. Here's what's been added:

## New Endpoints

### 1. Data Collection for AI Training

**POST `/collect-training-data`**

Collect comprehensive weather data for AI model training.

**Request Body:**
```json
{
    "start_date": "2024-08-01T00",
    "end_date": "2024-08-01T23", 
    "address": "CN Tower",
    "years_back": 10
}
```

**Response:**
```json
{
    "task_id": "uuid-string",
    "message": "Data collection started"
}
```

### 2. Train AI Model

**POST `/train-ai-model`**

Train the weather prediction AI model using collected data.

**Request Body:**
```json
{
    "csv_file": "weather_training_data_20241005_123456.csv",
    "epochs": 100
}
```

**Response:**
```json
{
    "task_id": "uuid-string",
    "message": "Model training started"
}
```

### 3. Weather Prediction

**POST `/predict-weather`**

Predict weather classification using the trained AI model.

**Request Body:**
```json
{
    "rainfall": 0.5,
    "humidity": 0.7,
    "air_temperature": 25.0,
    "wind_north": 2.0,
    "wind_east": 1.5
}
```

**Response:**
```json
{
    "success": true,
    "prediction": {
        "predicted_weather": "Light Rain",
        "confidence": 0.85,
        "probabilities": {
            "No Rain": 0.05,
            "Very Light Rain": 0.10,
            "Light Rain": 0.85,
            "Moderate Rain": 0.00,
            "Heavy Rain": 0.00,
            "Very Heavy Rain": 0.00,
            "Extreme Rain": 0.00
        }
    },
    "input_data": {
        "Rainf": 0.5,
        "Humidity": 0.7,
        "Air Temperature": 25.0,
        "Wind_N": 2.0,
        "Wind_E": 1.5
    }
}
```

### 4. Model Information

**GET `/model-info`**

Get information about the current AI model.

**Response:**
```json
{
    "model_path": "Petrichor_Model.keras",
    "input_shape": "(None, 1, 5)",
    "output_shape": "(None, 7)",
    "total_params": 156789,
    "available_classes": [
        "No Rain",
        "Very Light Rain", 
        "Light Rain",
        "Moderate Rain",
        "Heavy Rain",
        "Very Heavy Rain",
        "Extreme Rain"
    ]
}
```

### 5. Enhanced Rainfall Analysis

**POST `/enhanced-rainfall-analysis`**

Enhanced version of the original rainfall analysis that includes AI predictions.

**Request Body:**
```json
{
    "date": "2024-10-05",
    "lat": 43.6426,
    "lon": -79.3871,
    "location_name": "Toronto"
}
```

**Response:** Same as original rainfall-summary but includes `ai_predictions` array with weather classifications for each historical data point.

## Progress Tracking

All long-running operations (data collection, model training, analysis) return a `task_id`. Use the existing `/progress/<task_id>` endpoint to track progress:

**GET `/progress/<task_id>`**

**Response:**
```json
{
    "progress": 50,
    "total": 100,
    "status": "running",
    "result": null
}
```

When complete:
```json
{
    "progress": 100,
    "total": 100,
    "status": "done",
    "result": {
        // Results specific to the operation
    }
}
```

## Weather Classification System

The AI model classifies weather into 7 categories based on rainfall intensity:

1. **No Rain**: 0.0 mm
2. **Very Light Rain**: 0.0 - 0.25 mm/hour
3. **Light Rain**: 0.25 - 1.0 mm/hour
4. **Moderate Rain**: 1.0 - 4.0 mm/hour
5. **Heavy Rain**: 4.0 - 16.0 mm/hour
6. **Very Heavy Rain**: 16.0 - 50.0 mm/hour
7. **Extreme Rain**: > 50.0 mm/hour

## Files Generated

The system creates several files during operation:

- `weather_training_data_YYYYMMDD_HHMMSS.csv`: Training data
- `Petrichor_Model.keras`: Trained AI model
- `feature_scaler.pkl`: Scaler for input features
- `rainfall_analysis_YYYYMMDD_HHMMSS.ipynb`: Analysis notebooks
- `confusion_matrix.png`: Model evaluation charts

## Usage Workflow

1. **Collect Training Data**: Use `/collect-training-data` to gather historical weather data
2. **Train Model**: Use `/train-ai-model` with the generated CSV file
3. **Make Predictions**: Use `/predict-weather` for weather classification
4. **Enhanced Analysis**: Use `/enhanced-rainfall-analysis` for comprehensive reports

## Error Handling

All endpoints return appropriate HTTP status codes:
- 200: Success
- 400: Bad Request (missing parameters)
- 404: Not Found (invalid task_id)
- 500: Internal Server Error

Error responses include an `error` field with a descriptive message.

## Dependencies

The integration requires additional Python packages:
- tensorflow: Deep learning framework
- scikit-learn: Machine learning utilities
- geopy: Geocoding functionality
- joblib: Model serialization
- pandas, numpy: Data manipulation
- matplotlib, seaborn: Visualization

Install all dependencies with:
```bash
pip install -r requirements.txt
```