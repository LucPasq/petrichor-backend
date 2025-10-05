import numpy as np
import pandas as pd
import os
import joblib
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings("ignore")

class WeatherPredictionModel:
    def __init__(self, model_path='Petrichor_Model.keras', scaler_path='feature_scaler.pkl'):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self.class_labels = ['No Rain', 'Very Light Rain', 'Light Rain', 'Moderate Rain', 
                           'Heavy Rain', 'Very Heavy Rain', 'Extreme Rain']
        
    def prepare_data(self, df):
        """Prepare data for training"""
        # Separate features and target
        feature_columns = ['Rainf', 'Humidity', 'Air Temperature', 'Wind_N', 'Wind_E']
        X = df[feature_columns].copy()
        y = df['Weather'].copy()
        
        # Handle missing values
        X = X.fillna(0)
        
        return X, y
    
    def train_model(self, df, test_size=0.2, epochs=100):
        """Train the GRU model"""
        print("Preparing data for training...")
        X, y = self.prepare_data(df)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        self.scaler = RobustScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Save scaler
        joblib.dump(self.scaler, self.scaler_path)
        
        # One-hot encode labels
        all_classes = pd.Series(list(y_train) + list(y_test))
        y_train_encoded = pd.get_dummies(y_train).reindex(columns=all_classes.unique(), fill_value=0)
        y_test_encoded = pd.get_dummies(y_test).reindex(columns=all_classes.unique(), fill_value=0)
        
        y_train_encoded = y_train_encoded.astype(np.float32)
        y_test_encoded = y_test_encoded.astype(np.float32)
        
        # Reshape for GRU (add timestep dimension)
        X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
        X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
        
        print("Building model...")
        # Build model
        timesteps = 1
        features = X_train_reshaped.shape[2]
        
        self.model = Sequential([
            GRU(256, return_sequences=True, input_shape=(timesteps, features)),
            GRU(128, return_sequences=True),
            GRU(64, return_sequences=True),
            GRU(64),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(y_train_encoded.shape[1], activation='softmax')
        ])
        
        # Compile model
        optimizer = Adam(learning_rate=1e-3)
        self.model.compile(optimizer=optimizer,
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
        
        print("Training model...")
        # Set up callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=0)
        ]
        
        # Train model
        history = self.model.fit(
            X_train_reshaped, y_train_encoded,
            validation_data=(X_test_reshaped, y_test_encoded),
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate model
        print("Evaluating model...")
        y_pred_probs = self.model.predict(X_test_reshaped)
        y_pred_classes = np.argmax(y_pred_probs, axis=1)
        y_true_classes = np.argmax(y_test_encoded.values, axis=1)
        
        # Map to class labels
        available_classes = y_test_encoded.columns.tolist()
        y_true_labels = [available_classes[i] for i in y_true_classes]
        y_pred_labels = [available_classes[i] for i in y_pred_classes]
        
        # Print evaluation metrics
        cm = confusion_matrix(y_true_labels, y_pred_labels)
        report = classification_report(y_true_labels, y_pred_labels)
        
        print("Confusion Matrix:")
        print(cm)
        print("\\nClassification Report:")
        print(report)
        
        # Save model
        self.model.save(self.model_path)
        print(f"Model saved to {self.model_path}")
        
        return history
    
    def load_model(self):
        """Load trained model and scaler"""
        if os.path.exists(self.model_path):
            self.model = load_model(self.model_path)
            print(f"Model loaded from {self.model_path}")
        else:
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
        if os.path.exists(self.scaler_path):
            self.scaler = joblib.load(self.scaler_path)
            print(f"Scaler loaded from {self.scaler_path}")
        else:
            raise FileNotFoundError(f"Scaler file not found: {self.scaler_path}")
    
    def predict_weather(self, weather_data):
        """
        Predict weather classification
        
        Args:
            weather_data: dict with keys ['Rainf', 'Humidity', 'Air Temperature', 'Wind_N', 'Wind_E']
                         or DataFrame with these columns
        
        Returns:
            dict: prediction results with probabilities
        """
        if self.model is None or self.scaler is None:
            self.load_model()
        
        # Convert to DataFrame if dict
        if isinstance(weather_data, dict):
            df = pd.DataFrame([weather_data])
        else:
            df = weather_data.copy()
        
        # Ensure all required columns are present
        required_columns = ['Rainf', 'Humidity', 'Air Temperature', 'Wind_N', 'Wind_E']
        for col in required_columns:
            if col not in df.columns:
                df[col] = 0.0
        
        # Select and scale features
        X = df[required_columns].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        # Reshape for GRU
        X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
        
        # Make prediction
        y_pred_probs = self.model.predict(X_reshaped, verbose=0)
        y_pred_class_idx = np.argmax(y_pred_probs, axis=1)
        
        # Get available classes from model output shape
        available_classes = self.class_labels[:y_pred_probs.shape[1]]
        
        results = []
        for i in range(len(y_pred_class_idx)):
            predicted_class = available_classes[y_pred_class_idx[i]]
            probabilities = {
                available_classes[j]: float(y_pred_probs[i][j]) 
                for j in range(len(available_classes))
            }
            
            results.append({
                'predicted_weather': predicted_class,
                'confidence': float(np.max(y_pred_probs[i])),
                'probabilities': probabilities
            })
        
        return results[0] if len(results) == 1 else results
    
    def get_model_info(self):
        """Get information about the loaded model"""
        if self.model is None:
            return {"error": "No model loaded"}
        
        return {
            "model_path": self.model_path,
            "input_shape": str(self.model.input_shape),
            "output_shape": str(self.model.output_shape),
            "total_params": self.model.count_params(),
            "available_classes": self.class_labels
        }