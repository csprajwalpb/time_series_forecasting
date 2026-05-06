import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import joblib
from app.utils.logger import get_logger

logger = get_logger(__name__)

class LSTMModel:
    def __init__(self, sequence_length: int = 30):
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = MinMaxScaler()
        self.last_sequence = None

    def create_sequences(self, data):
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[i + self.sequence_length])
        return np.array(X), np.array(y)

    def train(self, df: pd.DataFrame):
        logger.info("Training LSTM model...")
        sales_data = df['sales'].values.reshape(-1, 1)
        
        # Scale data
        scaled_data = self.scaler.fit_transform(sales_data)
        
        X, y = self.create_sequences(scaled_data)
        
        # Define model
        self.model = Sequential([
            LSTM(50, activation='relu', input_shape=(self.sequence_length, 1), return_sequences=True),
            Dropout(0.2),
            LSTM(50, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse')
        
        # Train model
        self.model.fit(X, y, epochs=20, batch_size=32, verbose=0)
        
        # Save last sequence for future prediction
        self.last_sequence = scaled_data[-self.sequence_length:]
        logger.info("LSTM training complete.")

    def predict(self, steps: int) -> np.ndarray:
        if self.model is None or self.last_sequence is None:
            raise ValueError("Model is not trained yet.")
        
        current_seq = self.last_sequence.copy()
        predictions_scaled = []
        
        for _ in range(steps):
            # Reshape for LSTM input [samples, time steps, features]
            curr_seq_reshaped = current_seq.reshape((1, self.sequence_length, 1))
            
            pred = self.model.predict(curr_seq_reshaped, verbose=0)[0]
            predictions_scaled.append(pred)
            
            # Update sequence: remove first element, append prediction
            current_seq = np.append(current_seq[1:], pred)
            
        predictions = self.scaler.inverse_transform(predictions_scaled)
        return predictions.flatten()

    def save(self, path: str):
        # Keras models save to their own format
        model_path = f"{path}_lstm.keras"
        scaler_path = f"{path}_scaler.pkl"
        seq_path = f"{path}_seq.pkl"
        
        self.model.save(model_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.last_sequence, seq_path)
        
        joblib.dump({
            "model_path": model_path,
            "scaler_path": scaler_path,
            "seq_path": seq_path,
            "sequence_length": self.sequence_length
        }, path)

    def load(self, path: str):
        data = joblib.load(path)
        self.sequence_length = data['sequence_length']
        self.scaler = joblib.load(data['scaler_path'])
        self.last_sequence = joblib.load(data['seq_path'])
        
        # Load keras model
        self.model = tf.keras.models.load_model(data['model_path'])
