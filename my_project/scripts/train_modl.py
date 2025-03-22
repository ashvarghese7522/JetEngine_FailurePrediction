import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam


# Load the processed dataset
df = pd.read_csv("D:/ProjectML/Engine_Failure/my_project/data/processed/train_features.csv")

# Define features and target
feature_cols = [col for col in df.columns if col not in ['unit_number', 'time_in_cycles', 'RUL']]
target_col = 'RUL'

# Normalize features
scaler = MinMaxScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])

# Prepare sequences for LSTM
def create_sequences(data, target, seq_length=30):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(target[i+seq_length])
    return np.array(X), np.array(y)

# Convert data to sequences
sequence_length = 30
X, y = create_sequences(df[feature_cols].values, df[target_col].values, sequence_length)

# Split data into train and validation sets
train_size = int(0.8 * len(X))
X_train, X_val = X[:train_size], X[train_size:]
y_train, y_val = y[:train_size], y[train_size:]

# Build the LSTM model
model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(sequence_length, X.shape[2])),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25, activation='relu'),
    Dense(1)
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_val, y_val))

# Save the trained model
model.save("D:/ProjectML/Engine_Failure/my_project/models/lstm_rul_model.h5")

print("Model training complete and saved!")