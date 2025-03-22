import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1️⃣ Load the trained LSTM model
model = load_model("lstm_rul_model.h5")
print("✅ Model loaded successfully!")

# 2️⃣ Load test data (Replace with actual test data)
# Assuming X_test is already preprocessed (same steps as training)
# Replace with actual test set
X_test = np.load("X_test.npy")  # Example: Load preprocessed test data
y_test = np.load("y_test.npy")  # Example: Load actual RUL values

# 3️⃣ Make predictions
predicted_rul = model.predict(X_test)

# If you used normalization during training, apply inverse transformation
# Example:
# predicted_rul = scaler.inverse_transform(predicted_rul)
# y_test = scaler.inverse_transform(y_test)

# 4️⃣ Evaluate model performance
mae = mean_absolute_error(y_test, predicted_rul)
rmse = np.sqrt(mean_squared_error(y_test, predicted_rul))
r2 = r2_score(y_test, predicted_rul)

print(f"📊 Model Performance:")
print(f"🔹 Mean Absolute Error (MAE): {mae:.4f}")
print(f"🔹 Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"🔹 R² Score: {r2:.4f}")

# 5️⃣ Visualize Predictions vs Actual RUL
plt.figure(figsize=(10, 5))
plt.plot(y_test[:100], label="Actual RUL", marker='o', linestyle='dashed')
plt.plot(predicted_rul[:100], label="Predicted RUL", marker='s', linestyle='solid')
plt.xlabel("Sample Index")
plt.ylabel("Remaining Useful Life (RUL)")
plt.title("🔍 RUL Prediction: Actual vs Predicted")
plt.legend()
plt.show()
