import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# Load the data
df = pd.read_csv('Main/global_mean_sea_level_1993-2024.csv')

# Plot the data
plt.style.use('seaborn-v0_8-whitegrid')
plt.plot(df['YearPlusFraction'], df['SmoothedGMSLWithGIASigremoved'])
plt.xlabel('Year')
plt.ylabel('Sea Level (mm)')
plt.show()

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df['SmoothedGMSLWithGIASigremoved'].values.reshape(-1, 1))

# Sequence generation
time_steps = 30
def create_sequences(data, time_steps=30):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, time_steps=30)

# Train-test split
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Define Transformer model
def transformer_model(input_shape):
    inputs = Input(shape=input_shape)
    x = inputs

    # Transformer Encoder Layer
    x = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    x = LayerNormalization()(x)
    x = Dropout(0.2)(x)
    
    # Fully connected layers
    x = Dense(50, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(25, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    # Output layer
    outputs = Dense(1)(x)

    model = tf.keras.models.Model(inputs, outputs)
    return model

# Create the model
model = transformer_model((time_steps, 1))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping],
    verbose=1
)

# Make predictions
y_pred = model.predict(X_test)

# Inverse scaling
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_actual = scaler.inverse_transform(y_pred.reshape(-1, 1))

# Plot predictions vs actual
plt.figure(figsize=(12, 6))
plt.plot(y_test_actual, label='Actual', color='blue')
plt.plot(y_pred_actual, label='Predicted', color='red', linestyle='--')
plt.xlabel('Time Step (Test Data Index)')
plt.ylabel('Sea Level (mm)')
plt.title('Transformer Prediction vs Actual')
plt.legend()
plt.show()

# Forecast future values
current_sequence = scaled_data[-time_steps:].reshape(1, time_steps, 1)
future_predictions = []

for i in range(50):
    next_pred = model.predict(current_sequence)
    print("Next prediction shape:", next_pred.shape)  # In kích thước của next_pred để kiểm tra
    
    future_predictions.append(next_pred[0, 0])
    
    # Kiểm tra xem kích thước có phải là (1, 30) hay không và reshape cho phù hợp
    if next_pred.shape[1] == 1:  # Nếu next_pred có shape (1, 1), reshape như trước
        next_pred_reshaped = next_pred.reshape(1, 1, 1)
    else:  # Nếu không, bạn sẽ cần xử lý phù hợp theo chiều của next_pred
        next_pred_reshaped = next_pred[:, -1, :].reshape(1, 1, 1)
    
    # Cập nhật chuỗi đầu vào
    current_sequence = np.concatenate((current_sequence[:, 1:, :], next_pred_reshaped), axis=1)

# Inverse scaling cho các dự đoán trong tương lai
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Tạo các năm trong tương lai
last_year = df['YearPlusFraction'].max()
future_years = np.linspace(last_year + 1, last_year + 51, num=50)

# Vẽ đồ thị kết quả
plt.figure(figsize=(12, 6))
plt.plot(df['YearPlusFraction'], df['SmoothedGMSLWithGIASigremoved'], label='Historical Data', color='blue')
plt.plot(future_years, future_predictions, label='Transformer Future Predictions', color='green', linestyle='--')
plt.xlabel('Year')
plt.ylabel('Sea Level (mm)')
plt.title('Transformer Future Predictions: Next 50 years')
plt.legend()
plt.show()
