import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error 
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Load the data
df = pd.read_csv('Main/global_mean_sea_level_1993-2024.csv')
# Plot the data
sea_level = df['SmoothedGMSLWithGIASigremoved'].values
time_index = df['YearPlusFraction'].values

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df['SmoothedGMSLWithGIASigremoved'].values.reshape(-1, 1))
# scaler = MinMaxScaler(feature_range=(0,1))
# scaled_data = scaler.fit_transform(df['SmoothedGMSLWithGIASigremoved'].values.reshape(-1, 1))
df.head(15)
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
#Plot
plt.style.use('seaborn-v0_8-whitegrid')
plt.plot(df['YearPlusFraction'], df['SmoothedGMSLWithGIASigremoved'])
plt.xlabel('Year')
plt.ylabel('Sea Level (mm)')
plt.show()

#sequences for LSTM
time_steps = 60            
def create_sequences(data, time_steps=360):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, time_steps)
# Split into train/test
train_size = int(0.8*len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# simply do scatterplot(columnName) if you want to review the many other data
def scatterplot(col):
    plt.scatter(df['YearPlusFraction'], df[col], alpha=0.5)
    plt.xlabel('Year')
    plt.ylabel(col)
    plt.show()
scatterplot('SmoothedGMSLWithGIASigremoved')

#LSTM model
model = Sequential()
model.add(LSTM(units = 150, activation = 'tanh', input_shape = (time_steps, 1), return_sequences = True))
model.add(Dropout(0.1))
model.add(LSTM(units = 100, activation = 'tanh', return_sequences = False))
model.add(Dropout(0.15))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(25, activation = 'relu'))
model.add(Dense(1))
optimizer = Adam(learning_rate = 0.00005)
model.compile(optimizer = optimizer, loss='mse')

#Train
history = model.fit(
    X_train, y_train,
    epochs = 400,
    batch_size = 32,
    validation_data = (X_test, y_test),
    callbacks = [early_stopping],
    verbose = 1
)
# Trainign & Validation loss
plt.figure(figsize=(10,5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss (MSE)')
plt.xlabel('Epochs')
plt.legend(loc = 'upper right')
plt.show()

#predict on test data
y_pred = model.predict(X_test)
y_pred_actual = scaler.inverse_transform(y_pred)
print("First 10 predicted values (actual scale):", y_pred_actual[:10].flatten())
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
print("First 10 actual values:", y_test_actual[:10].flatten())

# Calculate evaluation metrics
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
mae = mean_absolute_error(y_test_actual, y_pred_actual)
test_time = time_index[time_steps+train_size:]

# Calculate points per year
time_diffs = np.diff(time_index)
avg_time_diff = np.mean(time_diffs)
points_per_year = 1.0 / avg_time_diff

# Add labels and title
plt.figure(figsize=(12,6))
plt.plot(df['YearPlusFraction'][time_steps+train_size:], y_test_actual, label = 'Actual', color = 'blue')
plt.plot(df['YearPlusFraction'][time_steps+train_size:], y_pred_actual, label = 'Predicted', color = 'red', linestyle='--')
plt.xlabel('Year')
plt.ylabel('Sea LeVeL (mm)')
plt.title('LSTM Prediction vs Actual')
plt.legend()
plt.show()

#start with the last time steps data points
future_years_predict = 30
future_steps = int(np.ceil(future_years_predict * points_per_year))
current_seq = scaled_data[-time_steps:].reshape(1, time_steps, 1)
future_preds = []
for i in range(future_steps):
    pred = model.predict(current_seq, verbose = 0)[0][0]
    future_preds.append(pred)
    current_seq = np.append(current_seq[:, 1:, :], [[[pred]]], axis=1)
last_year = df['YearPlusFraction'].iloc[-1]

#generate future years
future_preds_actual = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))
last_historical_time = time_index[-1]
future_years = last_historical_time + np.arange(1, future_steps + 1) * avg_time_diff

plt.figure(figsize=(14,7))
plt.plot(df['YearPlusFraction'], df['SmoothedGMSLWithGIASigremoved'], label = 'Historical Data', color = 'blue')
plt.plot(future_years, future_preds_actual, label='LSTM Future Predictions', color = 'green', linestyle='--')
plt.xlabel('Year')
plt.ylabel('Sea Level (mm)')
plt.title('LSTM Future Predictions: Next 50 years')
plt.legend()
plt.show()

# Vẽ dữ liệu 15 năm gần nhất
recent_years = df[df['YearPlusFraction'] > (df['YearPlusFraction'].max() - 15)]
plt.figure(figsize=(10,5))
plt.plot(recent_years['YearPlusFraction'], recent_years['SmoothedGMSLWithGIASigremoved'], marker='o')
plt.title('Sea Level - 15 năm gần nhất')
plt.xlabel('Year')
plt.ylabel('Sea Level (mm)')
plt.grid(True)
plt.show()

