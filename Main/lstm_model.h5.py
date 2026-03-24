import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Load the data
df = pd.read_csv('Main/global_mean_sea_level_1993-2024.csv')
# Plot the data
sea_level = df['SmoothedGMSLWithGIASigremoved'].values.reshape(-1, 1)
time_index = df['YearPlusFraction'].values.reshape(-1, 1)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(sea_level)
df.head(15)
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
#Plot
plt.style.use('seaborn-v0_8-whitegrid')
plt.plot(df['YearPlusFraction'], df['SmoothedGMSLWithGIASigremoved'])
plt.xlabel('Year')
plt.ylabel('Sea Level (mm)')
plt.show()

# #Normalize the data
# scaler = MinMaxScaler(feature_range=(0,1))
# scaled_data = scaler.fit_transform(df['SmoothedGMSLWithGIASigremoved'].values.reshape(-1, 1))

#sequences for LSTM
time_steps = 240
def create_sequences(data, time_steps=240):
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
optimixer = Adam(learning_rate = 0.0005)
model.compile(optimizer = 'adam', loss='mse')

#Train
history = model.fit(
    X_train, y_train,
    epochs = 200,
    batch_size = 32,
    validation_data = (X_test, y_test),
    callbacks = [early_stopping],
    verbose = 1
)

#predict on test data
y_pred = model.predict(X_test)
y_pred_actual = scaler.inverse_transform(y_pred)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
import os
os.makedirs('Backend/Models/', exist_ok=True)
model.save('Backend/Models/lstm_model.h5')

#plot predictions vs actual
plt.figure(figsize=(12,6))
plt.plot(y_test_actual, label = 'Actual', color = 'blue')
plt.plot(y_pred_actual, label = 'Predicted', color = 'red', linestyle='--')
plt.xlabel('Time Step (Test Data Index)')
plt.ylabel('Sea Level (mm)')
plt.title('LSTM Prediction vs Actual')
plt.legend()
plt.show()

# Add labels and title
df.describe()
df.shape
df.info()
plt.figure(figsize=(12,6))
plt.plot(df['YearPlusFraction'][time_steps+train_size:], y_test_actual, label = 'Actual', color = 'blue')
plt.plot(df['YearPlusFraction'][time_steps+train_size:], y_pred_actual, label = 'Predicted', color = 'red', linestyle='--')
plt.xlabel('Year')
plt.ylabel('Sea LeVeL (mm)')
plt.title('LSTM Prediction vs Actual')
plt.legend()
plt.show()

#start with the last time steps data points
future_steps = 50
current_seq = scaled_data[-time_steps:].reshape(1, time_steps, 1)
future_preds = []
for _ in range(future_steps):
    pred = model.predict(current_seq)[0][0]
    future_preds.append(pred)
    current_seq = np.append(current_seq[:, 1:, :], [[[pred]]], axis=1)
future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))
last_year = df['YearPlusFraction'].iloc[-1]

#generate future years
future_years = np.linspace(last_year + 1, last_year + 51, num=50)
plt.figure(figsize=(12,6))
plt.plot(df['YearPlusFraction'], df['SmoothedGMSLWithGIASigremoved'], label = 'Historical Data', color = 'blue')
plt.plot(future_years, future_preds, label='LSTM Future Predictions', color = 'green', linestyle='--')
plt.xlabel('Year')
plt.ylabel('Sea Level (mm)')
plt.title('LSTM Future Predictions: Next 50 years')
plt.legend()
plt.show()



