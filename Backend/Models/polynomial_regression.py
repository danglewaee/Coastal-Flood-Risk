import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('Main/global_mean_sea_level_1993-2024.csv')
year = df['YearPlusFraction'].values.reshape(-1, 1)
sea_level = df['SmoothedGMSLWithGIASigremoved'].values

#fit a cubic polynomial but force positive long-term slope
model = make_pipeline(
    PolynomialFeatures(degree = 3),
    LinearRegression(fit_intercept = False)
)
#scale data
scaler = StandardScaler()
year = scaler.fit_transform(year)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(year, sea_level, test_size=0.2, random_state=42)

# Polynomial Regression
poly = PolynomialFeatures(degree=3)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)

# Fit the model
model = LinearRegression()
model.fit(X_poly_train, y_train)

# Predict
last_year = df['YearPlusFraction'].max()
y_pred = model.predict(X_poly_test)
future_years = np.array(range(int(year[-1])+1, int(year[-1])+51)).reshape(-1, 1)
future_years_scaled = scaler.transform(future_years)
future_years_poly = poly.transform(future_years_scaled)
future_predictions = model.predict(future_years_poly)

#plot results
plt.figure(figsize=(12, 6))

#historical data
plt.scatter(year, sea_level, color='blue', s = 5, alpha=0.5, label='Historical Data')

#regression curve
sorted_idx = np.argsort(year.flatten())  # Get sorted indices
plt.plot(X_train, y_train, 'o', label='Historical Data', color='blue')
plt.plot(X_train, model.predict(X_poly_train), color='red', label='Polynomial Regression D3')

#future predictions
plt.plot(future_years, future_predictions, color='green', linestyle='--', label='Future Predictions')
plt.scatter(future_years, future_predictions, color='green', s=5, alpha=0.5)

plt.xlabel('Year')
plt.ylabel('Sea Level (mm)')
plt.title('Sea Level Rise Prediction')
plt.legend()
plt.grid(True)
plt.show()

# Print predicted sea levels for the next 50 years
print("\nPredicted sea levels for the next 50 years:")
for year, prediction in zip(future_years.flatten(), future_predictions.flatten()):
    print(f"Year {int(year)}: {prediction:.2f} mm")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
print(f"R2 Score: {r2_score(y_test, y_pred)}")
print(X_poly_train.shape)  # Should show (n_samples, degree + 1)
