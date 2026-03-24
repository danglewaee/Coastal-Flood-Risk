import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Load the data
plt.style.use('seaborn-v0_8-whitegrid')
df = pd.read_csv('Main/global_mean_sea_level_1993-2024.csv')
# Plot the data
df.head(15)
# Add labels and title
df.describe()
df.shape
df.info()
year = df['YearPlusFraction']
weighted_obsv = df['NumberOfWeightedObservations']
plt.plot(year,weighted_obsv)
plt.xlabel('Year')
plt.ylabel('NumberOfWeightedObservations')
plt.show()

# simply do scatterplot(columnName) if you want to review the many other data
def scatterplot(col):
    plt.scatter(df['YearPlusFraction'], df[col], alpha=0.5)
    plt.xlabel('Year')
    plt.ylabel(col)
    plt.show()
scatterplot('SmoothedGMSLWithGIASigremoved')

# Predict using a simple linear regression model
from sklearn.linear_model import LinearRegression
X = df['YearPlusFraction'].values.reshape(-1, 1)
y = df['SmoothedGMSLWithGIASigremoved'].values.reshape(-1, 1)
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
last_year = int(df['YearPlusFraction'].max())
future_years = np.array(range(last_year + 1, last_year + 51)).reshape(-1, 1)
future_predictions = model.predict(future_years)

# Plot
plt.figure(figsize=(12, 6))
plt.scatter(df['YearPlusFraction'], df['SmoothedGMSLWithGIASigremoved'], color='blue', alpha=0.5, label='Historical Data')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.plot(future_years, future_predictions, color='green', linestyle='--', label='Future Predictions')
plt.scatter(future_years, future_predictions, color='green', alpha=0.5)
plt.xlabel('Year')
plt.ylabel('Sea Level (mm)')
plt.title('Sea Level Rise Prediction')
plt.legend()
plt.grid(True)

std_dev = df['StdDevGMSLWithGIA'].values.mean()
plt.fill_between(future_years.flatten(),
                 (future_predictions - 2*std_dev).flatten(),
                 (future_predictions + 2*std_dev).flatten(),
                 color='green', alpha=0.1, label='95% Confidence Interval')
plt.show()
print("\nPredicted sea levels for the next 10 years:")
for year, prediction in zip(future_years.flatten(), future_predictions.flatten()):
    print(f"Year {int(year)}: {prediction:.2f} mm")

