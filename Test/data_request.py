import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

df = pd.read_csv("Test/8443970_meantrend.csv", delimiter="\t")  # Load your CSV with the full path

# Print the columns to debug
print(df.columns)

df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(day=1))  # Create a Date column
df.set_index('Date', inplace=True)  # Set Date as index
df_cleaned = df[['Monthly_MSL']].dropna()
df_cleaned = df_cleaned.copy()  # Avoid SettingWithCopyWarning
scaler = MinMaxScaler()
df_cleaned['Normalized_MSL'] = scaler.fit_transform(df_cleaned[['Monthly_MSL']])

plt.figure(figsize=(12, 6))
plt.plot(df_cleaned.index, df_cleaned['Normalized_MSL'], label="Sea Level Trend")
plt.xlabel("Year")
plt.ylabel("Normalized MSL")
plt.title("Normalized Monthly Mean Sea Level Over Time")
plt.legend()
plt.show()