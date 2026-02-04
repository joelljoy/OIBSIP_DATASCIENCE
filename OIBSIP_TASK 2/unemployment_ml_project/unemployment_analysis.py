#UNEMPLOYMENT ANALYSIS T2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("unemployment.csv")

print("First 5 rows of dataset:")
print(data.head())

data.columns = data.columns.str.strip()

data.rename(columns={
    "Estimated Unemployment Rate (%)": "Unemployment_Rate",
    "Region": "Region",
    "Date": "Date"
}, inplace=True)

data["Date"] = pd.to_datetime(data["Date"])

plt.figure(figsize=(10,5))
sns.lineplot(x="Date", y="Unemployment_Rate", data=data)
plt.title("Unemployment Rate Trend During COVID-19")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.grid(True)
plt.show()

avg_rate = data["Unemployment_Rate"].mean()
print("\nAverage Unemployment Rate:", round(avg_rate, 2), "%")
