# SALES PREDICTION T5
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("Advertising.csv")

print("First 5 rows of dataset:")
print(data.head())

data = data.drop(columns=["Unnamed: 0"])

X = data[["TV", "Radio", "Newspaper"]]
y = data["Sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("Mean Squared Error:", mse)
print("R2 Score:", r2)

tv = float(input("\nEnter TV Advertising Spend: "))
radio = float(input("Enter Radio Advertising Spend: "))
newspaper = float(input("Enter Newspaper Advertising Spend: "))

prediction = model.predict([[tv, radio, newspaper]])
print("\nPredicted Sales:", prediction[0])
