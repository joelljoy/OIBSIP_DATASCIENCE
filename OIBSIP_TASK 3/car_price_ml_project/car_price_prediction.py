#T3 CAR PRICEPREDICT
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("car_data.csv")

print("First 5 rows of dataset:")
print(data.head())

data = data.drop("Car_Name", axis=1)

data = pd.get_dummies(data, drop_first=True)

X = data.drop("Selling_Price", axis=1)
y = data["Selling_Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

print("\nModel Evaluation:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

print("\nEnter details to predict car price:")

year = int(input("Enter car year: "))
present_price = float(input("Enter present price (in lakhs): "))
kms = int(input("Enter driven kilometers: "))
owner = int(input("Enter owner (0 = first owner, 1 = second): "))

fuel_petrol = 1
fuel_diesel = 0
selling_type_individual = 0
transmission_manual = 1

input_data = np.array([[year, present_price, kms, owner,
                        fuel_diesel, fuel_petrol,
                        selling_type_individual,
                        transmission_manual]])

predicted_price = model.predict(input_data)

print("\nPredicted Selling Price (in lakhs):", round(predicted_price[0], 2))
