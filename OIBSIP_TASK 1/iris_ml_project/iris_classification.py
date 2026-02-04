# IRIS FLOWER CLASSIFICATION PROJECT
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv("Iris.csv")

print("First 5 rows of dataset:")
print(data.head())

data = data.drop("Id", axis=1)

X = data.drop("Species", axis=1)
y = data["Species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=200)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

new_flower = np.array([[5.1, 3.5, 1.4, 0.2]])
prediction = model.predict(new_flower)

print("\nPredicted Flower Type:", prediction[0])
