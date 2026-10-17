# Predictive Analytics Using Historical Data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Create a sample historical dataset
data = {
    "Month": np.arange(1, 21),
    "Sales": [
        120, 135, 128, 150, 160,
        155, 175, 180, 172, 195,
        205, 198, 220, 230, 225,
        245, 250, 260, 255, 280
    ]
}

df = pd.DataFrame(data)

# Display historical data
print("Historical Sales Data:")
print(df)

# Prepare input and output
X = df[["Month"]]
y = df["Sales"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the prediction model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict sales
predictions = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nMean Absolute Error:", round(mae, 2))
print("R2 Score:", round(r2, 2))

# Predict sales for the next month
next_month = pd.DataFrame({"Month": [21]})
future_prediction = model.predict(next_month)

print("\nPredicted Sales for Month 21:",
      round(future_prediction[0], 2))

# Visualize historical data and predictions
plt.scatter(df["Month"], df["Sales"], label="Historical Sales")
plt.plot(
    df["Month"],
    model.predict(X),
    label="Regression Line"
)

plt.scatter(
    [21],
    future_prediction,
    color="red",
    label="Future Prediction"
)

plt.xlabel("Month")
plt.ylabel("Sales")
plt.title("Historical Sales and Future Prediction")
plt.legend()
plt.grid(True)
plt.show()
