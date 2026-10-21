# Predictive Analytics Using Historical Data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Create a sample historical dataset
# Replace this sample with real historical data when available.

data = {
    "Month": pd.date_range(
        start="2024-01-01",
        periods=24,
        freq="MS"
    ),
    "Sales": [
        120, 135, 128, 145, 160, 155,
        170, 180, 175, 190, 205, 215,
        210, 225, 235, 230, 250, 265,
        255, 275, 285, 280, 300, 315
    ]
}

df = pd.DataFrame(data)

# 2. Clean and preprocess the historical data
df["Month"] = pd.to_datetime(df["Month"], errors="coerce")
df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")

df.dropna(subset=["Month", "Sales"], inplace=True)
df = df.sort_values("Month").reset_index(drop=True)

print("Historical Data:")
print(df)

# 3. Prepare data for prediction
df["Time"] = np.arange(len(df))

X = df[["Time"]]
y = df["Sales"]

# 4. Split data into training and testing sets
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# 5. Train the regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Predict sales for the test period
y_pred = model.predict(X_test)

# 7. Evaluate model accuracy
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared Score: {r2:.2f}")

# 8. Predict sales for the next 6 months
future_time = np.arange(len(df), len(df) + 6)

future_sales = model.predict(
    pd.DataFrame({"Time": future_time})
)

future_dates = pd.date_range(
    start=df["Month"].max() + pd.offsets.MonthBegin(1),
    periods=6,
    freq="MS"
)

future_df = pd.DataFrame({
    "Month": future_dates,
    "Predicted Sales": np.round(future_sales, 2)
})

print("\nFuture Sales Predictions:")
print(future_df)

# 9. Visualize historical and predicted sales
plt.figure(figsize=(12, 6))

plt.plot(
    df["Month"],
    df["Sales"],
    marker="o",
    label="Historical Sales"
)

plt.plot(
    df["Month"].iloc[split_index:],
    y_pred,
    marker="x",
    linestyle="--",
    label="Test Predictions"
)

plt.plot(
    future_df["Month"],
    future_df["Predicted Sales"],
    marker="s",
    linestyle="--",
    label="Future Predictions"
)

plt.title("Predictive Analytics: Historical and Forecast Sales")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("sales_prediction.png", dpi=300)
plt.show()

# 10. Save results
df.to_csv("historical_sales.csv", index=False)
future_df.to_csv("future_sales_predictions.csv", index=False)

print("\nProject completed successfully!")
print("Files saved:")
print("- historical_sales.csv")
print("- future_sales_predictions.csv")
print("- sales_prediction.png")
