
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Generate synthetic sales data
np.random.seed(42)
dates = pd.date_range(start="2023-01-01", periods=365, freq="D")
products = ["Strawberry Jelly", "Apple Jelly", "Orange Jelly", "Grape Jelly", "Lemon Jelly"]
prices = [5, 4, 6, 5, 4]

data = {
    "Order Date": np.random.choice(dates, size=1000),
    "Product": np.random.choice(products, size=1000),
    "Quantity": np.random.randint(1, 20, size=1000)
}

df = pd.DataFrame(data)
df["Unit Price"] = df["Product"].map(dict(zip(products, prices)))
df["Total Sales"] = df["Quantity"] * df["Unit Price"]

# Feature extraction
df["Day of Month"] = df["Order Date"].dt.day
df["Month"] = df["Order Date"].dt.month
df["Day of Week"] = df["Order Date"].dt.weekday

# Split data into training and testing sets
X = df[["Quantity", "Unit Price", "Day of Month", "Month", "Day of Week"]]
y = df["Total Sales"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict sales
y_pred = model.predict(X_test)

# Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R-Squared: {r2}")

# Plot Actual vs Predicted Sales
plt.scatter(y_test, y_pred, alpha=0.5, color="blue")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r', lw=2)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()
