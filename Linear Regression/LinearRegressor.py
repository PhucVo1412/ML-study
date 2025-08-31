import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class LinearRegressor:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = None
        self.b = None
        self.losses = []

    def fit(self, X, y):
        m, n = X.shape
        self.w = np.zeros(n)
        self.b = 0

        for _ in range(self.epochs):
            # Dự đoán y_hat
            y_pred = self.predict(X)

            # Tính đạo hàm (gradient)
            dw = (1/m) * np.dot(X.T, (y_pred - y))  # Corrected gradient for weights
            db = (1/m) * np.sum(y_pred - y)         # Gradient for bias

            # Cập nhật tham số
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

            # Tính loss và lưu lại
            loss = (1/m) * np.sum((y_pred - y) ** 2)  # Consistent with evaluate
            self.losses.append(loss)

    def predict(self, X):
        return np.dot(X, self.w) + self.b  # Added bias term

    def evaluate(self, X, y, method="MSE"):
        y_pred = self.predict(X)
        mse = np.mean((y_pred - y) ** 2)
        return mse

    def plot_regression_line(self, X, y, feature_idx=0, feature_name="Feature"):
        if X.shape[1] < feature_idx + 1:
            raise ValueError("Invalid feature index for plotting")
        
        # Select one feature for plotting
        X_single = X[:, feature_idx].reshape(-1, 1)
        
        plt.figure(figsize=(8, 5))
        plt.scatter(X_single, y, color="blue", label="Data")
        y_pred = self.predict(X)  # Predict using all features
        plt.plot(X_single, y_pred, color="red", label="Predictions", alpha=0.5)
        plt.xlabel(feature_name)
        plt.ylabel("y")
        plt.legend()
        plt.title("Linear Regression (Feature: {})".format(feature_name))
        plt.show()

# Load dataset
df = pd.read_csv("advertising.csv")

# Prepare features and target
X = df.drop('Sales', inplace=False, axis=1)  # Convert to numpy array
y = df['Sales'].values

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# print(X_train)
# print(y_train)

# Train model
LNR = LinearRegressor(learning_rate=0.01, epochs=10000)
LNR.fit(X_train, y_train)

# Evaluate model
mse = LNR.evaluate(X_test, y_test)
print(f"MSE on test set: {mse:.4f}")


LNR.plot_regression_line(X_test, y_test,1,'Radio')