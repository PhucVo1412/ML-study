import numpy as np
import matplotlib.pyplot as plt

data = np.array([
    [92.5, 2.1, 65.3, 1800],
    [93.2, 2.5, 67.2, 1825],
    [91.8, 2.3, 64.0, 1795],
    [94.0, 2.8, 70.1, 1850],
    [95.2, 3.0, 72.5, 1880],
    [96.1, 3.2, 74.3, 1905],
    [90.5, 1.8, 61.0, 1750],
    [92.0, 2.0, 63.2, 1780],
    [89.5, 1.5, 59.8, 1725],
    [97.0, 3.5, 76.2, 1925],
    [95.8, 3.1, 73.8, 1890],
    [94.5, 2.9, 71.5, 1860],
    [91.2, 2.2, 62.8, 1775],
    [90.0, 1.7, 60.5, 1740],
    [98.0, 3.7, 78.0, 1950],
    [99.2, 4.0, 80.5, 1980],
    [88.5, 1.3, 58.0, 1700],
    [87.8, 1.1, 56.5, 1680],
    [86.5, 1.0, 55.0, 1650],
    [100.0, 4.2, 82.0, 2000]
])

# Split into X (features) and y (target)
X = data[:, :-1]  # All columns except the last one
y = data[:, -1]   # Last column (Giá vàng)

class LinearRegression:
    def __init__(self):
    # Khởi tạo weights và bias bằng 0
        self.w = np.zeros(X.shape[1]) # weights cho 3 features
        self.b = 0 # bias

        self.current_w = self.w
        self.current_b = self.b


    def compute_gradients(self, X, y):
    # Your code here #
        y_pred =  np.dot(X, self.current_w) + self.current_b
        error = y_pred - y

        dw =  (1/len(X)) * np.dot(X.T, error)
        db =  (1/len(X)) * np.sum(error)
        return dw, db

    def fit(self, X, y, learning_rate=0.001, epochs=10):
        min_loss = float('inf')

        for epoch in range(epochs):
            dw, db = self.compute_gradients(X, y)
            self.current_w -= learning_rate * dw
            self.current_b -= learning_rate * db

            y_pred = self.predict(X)
            loss = (1/len(X)) * np.sum((y - y_pred) ** 2)
            
            if loss < min_loss:
                min_loss = loss
                self.w = self.current_w
                self.b = self.current_b


        # Your code here #
        print(f'Epoch {epoch}, Loss: {loss:.2f}')

    def predict(self, X):
        return np.dot(X, self.w) + self.b


 # Normalize chỉ features X để tránh overflow
X_normalized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Khởi tạo và huấn luyện mô hình
model = LinearRegression()
model.fit(X_normalized, y, learning_rate=0.05, epochs=1000)

# In ra weights và bias cuối cùng
print("\nFinal weights:", model.w)
print("Final bias:", model.b)

# Dự đoán và so sánh kết quả
y_pred = model.predict(X_normalized)

print("\nSo sánh kết quả thực tế và dự đoán:")
for i in range(5):
    print(f"Thực tế: {y[i]:.2f}, Dự đoán: {y_pred[i]:.2f}")
