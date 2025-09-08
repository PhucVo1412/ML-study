import numpy as np
import matplotlib.pyplot as plt

def exercise_1():
    # Tìm cực tiểu của hàm số f(x) = x^4 + 3x^3 + 2
    # --> Dùng Gradient Descent: f'(x)
    f = lambda x : x**4 - 3*x**3 +2

    f_prime = lambda x: 4*x**3 - 9*x**2 

    def gradient_descent(f_prime, x_init, learning_rate=0.01, epochs=100):
        x = x_init
        history = [x]
        for _ in range(epochs):
            x = x - learning_rate * f_prime(x)
            history.append(x)
        return history

    # Chạy Gradient Descent
    x_init = 0.5
    learning_rate = 0.01
    epochs = 100
    history = gradient_descent(f_prime, x_init, learning_rate,epochs)

    x_optimal = min(history, key=f)

    x_vals = np.linspace(-1, 3, 100)
    y_vals = f(x_vals)

    plt.plot(x_vals, y_vals, label="f(x)")
    plt.scatter(history, [f(x) for x in history], color="red", s=10, label=" Iterations")
    plt.scatter(x_optimal, f(x_optimal), color="green", marker="x", s=100,label=f"Min at x={x_optimal:.3f}")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid()
    plt.show()

def exercise_2():
    # Tìm hệ số w,b tối ưu cho mô hình hồi quy: y = wx + b

    X = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 3, 4, 5, 6])

    w, b = 0.0, 0.0
    learning_rate = 0.01
    epochs = 1000
    for _ in range(epochs):
        y_pred = w * X + b
        error = y_pred - y
        w_grad = (2/len(X)) * np.dot(error, X)
        b_grad = (2/len(X)) * np.sum(error)
        w -= learning_rate * w_grad
        b -= learning_rate * b_grad
        

    
    plt.scatter(X, y, label="Data")
    plt.plot(X, y_pred, color='red', label="Regression Line")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.grid()
    plt.show()
    print(f"Hệ số tối ưu: w = {w:.3f}, b = {b:.3f}")