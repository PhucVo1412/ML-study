import numpy as np
import matplotlib.pyplot as plt

def numerical_derivative(f, x, h=1e-5):
    return (f(x + h) - f(x)) / (h)


def plot_function_and_derivative(f, x_range=(-2, 2), title="Đồ thị f(x) vàf'(x)"):
    x_vals = np.linspace(x_range[0], x_range[1], 400)
    y_vals = f(x_vals)
    y_derivative_vals = numerical_derivative(f, x_vals)

    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, y_vals, label="f(x)", color='blue')
    plt.plot(x_vals, y_derivative_vals, label="f'(x) (numerical)", color='red', linestyle="--")
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
    plt.xlabel("x")
    plt.ylabel("f(x) & f'(x)")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

# f = x^2 - 3x + 2
f = lambda x: x**2 - 3*x + 2
plot_function_and_derivative(f, x_range=(-1, 4), title="Đồ thị f(x) = x^2 - 3x + 2 và f'(x)")


