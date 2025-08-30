import numpy as np
import matplotlib.pyplot as plt

def numerical_derivative(f, x, h=1e-5):
    return (f(x + h) - f(x)) / (h)

def plot_chain_rule(f, g, x_range=(-2, 2), title="Đạo hàm Chain Rule"):
    x_vals = np.linspace(x_range[0], x_range[1], 400)

    # Tính giá trị của g(x) và f(g(x))
    g_vals = g(x_vals)
    f_vals = f(g_vals)

    # Tính đạo hàm theo Chain Rule
    chain_rule_derivative_vals = numerical_derivative(f, g_vals) * numerical_derivative(g, x_vals)

    # Vẽ đồ thị của f(g(x)) và đạo hàm
    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, f_vals, label="f(g(x))", color='blue')
    plt.plot(x_vals, chain_rule_derivative_vals, label="f'(x) theo Chain Rule", color='red', linestyle="--")

    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.axvline(0, color='black', linewidth=0.5, linestyle='--')

    plt.xlabel("x")
    plt.ylabel("Giá trị")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

# f = e^(x^3 - 2x  + 1 )
g = lambda x: x**3 - 2*x + 1
f = lambda x: np.exp(g(x))
plot_chain_rule(f, g, x_range=(0, 2.0), title="Đồ thị f(g(x)) = e^(x^3 - 2x + 1) và đạo hàm theo Chain Rule")