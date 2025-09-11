def find_squared_root(a):
    EPSILON = 0.000001

    x = a 

    while True:
        new_x = x - ((x**2 - a) / (2*x))

        if abs(new_x - x) < EPSILON:
            return new_x
        x = new_x

print(find_squared_root(2.5*2.5))