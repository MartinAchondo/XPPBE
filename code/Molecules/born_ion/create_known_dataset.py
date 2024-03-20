import numpy as np
import os


rI = 1
epsilon_1 = 1
epsilon_2 = 80
kappa = 0.125
q = 1.0

R = 7


def analytic(r):

    f_IN = lambda r: (q / (4 * np.pi)) * (1 / (epsilon_1 * r) - 1 / (epsilon_1 * rI) + 1 / (epsilon_2 * (1 + kappa * rI) * rI))
    f_OUT = lambda r: (q / (4 * np.pi)) * (np.exp(-kappa * (r - rI)) / (epsilon_2 * (1 + kappa * rI) * r))

    y = np.piecewise(r, [r <= rI, r > rI], [f_IN, f_OUT])

    return y


num_points1 = 500
x1 = np.random.uniform(-rI, rI, num_points1)
y1 = np.random.uniform(-rI, rI, num_points1)
z1 = np.random.uniform(-rI, rI, num_points1)

num_points2 = 500
x2 = np.random.uniform(-R, R, num_points2)
y2 = np.random.uniform(-R, R, num_points2)
z2 = np.random.uniform(-R, R, num_points2)


x = np.concatenate((x1, x2))
y = np.concatenate((y1, y2))
z = np.concatenate((z1, z2))

r = np.sqrt(x**2 + y**2 + z**2)


analytic_values = analytic(r)

file_path = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(file_path,'data_precond.dat'), 'w') as file:
    for i in range(num_points1+num_points2):
        condition = 1 if r[i] <= 1 else 2
        file.write(f"{condition} {x[i]} {y[i]} {z[i]} {analytic_values[i]}\n")
