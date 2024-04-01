import numpy as np
import os


rI = 1
epsilon_1 = 1
epsilon_2 = 80
kappa = 0.125
q = 1.0

Rmax = 5.4
Rmin = 2

def G_Yukawa(r):
    sum = 0
    qk = q
    sum += qk*np.exp(-kappa*r)/r
    return (1/(4*np.pi*epsilon_2))*sum


num_points = 100
x = np.random.uniform(-Rmax, Rmax, num_points)
y = np.random.uniform(-Rmax, Rmax, num_points)
z = np.random.uniform(-Rmax, Rmax, num_points)

r = np.sqrt(x**2 + y**2 + z**2)

analytic_values = G_Yukawa(r)

file_path = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(file_path,'data_known.dat'), 'w') as file:
    for i in range(num_points):
        condition = 1 if r[i] <= 1 else 2
        if r[i]>Rmin:
            file.write(f"{condition} {x[i]} {y[i]} {z[i]} {analytic_values[i]}\n")
