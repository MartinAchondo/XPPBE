import numpy as np
import os
import sys

current_directory = os.getcwd()
path_components = current_directory.split(os.path.sep)
new_directory = os.path.sep.join(path_components[:-1])
sys.path.append(new_directory)

from Mesh.Charges_utils import get_charges_list


molecule = 'methanol'


Rs = {'born_ion': 1,
      'methanol': 2.54233,
      'arg': 5.9695745}


q_list = get_charges_list(os.path.join(current_directory,molecule,f'{molecule}.pqr'))
epsilon_1 = 1
epsilon_2 = 80
kappa = 0.125
pi = np.pi


Rmin = 1 + Rs[molecule]
Rmax = Rmin + 3


num_points = 300
x = np.random.uniform(-Rmax, Rmax, num_points)
y = np.random.uniform(-Rmax, Rmax, num_points)
z = np.random.uniform(-Rmax, Rmax, num_points)

sum = 0
for q_obj in q_list:
    qk = q_obj.q
    xk,yk,zk = q_obj.x_q
    r = np.sqrt((x-xk)**2+(y-yk)**2+(z-zk)**2)
    sum += qk*np.exp(-kappa*r)/r
phi_values = (1/(4*pi*epsilon_2))*sum



file_path = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(file_path,molecule,'data_known.dat'), 'w') as file:
    for i in range(num_points):
        condition = 1 if r[i] <= Rmin else 2
        if r[i]>Rmin:
            file.write(f"{condition} {x[i]} {y[i]} {z[i]} {phi_values[i]}\n")
