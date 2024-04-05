import numpy as np
import os
import sys

current_directory = os.getcwd()
path_components = current_directory.split(os.path.sep)
new_directory = os.path.sep.join(path_components[:-1])
sys.path.append(new_directory)

from Mesh.Charges_utils import get_charges_list
from Model.Solutions_utils import Solution_utils as Model_Funcs

molecule = 'methanol'


Rs = {'born_ion': 1,
      'methanol': 2.54233,
      'arg': 5.9695745}


q_list = get_charges_list(os.path.join(current_directory,molecule,f'{molecule}.pqr'))
Model_Funcs.epsilon_1 = 1
Model_Funcs.epsilon_2 = 80
Model_Funcs.kappa = 0.125
Model_Funcs.pi = np.pi
Model_Funcs.q_list = q_list


Rmin = 1 + Rs[molecule]
Rmax = Rmin + 3


num_points = 300
x = np.random.uniform(-Rmax, Rmax, num_points)
y = np.random.uniform(-Rmax, Rmax, num_points)
z = np.random.uniform(-Rmax, Rmax, num_points)
r = np.sqrt(x**2+y**2+z**2)

phi_values = Model_Funcs.G_Yukawa(Model_Funcs, x,y,z)


file_path = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(file_path,molecule,'data_known.dat'), 'w') as file:
    for i in range(num_points):
        condition = 1 if r[i] <= Rmin else 2
        if r[i]>Rmin:
            file.write(f"{condition} {x[i]} {y[i]} {z[i]} {phi_values[i]}\n")
