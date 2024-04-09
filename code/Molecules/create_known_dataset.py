import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.getcwd()))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from Mesh.Charges_utils import get_charges_list
from Model.Solutions_utils import Solution_utils as Model_Funcs

file_path = os.path.dirname(os.path.realpath(__file__))


def calculate_print_phi(domain,method, num_points):

    R = R_mol if domain == 'molecule' else Rmax if domain == 'solvent' else None
    x = np.random.uniform(-R, R, num_points)
    y = np.random.uniform(-R, R, num_points)
    z = np.random.uniform(-R, R, num_points)
    r = np.sqrt(x**2+y**2+z**2)
    X = np.stack([x,y,z], axis=1)

    if method == 'Harmonic_spheres':
        phi_values = Model_Funcs.Harmonic_spheres(Model_Funcs, X, 'molecule', R_mol)
    elif method == 'G_Yukawa':
        phi_values = Model_Funcs.G_Yukawa(Model_Funcs, x,y,z)
    elif method == 'analytic_Born_Ion':
        phi_values = Model_Funcs.analytic_Born_Ion(Model_Funcs,r)

    if domain == 'molecule':
        with open(os.path.join(file_path,molecule,'data_known.dat'), 'a') as file:
            for i in range(num_points):
                condition = 1 if r[i] <= R_mol else 2
                if r[i]<R_mol:
                    file.write(f"{condition} {x[i]} {y[i]} {z[i]} {phi_values[i]}\n")

    elif domain == 'solvent':
        with open(os.path.join(file_path,molecule,'data_known.dat'), 'a') as file:
            for i in range(num_points):
                condition = 1 if r[i] <= R_mol else 2
                if r[i]>Rmin:
                    file.write(f"{condition} {x[i]} {y[i]} {z[i]} {phi_values[i]}\n")


molecule = 'born_ion'

with open(os.path.join(file_path,molecule,'data_known.dat'), 'w') as file:
    pass


Rs = {'born_ion': 1,
      'methanol': 2.54233,
      'arg': 5.9695745,
      'sphere': 1.2}


q_list = get_charges_list(os.path.join(os.getcwd(),molecule,f'{molecule}.pqr'))
Model_Funcs.epsilon_1 = 1
Model_Funcs.epsilon_2 = 80
Model_Funcs.kappa = 0.125
Model_Funcs.pi = np.pi
Model_Funcs.q_list = q_list

n = len(Model_Funcs.q_list)
Model_Funcs.qs = np.zeros(n)
Model_Funcs.x_qs = np.zeros((n,3))
for i,q in enumerate(Model_Funcs.q_list):
    Model_Funcs.qs[i] = q.q
    Model_Funcs.x_qs[i,:] = q.x_q
Model_Funcs.total_charge = np.sum(Model_Funcs.qs)

dR = 0
R_mol = Rs[molecule]
Rmin = dR + R_mol
Rmax = Rmin + 3

calculate_print_phi('solvent','G_Yukawa',300)
calculate_print_phi('molecule','Harmonic_spheres',300)

print('File created')