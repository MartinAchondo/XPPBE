import numpy as np
import os
import sys

current_directory = os.getcwd()
path_components = current_directory.split(os.path.sep)
new_directory = os.path.sep.join(path_components[:-1])
sys.path.append(new_directory)

from Mesh.Charges_utils import get_charges_list


def move_to_centroid(atom_coords):
    centroid = np.mean(atom_coords, axis=0)
    atom_coords -= centroid
    return atom_coords

def write_pqr_file(file_path, q_list):
    max_atom_name_len = max(len(q.atom_name) for q in q_list)
    max_res_name_len = max(len(q.res_name) for q in q_list)
    max_res_num_len = max(len(str(q.res_num)) for q in q_list)
    
    with open(file_path, 'w') as file:
        for i, q in enumerate(q_list):
            formatted_line = f'ATOM     {i+1:<4} {q.atom_name:<{max_atom_name_len}} ' + \
                             f'{q.res_name:<{max_res_name_len}}  {q.res_num:<{max_res_num_len}}' + \
                             f'{q.x_q[0]:9.4f}{q.x_q[1]:9.4f}{q.x_q[2]:9.4f}' + \
                             f'{q.q:9.4f}{q.r_q:9.4f}\n'
            file.write(formatted_line)


if __name__ == "__main__":

    molecule = 'methanol'
    pqr_file = os.path.join(os.path.dirname(__file__),f'{molecule}',f"{molecule}.pqr")

    q_list = get_charges_list(pqr_file)
    for q_obj in q_list:
        print(q_obj.x_q)

    n = len(q_list)
    qs = np.zeros(n)
    x_qs = np.zeros((n,3))
    for i,q in enumerate(q_list):
        qs[i] = q.q
        x_qs[i,:] = q.x_q
    total_charge = np.sum(qs)

    xmax,xmin = np.max(x_qs[:,0]),np.min(x_qs[:,0])
    x_centr = (xmax+xmin)/2
    x_qs[:,0] -= x_centr

    ymax,ymin = np.max(x_qs[:,1]),np.min(x_qs[:,1])
    y_centr = (ymax+ymin)/2
    x_qs[:,1] -= y_centr

    zmax,zmin = np.max(x_qs[:,2]),np.min(x_qs[:,2])
    z_centr = (zmax+zmin)/2
    x_qs[:,2] -= z_centr

    for i,q_obj in enumerate(q_list):
        q_obj.x_q = x_qs[i,:]

    write_pqr_file(pqr_file, q_list)
