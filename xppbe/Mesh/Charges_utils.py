import os
import numpy as np
from dataclasses import dataclass


@dataclass
class Charges():

    q: float
    x_q: np.ndarray
    r_q: float
    atom_name: str
    res_name: str
    res_num: int
    ion_r_explode: float = 3.5

    @property
    def r_explode(self):
        return self.r_q + self.ion_r_explode


def get_charges_list(pqr_path):
    q, x_q, r_q, atom_name, res_name, res_num = import_charges_from_pqr(pqr_path)
    q_list = list()
    for i in range(len(q)):
        q_list.append(Charges(q=q[i],
                              x_q=x_q[i], 
                              r_q=r_q[i], 
                              atom_name=atom_name[i], 
                              res_name=res_name[i], 
                              res_num=res_num[i]))
    return q_list

def import_charges_from_pqr(pqr_path):
    molecule_file = open(pqr_path, "r")
    molecule_data = molecule_file.read().split("\n")
    atom_count = 0
    for line in molecule_data:
        line = line.split()
        if len(line) == 0 or line[0] != "ATOM":
            continue
        atom_count += 1

    q, x_q, r_q, atom_name, res_name, res_num = (
        np.empty((atom_count,)),
        np.empty((atom_count, 3)),
        np.empty((atom_count,)),
        np.empty((atom_count,), dtype=object),
        np.empty((atom_count,), dtype=object),
        np.empty((atom_count,), dtype=object),
    )
    count = 0
    for line in molecule_data:
        line = line.split()
        if len(line) == 0 or line[0] != "ATOM":
            continue
        q[count] = float(line[8])
        x_q[count, :] = line[5:8]
        r_q[count] = float(line[9])
        atom_name[count] = line[2]
        res_name[count] = line[3]
        res_num[count] = line[4]
        count += 1

    return q, x_q, r_q, atom_name, res_name, res_num

def convert_pqr2xyzr(mesh_pqr_path, mesh_xyzr_path,for_mesh=False):
    
    with open(mesh_xyzr_path, "w") as xyzr_file:
        _,x_q,r_q,_,_,_ = import_charges_from_pqr(mesh_pqr_path)
        for cont in range(len(r_q)):
            nn = '\n' if cont != 0 else ''
            xyzr_file.write(f'{nn}{float(x_q[cont,0]):9.4f} {float(x_q[cont,1]):9.4f} {float(x_q[cont,2]):9.4f} {float(r_q[cont]):9.4f}')
        
        if for_mesh and cont==0:
            xyzr_file.write(f'\n{float(x_q[cont,0]):9.4f} {float(x_q[cont,1])+0.00001:9.4f} {float(x_q[cont,2])*1.00001:9.4f} {float(r_q[cont])*0.00001:9.4f}')

def convert_pdb2pqr(mesh_pdb_path, mesh_pqr_path, force_field, str_flag=""):

    force_field = force_field.upper()
    if str_flag:
        os.system(f"pdb2pqr {str_flag} --ff={force_field} {mesh_pdb_path} {mesh_pqr_path}")
    else:
         os.system(f"pdb2pqr --ff={force_field} {mesh_pdb_path} {mesh_pqr_path}")

    base_path, _ = os.path.splitext(mesh_pdb_path)
    if os.path.exists(base_path+'.log'):
        os.remove(base_path+'.log')


def center_molecule_pqr(pqr_path):
    q, x_q, r_q, atom_name, res_name, res_num = import_charges_from_pqr(pqr_path)
    center = np.mean(np.vstack((np.max(x_q+np.reshape(r_q,(-1,1)), axis=0), np.min(x_q-np.reshape(r_q,(-1,1)), axis=0))), axis=0)
    x_q -= center

    with open(pqr_path, "w") as pqr_file:
        for cont in range(len(q)):
            nn = '\n' if cont != 0 else ''
            pqr_file.write(f'{nn}ATOM     {str(cont+1):6}  {atom_name[cont]:6} {res_name[cont]:5} {res_num[cont]:4} {float(x_q[cont,0]):9.4f} {float(x_q[cont,1]):9.4f} {float(x_q[cont,2]):9.4f} {float(q[cont]):9.4f} {float(r_q[cont]):9.4f}')

    

if __name__=='__main__':
    from xppbe import xppbe_path
    molecule = '9ant'
    path_molecule = os.path.join(xppbe_path,'Molecules',molecule,molecule+'.pqr')
    center_molecule_pqr(path_molecule)