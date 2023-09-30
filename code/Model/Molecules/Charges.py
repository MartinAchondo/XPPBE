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



if __name__=='__main__':
    path_files = os.path.join(os.getcwd(),'code','Model','Molecules')
    ch = get_charges_list(os.path.join(path_files,'1ubq','1ubq'+'.pqr'))
    print(ch)