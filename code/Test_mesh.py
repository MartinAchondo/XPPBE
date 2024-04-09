import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from Mesh.Mesh import Domain_Mesh
from Post.Postcode import Postprocessing

from Simulation import get_simulation_name

simulation_name,results_path,main_path,logger = get_simulation_name(__file__)


domain_properties = {
    'molecule': 'sphere',
    }
    
mesh_properties = {
        'vol_mx_interior': 0.05,
        'vol_max_exterior': 0.5,
        'density_mol': 6,
        'density_border': 3,
        'dx_experimental': 0.8,
        'N_pq': 100,
        'G_sigma': 0.04,
        'mesh_generator': 'msms',
        'dR_exterior': 5
        }



Mol_mesh = Domain_Mesh(domain_properties['molecule'], 
                mesh_properties=mesh_properties, 
                save_points=True,
                path=main_path,
                simulation=simulation_name
                )

print(Mol_mesh.R_mol,Mol_mesh.R_max_dist)

Postprocessing.mesh = Mol_mesh
Postprocessing.directory = results_path
Postprocessing.path_plots_meshes = ''
Postprocessing.plot_vol_mesh_3D(Postprocessing)
Postprocessing.plot_collocation_points_3D(Postprocessing)
Postprocessing.plot_surface_mesh_3D(Postprocessing)