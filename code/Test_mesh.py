
from Mesh.Mesh import Domain_Mesh
from Post.Postcode import Postprocessing

from Simulation import get_simulation_name

simulation_name,results_path,main_path,logger = get_simulation_name(__file__)


domain_properties = {
    'molecule': 'arg',
    }
    
mesh_properties = {
        'vol_max_interior': 0.05,
        'vol_max_exterior': 0.5,
        'density_mol': 3.0,
        'density_border': 4,
        'dx_experimental': 0.8,
        'N_pq': 100,
        'G_sigma': 0.04,
        'mesh_generator': 'nanoshaper',
        'dR_exterior': 5
        }



Mol_mesh = Domain_Mesh(domain_properties['molecule'], 
                mesh_properties=mesh_properties, 
                save_points=True,
                path=main_path,
                simulation=simulation_name
                )

Postprocessing.mesh = Mol_mesh
Postprocessing.directory = results_path
Postprocessing.path_plots_meshes = ''
Postprocessing.plot_vol_mesh_3D(Postprocessing)