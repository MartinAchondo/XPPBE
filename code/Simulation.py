import os
import logging
import shutil

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from Mesh.Molecule_Mesh import Molecule_Mesh
from NN.XPINN import XPINN

def get_simulation_name(file):

    main_path = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    simulation_name =  os.path.basename(os.path.abspath(file)).replace('.py','')

    folder_name = simulation_name
    folder_path = os.path.join(main_path,'results',folder_name)
    if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
    os.makedirs(folder_path)

    filename = os.path.join(folder_path,'logfile.log')
    LOG_format = '%(levelname)s - %(name)s: %(message)s'
    logging.basicConfig(filename=filename, filemode='w', level=logging.INFO, format=LOG_format)
    logger = logging.getLogger(__name__)
    logger.info('================================================')
    
    return simulation_name,folder_path,main_path


class Simulation():
       
    def __init__(self,simulation_file):

        self.simulation_name, self.folder_path, self.main_path = get_simulation_name(simulation_file)
            
        self.equation = 'standard'
        self.pbe_model = 'linear'
        
        self.domain_properties = {
            'molecule': 'born_ion',
            'epsilon_1':  1,
            'epsilon_2': 80,
            'kappa': 0.125,
            'T' : 300 
            }
            
        self.mesh_properties = {
            'vol_max_interior': 0.04,
            'vol_max_exterior': 0.1,
            'density_mol': 40,
            'density_border': 4,
            'dx_experimental': 2,
            'N_pq': 100,
            'G_sigma': 0.04,
            'mesh_generator': 'msms',
            'dR_exterior': 8
            }

        self.sample_method='random_sample'

        self.losses = ['R1','R2','D2','I','K1','K2']

        self.weights = {'E2': 10**-10}

        self.adapt_weights = True,
        self.adapt_w_iter = 1000
        self.adapt_w_method = 'gradients'
        self.alpha_w = 0.7

        self.G_solve_iter=1000

        self.network = 'xpinn'

        self.hyperparameters_in = {
                            'input_shape': (None,3),
                            'num_hidden_layers': 4,
                            'num_neurons_per_layer': 200,
                            'output_dim': 1,
                            'activation': 'tanh',
                            'adaptative_activation': True,
                            'architecture_Net': 'FCNN',
                            'fourier_features': True,
                            'num_fourier_features': 256
                        }

        self.hyperparameters_out = {
                        'input_shape': (None,3),
                        'num_hidden_layers': 4,
                        'num_neurons_per_layer': 200,
                        'output_dim': 1,
                        'activation': 'tanh',
                        'adaptative_activation': True,
                        'architecture_Net': 'FCNN',
                        'fourier_features': True,
                        'num_fourier_features': 256
                    }   


        self.optimizer = 'Adam'
        self.lr = tf.keras.optimizers.schedules.ExponentialDecay(
                        initial_learning_rate=0.001,
                        decay_steps=2000,
                        decay_rate=0.9,
                        staircase=True)
        self.lr_p = 0.001

        self.N_iters = 10000

        self.precondition = False
        self.N_precond = 0

        self.iters_save_model = 1000


    def create_simulation(self):

        self.Mol_mesh = Molecule_Mesh(self.domain_properties['molecule'], 
                        N_points=self.mesh_properties, 
                        plot='batch',
                        path=self.main_path,
                        simulation=self.simulation_name
                        )

        if self.equation == 'standard':
                from Model.Equations import PBE_Std as PBE
        elif self.equation == 'regularized_scheme_1':
                from Model.Equations import PBE_Reg_1 as PBE
        elif self.equation == 'regularized_scheme_2':
                from Model.Equations import PBE_Reg_2 as PBE

        self.PBE_model = PBE(self.domain_properties,
                mesh=self.Mol_mesh, 
                model=self.pbe_model,
                path=self.main_path
                ) 

        meshes_domain = dict()           
        if self.equation == 'standard':
            meshes_domain['R1'] = {'domain': 'molecule', 'type':'R1', 'fun':lambda x,y,z: self.PBE_model.source(x,y,z)}
            meshes_domain['Q1'] = {'domain': 'molecule', 'type':'Q1', 'fun':lambda x,y,z: self.PBE_model.source(x,y,z)}
            meshes_domain['R2'] = {'domain': 'solvent', 'type':'R2', 'value':0.0}
            meshes_domain['D2'] = {'domain': 'solvent', 'type':'D2', 'fun':lambda x,y,z: self.PBE_model.border_value(x,y,z)}
                
        elif self.equation == 'regularized_scheme_1':
            meshes_domain['R1'] = {'domain': 'molecule', 'type':'R1', 'value':0.0}
            meshes_domain['R2'] = {'domain': 'solvent', 'type':'R2', 'value':0.0}
            meshes_domain['D2'] = {'domain': 'solvent', 'type':'D2', 'fun':lambda x,y,z: self.PBE_model.border_value(x,y,z)-self.PBE_model.G(x,y,z)}

        elif self.equation == 'regularized_scheme_2':
            meshes_domain['R1'] = {'domain': 'molecule', 'type':'R1', 'value':0.0}
            meshes_domain['R2'] = {'domain': 'solvent', 'type':'R2', 'value':0.0}
            meshes_domain['D2'] = {'domain': 'solvent', 'type':'D2', 'fun':lambda x,y,z: self.PBE_model.border_value(x,y,z)}

        meshes_domain['K1'] = {'domain': 'molecule', 'type':'K1', 'file':'data_known.dat'}
        meshes_domain['K2'] = {'domain': 'solvent', 'type':'K2', 'file':'data_known.dat'}
        meshes_domain['P1'] = {'domain': 'molecule', 'type':'P1', 'file':'data_precond.dat'}
        meshes_domain['P2'] = {'domain': 'solvent', 'type':'P2', 'file':'data_precond.dat'}
        meshes_domain['E2'] = {'domain': 'solvent', 'type': 'E2', 'file': 'data_experimental.dat'}

        if self.network=='xpinn':
                meshes_domain['I'] = {'domain':'interface', 'type':'I'}
        meshes_domain['G'] = {'domain':'interface', 'type':'G'}

        self.meshes_domain = dict()
        for t in self.losses:
            self.meshes_domain[t] = meshes_domain[t]

        self.PBE_model.mesh.adapt_meshes_domain(self.meshes_domain,self.PBE_model.q_list)

        self.XPINN_solver = XPINN()
        self.XPINN_solver.folder_path = self.folder_path
        self.XPINN_solver.adapt_PDE(self.PBE_model)


    def adapt_simulation(self):

        self.XPINN_solver.adapt_weights(
                self.weights,
                adapt_weights = self.adapt_weights,
                adapt_w_iter = self.adapt_w_iter,
                adapt_w_method = self.adapt_w_method,
                alpha_w = self.alpha_w
                )             

        self.hyperparameters_in['scale'] = self.XPINN_solver.mesh.scale_1
        self.hyperparameters_out['scale'] = self.XPINN_solver.mesh.scale_2
                
        if self.network == 'xpinn':
            from NN.NeuralNet import XPINN_NeuralNet as NeuralNet
        elif self.network == 'pinn':
            from NN.NeuralNet import PINN_NeuralNet as NeuralNet

        self.XPINN_solver.create_NeuralNet(NeuralNet,[self.hyperparameters_in,self.hyperparameters_out])
        self.XPINN_solver.set_points_methods(sample_method=self.sample_method)
        self.XPINN_solver.adapt_optimizer(self.optimizer,self.lr,self.lr_p)


    def solve_model(self):
          
        self.XPINN_solver.solve(
            N=self.N_iters, 
            precond = self.precondition, 
            N_precond = self.N_precond,  
            save_model = self.iters_save_model, 
            G_solve_iter= self.G_solve_iter
            )
        

    def postprocessing(self):
          
        if self.domain_properties['molecule'] == 'born_ion':
            from Post.Postcode import Born_Ion_Postprocessing as Postprocessing
        else:
            from Post.Postcode import Postprocessing

        Post = Postprocessing(self.XPINN_solver, save=True, directory=self.folder_path)

        Post.plot_loss_history(domain=1);
        Post.plot_loss_history(domain=2);
        Post.plot_loss_history(domain=1, plot_w=True);
        Post.plot_loss_history(domain=2, plot_w=True);

        # Post.plot_loss_history(domain=1,loss='RIu');
        # Post.plot_loss_history(domain=2,loss='RIuD');
        
        Post.plot_loss_validation_history(domain=1,loss='TL')
        Post.plot_loss_validation_history(domain=2,loss='TL')
        Post.plot_loss_validation_history(domain=1,loss='R')
        Post.plot_loss_validation_history(domain=2,loss='R')

        Post.plot_weights_history(domain=1);
        Post.plot_weights_history(domain=2);

        Post.plot_collocation_points_3D();
        Post.plot_vol_mesh_3D();
        Post.plot_surface_mesh_3D();

        Post.plot_G_solv_history();
        Post.plot_phi_line(); 
        Post.plot_phi_line(value='react'); 
        Post.plot_phi_contour(); 
        Post.plot_phi_contour(value='react'); 
        Post.plot_interface_3D(variable='phi');
        Post.plot_interface_3D(variable='dphi');

        if self.domain_properties['molecule'] == 'born_ion':
            Post.plot_aprox_analytic(); 
            Post.plot_aprox_analytic(value='react'); 
            Post.plot_aprox_analytic(zoom=True); 
            Post.plot_aprox_analytic(zoom=True, value='react'); 
            Post.plot_line_interface(); 
            Post.plot_line_interface(value='react'); 
            Post.plot_line_interface(plot='du'); 
            Post.plot_line_interface(plot='du',value='react'); 

        Post.save_values_file();
        Post.save_model_summary();
        Post.plot_architecture(domain=1);
        Post.plot_architecture(domain=2);

  
def main():
    sim = Simulation()
    sim.create_simulation(__file__)
    sim.adapt_simulation()
    sim.solve_model()
    sim.postprocessing()


if __name__=='__main__':
        main()

