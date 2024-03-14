import os
import logging
import shutil

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from Mesh.Molecule_Mesh import Molecule_Mesh
from Model.PDE_Model import PBE
from NN.NeuralNet import XPINN_NeuralNet
from NN.XPINN import XPINN
from Post.Postcode import Born_Ion_Postprocessing as Postprocessing


simulation_name = os.path.basename(os.path.abspath(__file__)).replace('.py','')
main_path = os.path.join(os.path.dirname(os.path.abspath(__file__)))

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



# Inputs
###############################################

class PDE():

        def __init__(self):
                
                self.inputs = {'molecule': 'born_ion',
                                'epsilon_1':  1,
                                'epsilon_2': 80,
                                'kappa': 0.125,
                                'T' : 300 
                                }
                
                self.N_points = {'hmin_interior': 0.01,
                                'hmin_exterior': 0.5,
                                'density_mol': 40,
                                'density_border': 4,
                                'dx_experimental': 2,
                                'N_pq': 100,
                                'G_sigma': 0.04,
                                'mesh_generator': 'msms',
                                'dR_exterior': 8
                                }

        def create_simulation(self):

                self.Mol_mesh = Molecule_Mesh(self.inputs['molecule'], 
                                N_points=self.N_points, 
                                plot='batch',
                                path=main_path,
                                simulation=simulation_name
                                )
        
                self.PBE_model = PBE(self.inputs,
                        mesh=self.Mol_mesh, 
                        model='linear',
                        path=main_path
                        ) 
                

                self.meshes_domain = dict()

                self.meshes_domain['1'] = {'domain': 'molecule', 'type':'R1', 'fun':lambda x,y,z: self.PBE_model.source(x,y,z)}
                self.meshes_domain['2'] = {'domain': 'molecule', 'type':'Q1', 'fun':lambda x,y,z: self.PBE_model.source(x,y,z)}
                self.meshes_domain['3'] = {'domain': 'molecule', 'type':'K1', 'file':'data_known.dat', 'noise': True}
                #self.meshes_domain['4'] = {'domain': 'molecule', 'type':'P1', 'file':'data_precond.dat'}

                self.meshes_domain['5'] = {'domain': 'solvent', 'type':'R2', 'value':0.0}
                self.meshes_domain['6'] = {'domain': 'solvent', 'type':'D2', 'fun':lambda x,y,z: self.PBE_model.border_value(x,y,z)}
                self.meshes_domain['7'] = {'domain': 'solvent', 'type':'K2', 'file':'data_known.dat'}
                #self.meshes_domain['8'] = {'domain': 'solvent', 'type':'P2', 'file':'data_precond.dat'}
                #self.meshes_domain['9'] = {'type': 'E2', 'file': 'data_experimental.dat'}
 
                self.meshes_domain['10'] = {'domain':'interface', 'type':'I'}
                #self.meshes_domain['11'] = {'domain':'interface', 'type':'G'}

                self.PBE_model.mesh.adapt_meshes_domain(self.meshes_domain,self.PBE_model.q_list)
        
                self.XPINN_solver = XPINN()

                self.XPINN_solver.adapt_PDEs(self.PBE_model)



def main():

        sim = PDE()
        sim.create_simulation()

        XPINN_solver = sim.XPINN_solver


        weights = {'E2': 100.5,
                  }

        XPINN_solver.adapt_weights([weights,weights],
                                   adapt_weights = True,
                                   adapt_w_iter = 5,
                                   adapt_w_method = 'gradients',
                                   alpha = 0.7)             

        hyperparameters_in = {
                        'input_shape': (None,3),
                        'num_hidden_layers': 4,
                        'num_neurons_per_layer': 20,
                        'output_dim': 1,
                        'activation': 'tanh',
                        'adaptative_activation': True,
                        'architecture_Net': 'FCNN',
                        'fourier_features': True,
                        'num_fourier_features': 12,
                        'scale': XPINN_solver.mesh.scale_1
                }

        hyperparameters_out = {
                        'input_shape': (None,3),
                        'num_hidden_layers': 4,
                        'num_neurons_per_layer': 20,
                        'output_dim': 1,
                        'activation': 'tanh',
                        'adaptative_activation': True,
                        'architecture_Net': 'FCNN',
                        'fourier_features': False,
                        'scale': XPINN_solver.mesh.scale_2
                }

        XPINN_solver.create_NeuralNet(XPINN_NeuralNet,[hyperparameters_in,hyperparameters_out])

        XPINN_solver.set_points_methods(sample_method='random_sample')

        optimizer = 'Adam'
        lr_s = ([1000,1600],[1e-2,5e-3,5e-4])
        lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay(*lr_s)
        lr = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=0.001,
                decay_steps=2000,
                decay_rate=0.9,
                staircase=True)
        lr_p = 0.001
        XPINN_solver.adapt_optimizer(optimizer,lr,lr_p)

        N_iters = 16

        precondition = False
        N_precond = 5

        iters_save_model = 6
        XPINN_solver.folder_path = folder_path

        XPINN_solver.solve(N=N_iters, 
                        precond = precondition, 
                        N_precond = N_precond,  
                        save_model = iters_save_model, 
                        G_solve_iter=4)


        Post = Postprocessing(XPINN_solver, save=True, directory=folder_path)

        Post.plot_loss_history(domain=1);
        Post.plot_loss_history(domain=1,loss='RIuD');
        Post.plot_loss_history(domain=2);
        Post.plot_loss_history(domain=1, plot_w=True);
        Post.plot_loss_history(domain=2, plot_w=True);
        Post.plot_weights_history(domain=1);
        Post.plot_weights_history(domain=2);
        Post.plot_loss_validation_history(domain=1,loss='TL')
        Post.plot_loss_validation_history(domain=2,loss='TL')
        Post.plot_loss_validation_history(domain=1,loss='R')
        Post.plot_loss_validation_history(domain=1,loss='Q')

        Post.plot_G_solv_history();
        Post.plot_collocation_points_3D();
        Post.plot_mesh_3D();
        Post.plot_interface_3D(variable='phi');
        Post.plot_interface_3D(variable='dphi');
        Post.plot_phi_line();
        Post.plot_phi_contour();
        Post.save_values_file();
        Post.save_model_summary();

        Post.plot_aprox_analytic();
        Post.plot_aprox_analytic(zoom=True);
        Post.plot_line_interface();
        Post.plot_line_interface(plot='du');

        Post.plot_architecture(domain=1);
        Post.plot_architecture(domain=2);

if __name__=='__main__':
        main()


