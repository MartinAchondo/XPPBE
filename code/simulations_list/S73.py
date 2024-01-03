import os
import logging
import shutil

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


from Model.Mesh.Molecule_Mesh import Molecule_Mesh
from Model.PDE_Model import PBE
from NN.NeuralNet import NeuralNet
from NN.PINN import PINN 
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
                
                self.N_points = {'dx_interior': 0.12,
                                'dx_exterior': 0.8,
                                'N_border': 15,
                                'dR_exterior': 9,
                                'dx_experimental': 0.8,
                                'N_pq': 70,
                                'G_sigma': 0.04,
                                'mesh_density': 3
                                }

        def create_simulation(self):

                self.Mol_mesh = Molecule_Mesh(self.inputs['molecule'], 
                                N_points=self.N_points, 
                                plot=True,
                                path=main_path,
                                simulation=simulation_name
                                )
        
                self.PBE_model = PBE(self.inputs,
                        mesh=self.Mol_mesh, 
                        model='nonlinear',
                        path=main_path
                        ) 
                
                self.meshes_in = dict()
                self.meshes_in['1'] = {'type':'R', 'value':None, 'fun':lambda x,y,z: self.PBE_model.source(x,y,z)}
                self.meshes_in['2'] = {'type':'Q', 'value':None, 'fun':lambda x,y,z: self.PBE_model.source(x,y,z)}
                self.meshes_in['3'] = {'type':'K', 'value':None, 'fun':None, 'file':'data_known.dat', 'noise': True}
                #self.meshes_in['4'] = {'type':'P', 'value':None, 'fun':None, 'file':'data_precond.dat'}

                self.PBE_model.PDE_in.mesh.adapt_meshes(self.meshes_in)

                self.meshes_out = dict()
                self.meshes_out['1'] = {'type':'R', 'value':0.0, 'fun':None}
                self.meshes_out['2'] = {'type':'D', 'value':None, 'fun':lambda x,y,z: self.PBE_model.border_value(x,y,z)}
                self.meshes_out['3'] = {'type':'K', 'value':None, 'fun':None, 'file':'data_known.dat', 'noise': True}
                #self.meshes_out['4'] = {'type':'P', 'value':None, 'fun':None, 'file':'data_precond.dat'}
                self.PBE_model.PDE_out.mesh.adapt_meshes(self.meshes_out)

                self.meshes_domain = dict()
                self.meshes_domain['1'] = {'type':'I', 'value':None, 'fun':None}
                self.meshes_domain['2'] = {'type': 'E', 'file': 'data_experimental.dat'}
                #self.meshes_domain['3'] = {'type':'G', 'value':None, 'fun':None}
                self.PBE_model.mesh.adapt_meshes_domain(self.meshes_domain,self.PBE_model.q_list)
        
                self.XPINN_solver = XPINN(PINN)

                self.XPINN_solver.adapt_PDEs(self.PBE_model)


def main():

        sim = PDE()
        sim.create_simulation()

        XPINN_solver = sim.XPINN_solver


        weights = {'w_r': 1,
                   'w_d': 1,
                   'w_n': 1,
                   'w_i': 1,
                   'w_k': 1,
                   'w_e': 1e-4,
                   'w_g': 1e-4
                  }

        XPINN_solver.adapt_weights([weights,weights],
                                   adapt_weights = True,
                                   adapt_w_iter = 1000,
                                   adapt_w_method = 'gradients',
                                   alpha = 0.7)             

        hyperparameters_in = {
                        'input_shape': (None,3),
                        'num_hidden_layers': 4,
                        'num_neurons_per_layer': 300,
                        'output_dim': 1,
                        'activation': 'tanh',
                        'architecture_Net': 'FCNN',
                        'fourier_features': True,
                        'num_fourier_features': 256
                }

        hyperparameters_out = {
                        'input_shape': (None,3),
                        'num_hidden_layers': 4,
                        'num_neurons_per_layer': 300,
                        'output_dim': 1,
                        'activation': 'tanh',
                        'architecture_Net': 'FCNN',
                        'fourier_features': True,
                        'num_fourier_features': 256
                }

        XPINN_solver.create_NeuralNets(NeuralNet,[hyperparameters_in,hyperparameters_out])

        XPINN_solver.set_points_methods(
                sample_method='batches', 
                N_batches=1, 
                sample_size=50)

        optimizer = 'Adam'
        #lr_s = ([1000,1600],[1e-2,5e-3,5e-4])
        #lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay(*lr_s)
        lr = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=0.001,
                decay_steps=2000,
                decay_rate=0.9,
                staircase=True)
        lr_p = 0.001
        XPINN_solver.adapt_optimizers(optimizer,[lr,lr],lr_p)

        N_iters = 10000

        precondition = False
        N_precond = 5

        iters_save_model = 1000
        XPINN_solver.folder_path = folder_path

        XPINN_solver.solve(N=N_iters, 
                        precond = precondition, 
                        N_precond = N_precond,  
                        save_model = iters_save_model, 
                        G_solve_iter=200)


        Post = Postprocessing(XPINN_solver, save=True, directory=folder_path)

        Post.plot_loss_history(domain=1);
        Post.plot_loss_history(domain=2);
        Post.plot_loss_history(domain=1, plot_w=True);
        Post.plot_loss_history(domain=2, plot_w=True);
        Post.plot_weights_history(domain=1);
        Post.plot_weights_history(domain=2);

        
        Post.plot_G_solv_history();
        Post.plot_collocation_points_3D();
        Post.plot_mesh_3D();
        Post.plot_interface_3D(variable='phi');
        Post.plot_interface_3D(variable='dphi');
        Post.plot_phi_line();
        Post.plot_phi_contour();
        Post.plot_aprox_analytic();
        Post.plot_aprox_analytic(zoom=True);
        Post.plot_line_interface();
        Post.save_values_file();

        Post.plot_architecture(domain=1);
        Post.plot_architecture(domain=2);

if __name__=='__main__':
        main()


