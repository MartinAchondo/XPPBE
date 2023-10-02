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
from Post.Postprocessing import View_results
from Post.Postprocessing import View_results_X


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

def main():

        inputs = {'molecule': 'sphere',
                'epsilon_1':  1,
                'epsilon_2': 80,
                'kappa': 0.125,
                'T' : 300 
                }

        N_points = {'N_interior': 40,
                'N_exterior': 40,
                'N_border': 30,
                'dR_exterior': 9
                }

        Mol_mesh = Molecule_Mesh(inputs['molecule'], 
                                N_points=N_points, 
                                N_batches=1,
                                refinement=True,
                                plot=False,
                                path=main_path
                                )
        
        PBE_model = PBE(inputs,
                        mesh=Mol_mesh, 
                        model='linear',
                        path=main_path
                        ) 

        meshes_in = dict()
        meshes_in['1'] = {'type':'R', 'value':None, 'fun':lambda x,y,z: PBE_model.source(x,y,z)}
        meshes_in['2'] = {'type':'K', 'value':None, 'fun':None, 'file':'data_known.dat'}
        meshes_in['3'] = {'type':'P', 'value':None, 'fun':None, 'file':'data_precond.dat'}
        PBE_model.PDE_in.mesh.adapt_meshes(meshes_in)

        meshes_out = dict()
        meshes_out['1'] = {'type':'R', 'value':0.0, 'fun':None}
        meshes_out['2'] = {'type':'D', 'value':None, 'fun':lambda x,y,z: PBE_model.border_value(x,y,z)}
        meshes_out['3'] = {'type':'K', 'value':None, 'fun':None, 'file':'data_known.dat'}
        meshes_out['4'] = {'type':'P', 'value':None, 'fun':None, 'file':'data_precond.dat'}
        PBE_model.PDE_out.mesh.adapt_meshes(meshes_out)

        meshes_domain = dict()
        meshes_domain['1'] = {'type':'I', 'value':None, 'fun':None}
        meshes_domain['2'] = {'type': 'E', 'file': 'data_experimental.dat'}
        PBE_model.mesh.adapt_meshes_domain(meshes_domain,PBE_model.q_list)
       
        XPINN_solver = XPINN(PINN)
        XPINN_solver.adapt_PDEs(PBE_model)

        weights = {'w_r': 1,
                   'w_d': 1,
                   'w_n': 1,
                   'w_i': 1,
                   'w_k': 1,
                   'w_e': 1
                  }

        XPINN_solver.adapt_weights([weights,weights],
                                   adapt_weights = False,
                                   adapt_w_iter = 4000,
                                   adapt_w_method = 'gradients')             

        hyperparameters_in = {
                        'input_shape': (None,3),
                        'num_hidden_layers': 4,
                        'num_neurons_per_layer': 160,
                        'output_dim': 1,
                        'activation': 'tanh',
                        'architecture_Net': 'FCNN'
                }

        hyperparameters_out = {
                        'input_shape': (None,3),
                        'num_hidden_layers': 4,
                        'num_neurons_per_layer': 160,
                        'output_dim': 1,
                        'activation': 'tanh',
                        'architecture_Net': 'FCNN'
                }

        XPINN_solver.create_NeuralNets(NeuralNet,[hyperparameters_in,hyperparameters_out])

        optimizer = 'Adam'
        lr = ([5000,8000],[1e-3,5e-4,1e-4])
        lr_p = 0.001
        XPINN_solver.adapt_optimizers(optimizer,[lr,lr],lr_p)

        
        N_iters = 12000

        precondition = True
        N_precond = 2000

        iters_save_model = 1000
        XPINN_solver.folder_path = folder_path

        XPINN_solver.solve(N=N_iters, 
                        precond = precondition, 
                        N_precond = N_precond,  
                        save_model = iters_save_model, 
                        shuffle = False, 
                        shuffle_iter = 150 )


        Post = View_results_X(XPINN_solver, View_results, save=True, directory=folder_path)

        Post.plot_loss_history();
        Post.plot_loss_history(plot_w=True);
        Post.plot_weights_history();

        Post.plot_u_plane();
        Post.plot_aprox_analytic();
        Post.plot_interface();

        # Post.plot_u_domain_contour();


if __name__=='__main__':

        main()


