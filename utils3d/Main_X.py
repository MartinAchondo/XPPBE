import os
import logging
import shutil

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


from DCM.PDE_Model import Poisson
from DCM.PDE_Model import Helmholtz
from DCM.PDE_Model import PBE_Interface

from Simulation_X import Simulation


main_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'results')
#if os.path.exists(main_path):
#        shutil.rmtree(main_path)
#os.makedirs(main_path)

folder_name = 'data'
folder_path = os.path.join(main_path,folder_name)
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

    Sim = Simulation(PBE_Interface)

    # PDE
    q_list = [(1,[0,0,0])]

    inputs = {'Problem': 'Main_X',
              'rmin': 0,
              'rI': 1,
              'rB': 10,
              'epsilon_1':1,
              'epsilon_2':80,
              'kappa': 0.125,
              }
    
    Sim.problem = inputs
    Sim.q = q_list
    
    rI = inputs['rI']
    rB = inputs['rB']

    Sim.domain_in = ([-rI,rI],[-rI,rI],[-rI,rI])
    Sim.PDE_in = Poisson()
    Sim.PDE_in.sigma = 0.04
    Sim.PDE_in.epsilon = inputs['epsilon_1']
    Sim.PDE_in.epsilon_G = inputs['epsilon_1']
    Sim.PDE_in.q = q_list
    Sim.PDE_in.problem = inputs

    inner_interface = {'type':'I', 'value':None, 'fun':None, 'r':rI, 'N': 15}
    inner_data = {'type':'K', 'value':None, 'fun':lambda x,y,z: Sim.PDE_in.analytic(x,y,z), 'r':'Random', 'N': 15, 'noise': True}
    Sim.extra_meshes_in = {'1':inner_interface, '2': inner_data}
    Sim.ins_domain_in = {'rmax': rI}


    Sim.domain_out = ([-rB,rB],[-rB,rB],[-rB,rB])
    Sim.PDE_out = Helmholtz()
    Sim.PDE_out.epsilon = inputs['epsilon_2']
    Sim.PDE_out.epsilon_G = inputs['epsilon_1']
    Sim.PDE_out.kappa = inputs['kappa']
    Sim.PDE_out.q = q_list 
    Sim.PDE_out.problem = inputs

    u_an = Sim.PDE_out.border_value(rB,0,0,rI)
    outer_interface = {'type':'I', 'value':None, 'fun':None, 'r':rI, 'N':15}
    outer_dirichlet = {'type':'D', 'value':u_an, 'fun':None, 'r':rB, 'N': 15}
    outer_data = {'type':'K', 'value':None, 'fun':lambda x,y,z: Sim.PDE_out.analytic(x,y,z), 'r':'Random', 'N': 15, 'noise': True}
    Sim.extra_meshes_out = {'1':outer_interface,'2':outer_dirichlet, '3': outer_data}
    Sim.ins_domain_out = {'rmax': rB,'rmin':rI}


    # Mesh
    Sim.mesh_in = {'N_r': 20,
                   'N_r_P': 20}
    Sim.mesh_out = {'N_r': 20,
                    'N_r_P': 20}

    # Neural Network
    Sim.weights = {
        'w_r': 1,
        'w_d': 1,
        'w_n': 1,
        'w_i': 1,
        'w_k': 1
    }

    Sim.lr = ([3000,6000],[1e-2,5e-3,5e-4])


    Sim.hyperparameters_in = {
                'input_shape': (None,3),
                'num_hidden_layers': 2,
                'num_neurons_per_layer': 12,
                'output_dim': 1,
                'activation': 'tanh',
                'architecture_Net': 'FCNN'
        }

    Sim.hyperparameters_out = {
                'input_shape': (None,3),
                'num_hidden_layers': 2,
                'num_neurons_per_layer': 12,
                'output_dim': 1,
                'activation': 'tanh',
                'architecture_Net': 'FCNN'
        }


    Sim.N_batches = 2

    adapt_weights = True
    adapt_w_iter = 15

    iters_save_model = 0
    Sim.folder_path = folder_path

    Sim.precondition = False
    N_precond = 10

    N_iters = 2


    Sim.setup_algorithm()

    # Solve
    Sim.solve_algorithm(N_iters = N_iters, 
                        precond = Sim.precondition, 
                        N_precond = N_precond, 
                        save_model = iters_save_model, 
                        adapt_weights = adapt_weights, 
                        adapt_w_iter = adapt_w_iter,
                        shuffle = True,
                        shuffle_iter=100)
    
    Sim.postprocessing(folder_path=folder_path)



if __name__=='__main__':
    main()


