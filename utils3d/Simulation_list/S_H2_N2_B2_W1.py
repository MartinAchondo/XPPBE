import os
import logging
import shutil

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from DCM.PDE_Model import Poisson
from DCM.PDE_Model import Helmholtz
from DCM.PDE_Model import PBE_Interface

from Simulation_X import Simulation


main_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'results')
#if os.path.exists(main_path):
#        shutil.rmtree(main_path)
#os.makedirs(main_path)

folder_name = 'S_H2_N2_B2_W1'
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

    inputs = {'Problem': 'S_H2_N2_B2_W1',
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

    inner_interface = {'type':'I', 'value':None, 'fun':None, 'r':rI, 'N': 35}
    inner_data = {'type':'K', 'value':None, 'fun':lambda x,y,z: Sim.PDE_in.analytic(x,y,z), 'r':'Random', 'N': 35}
    Sim.extra_meshes_in = {'1':inner_interface} #, '2': inner_data}
    Sim.ins_domain_in = {'rmax': rI}


    Sim.domain_out = ([-rB,rB],[-rB,rB],[-rB,rB])
    Sim.PDE_out = Helmholtz()
    Sim.PDE_out.epsilon = inputs['epsilon_2']
    Sim.PDE_out.epsilon_G = inputs['epsilon_1']
    Sim.PDE_out.kappa = inputs['kappa']
    Sim.PDE_out.q = q_list 
    Sim.PDE_out.problem = inputs

    u_an = Sim.PDE_out.border_value(rB,0,0,rI)
    outer_interface = {'type':'I', 'value':None, 'fun':None, 'r':rI, 'N': 35}
    outer_dirichlet = {'type':'D', 'value':u_an, 'fun':None, 'r':rB, 'N': 35}
    outer_data = {'type':'K', 'value':None, 'fun':lambda x,y,z: Sim.PDE_out.analytic(x,y,z), 'r':'Random', 'N': 35}
    Sim.extra_meshes_out = {'1':outer_interface,'2':outer_dirichlet}#, '3': outer_data}
    Sim.ins_domain_out = {'rmax': rB,'rmin':rI}


    # Mesh
    Sim.mesh_in = {'N_r': 40}
    Sim.mesh_out = {'N_r': 40}

    # Neural Network
    Sim.weights = {
        'w_r': 1,
        'w_d': 1,
        'w_n': 1,
        'w_i': 1,
        'w_k': 1
    }

    Sim.lr = ([6000],[1e-3,1e-4])


    Sim.hyperparameters_in = {
                'input_shape': (None,3),
                'num_hidden_layers': 8,
                'num_neurons_per_layer': 200,
                'output_dim': 1,
                'activation': 'tanh',
                'architecture_Net': 'FCNN'
        }

    Sim.hyperparameters_out = {
                'input_shape': (None,3),
                'num_hidden_layers': 8,
                'num_neurons_per_layer': 200,
                'output_dim': 1,
                'activation': 'tanh',
                'architecture_Net': 'FCNN'
        }


    Sim.N_batches = 8

    Sim.adapt_weights = True
    Sim.adapt_w_iter = 1001

    Sim.iters_save_model = 500
    Sim.folder_path = folder_path

    Sim.precondition = False
    Sim.N_precond = 5

    Sim.N_iters = 3000


    Sim.setup_algorithm()

    # Solve
    Sim.solve_algorithm(N_iters=Sim.N_iters, 
                        precond=Sim.precondition, 
                        N_precond=Sim.N_precond, 
                        save_model=Sim.iters_save_model, 
                        adapt_weights=Sim.adapt_weights, 
                        adapt_w_iter=Sim.adapt_w_iter,
                        shuffle = True,
                        shuffle_iter=500)
    
    Sim.postprocessing(folder_path=folder_path)



if __name__=='__main__':
    main()

