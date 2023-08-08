import tensorflow as tf
import os
import logging
import shutil

from DCM.PDE_Model import Poisson
from DCM.PDE_Model import Helmholtz
from DCM.PDE_Model import PBE_Interface

from Simulation import Simulation

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


main_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'results')
if os.path.exists(main_path):
        shutil.rmtree(main_path)
os.makedirs(main_path)

folder_name = 'data'
folder_path = os.path.join(main_path,folder_name)
folder_path = main_path
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
              'epsilon_1':2,
              'epsilon_2':10,
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

    lb = {'type':'I', 'value':None, 'fun':None, 'dr':None, 'r':rI}
    Sim.borders_in = {'1':lb}
    Sim.ins_domain_in = {'rmax': rI}


    Sim.domain_out = ([-rB,rB],[-rB,rB],[-rB,rB])
    Sim.PDE_out = Helmholtz()
    Sim.PDE_out.epsilon = inputs['epsilon_2']
    Sim.PDE_out.epsilon_G = inputs['epsilon_1']
    Sim.PDE_out.kappa = inputs['kappa']
    Sim.PDE_out.q = q_list 

    u_an = Sim.PDE_out.border_value(rB,0,0,rI)
    lb = {'type':'I', 'value':None, 'fun':None, 'dr':None, 'r':rI}
    lb2 = {'type':'D', 'value':u_an, 'fun':None, 'dr':None, 'r':rB}
    
    Sim.borders_out = {'1':lb,'2':lb2}
    Sim.ins_domain_out = {'rmax': rB,'rmin':rI}


    # Mesh
    Sim.mesh = {'N_b': 60,
                'N_r': 1500}

    # Neural Network
    Sim.weights = {
        'w_r': 1,
        'w_d': 1,
        'w_n': 1,
        'w_i': 1
    }

    Sim.lr = ([3000,6000],[1e-2,5e-3,5e-4])

    
    Sim.hyperparameters_in = {
                'input_shape': (None,3),
                'num_hidden_layers': 4,
                'num_neurons_per_layer': 100,
                'output_dim': 1,
                'activation': 'ReLU',
                'architecture_Net': 'FCNN'
        }

    Sim.hyperparameters_out = {
                'input_shape': (None,3),
                'num_hidden_layers': 4,
                'num_neurons_per_layer': 100,
                'output_dim': 1,
                'activation': 'tanh',
                'architecture_Net': 'FCNN'
        }



    Sim.setup_algorithm()

    # Solve
    N_iters = 15000
    Sim.solve_algorithm(N_iters=N_iters)
    
    Sim.postprocessing(folder_path=folder_path)



if __name__=='__main__':
    main()


