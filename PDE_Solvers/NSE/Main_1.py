import tensorflow as tf
import os
import logging
import shutil
import numpy as np

from DCM.PDE_Model import Navier_Stokes

from Simulation_1 import Simulation

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


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

    Sim = Simulation(Navier_Stokes)

    # PDE
    q_list = [(1000,[0,0,0])]

    inputs = {'Problem': 'Main',
              'xmin': 0,
              'xmax': np.pi*2,
              'ymin': 0,
              'ymax': np.pi*2,
              'tmin': 0,
              'tmax': 1,
              'rho': 1,
              'mu': 1
              }
    
    Sim.problem = inputs
   
    xmin = inputs['xmin']
    xmax = inputs['xmax']
    ymin = inputs['ymin']
    ymax = inputs['ymax']
    tmin = inputs['tmin']
    tmax = inputs['tmax']

    Sim.domain_in = ([-xmin,xmax],[-ymin,ymax],[-tmin,tmax])
    Sim.PDE_in = Navier_Stokes()
    Sim.PDE_in.rho = inputs['rho']
    Sim.PDE_in.mu = inputs['mu']
    Sim.PDE_in.problem = inputs

    down = {'type':'D', 'fun':lambda x,y,t,n: Sim.PDE_in.initial(x,ymin,t,n), 'value':None, 'xmin': xmin, 'xmax': xmax, 'y': ymin, 'r':'x', 'N': 10, 'dr': None}
    up = {'type':'D', 'fun':lambda x,y,t,n: Sim.PDE_in.initial(x,ymax,t,n), 'value':None, 'xmin': xmin, 'xmax': xmax, 'y': ymax, 'r':'x', 'N': 10, 'dr': None}
    left = {'type':'D', 'fun':lambda x,y,t,n: Sim.PDE_in.initial(xmin,y,t,n), 'value':None, 'ymin': ymin, 'ymax': ymax, 'x': xmin, 'r':'y', 'N': 10, 'dr': None}
    right = {'type':'D', 'fun':lambda x,y,t,n: Sim.PDE_in.initial(xmax,y,t,n), 'value':None, 'ymin': ymin, 'ymax': ymax, 'x': xmax, 'r':'y', 'N': 10, 'dr': None}

    initial = {'type': '0', 'fun': lambda x,y,t,n: Sim.PDE_in.initial(x,y,0,n), 'N': 30, 'value': None, 'dr': None, 'r': 5}

    Sim.borders_in = {'1':down,
                      '2': up,
                      '3': left,
                      '4': right,
                      '5': initial}

    # Mesh
    Sim.mesh = {'N_r': 30,
                'N_r_P': 5}

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
                'output_dim': 3,
                'activation': 'tanh',
                'architecture_Net': 'FCNN',
                'kernel_initializer': 'glorot_normal'
        }


    Sim.N_iters = 20
    Sim.N_batches = 1
    Sim.precondition = True
    Sim.N_precond = -1
    Sim.iters_save_model = 10

    Sim.folder_path = folder_path

    Sim.setup_algorithm()

    # Solve
    Sim.solve_algorithm(N_iters=Sim.N_iters, precond=Sim.precondition, N_precond=Sim.N_precond, N_batches=Sim.N_batches, save_model=Sim.iters_save_model)
    
    Sim.postprocessing(folder_path=folder_path)



if __name__=='__main__':
    main()


