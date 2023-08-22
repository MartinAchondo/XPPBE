import tensorflow as tf
import os
import logging
import shutil

from DCM.PDE_Model_Regularized import Helmholtz

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

    Sim = Simulation(Helmholtz)

    # PDE
    q_list = [(1000,[0,0,0])]

    inputs = {'Problem': 'Main',
              'rmin': 1,
              'rB': 10,
              'rI': 1,
              'epsilon_1':1,
              'epsilon_2':80,
              'kappa': 0.125,
              }
    
    Sim.problem = inputs
    Sim.q = q_list
    
    rB = inputs['rB']

    Sim.domain_in = ([-rB,rB],[-rB,rB],[-rB,rB])
    Sim.PDE_in = Helmholtz()
    Sim.PDE_in.sigma = 0.04
    Sim.PDE_in.epsilon = inputs['epsilon_1']
    Sim.PDE_in.epsilon_G = inputs['epsilon_1']
    Sim.PDE_in.q = q_list
    Sim.PDE_in.kappa = inputs['kappa']
    Sim.PDE_in.problem = inputs

    u_an = Sim.PDE_in.border_value(rB,0,0,rB)
    outer_dirichlet = {'type':'D', 'value':u_an, 'fun':None, 'dr':None, 'r':rB, 'N': 60}
    interior_known = {'type': 'K', 'fun': lambda x,y,z: Sim.PDE_in.analytic(x,y,z) , 'N': 12, 'value': None, 'dr': None, 'r': 5}

    Sim.borders_in = {'1':outer_dirichlet, '2': interior_known}
    Sim.ins_domain_in = {'rmax': rB}

    # Mesh
    Sim.mesh = {'N_r': 40,
                'N_r_P': 40}

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
                'architecture_Net': 'FCNN',
                'kernel_initializer': 'glorot_normal'
        }


    Sim.N_iters = 20
    Sim.precondition = True
    Sim.N_precond = 20
    Sim.N_batches = 20

    Sim.folder_path = folder_path

    Sim.setup_algorithm()

    # Solve
    Sim.solve_algorithm(N_iters=Sim.N_iters, precond=Sim.precondition, N_precond=Sim.N_precond, N_batches=Sim.N_batches, save_model=5)
    
    Sim.postprocessing(folder_path=folder_path)



if __name__=='__main__':
    main()


