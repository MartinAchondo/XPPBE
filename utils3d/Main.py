import tensorflow as tf
import numpy as np
from time import time
import os
import json
import logging
import shutil

from DCM.PDE_Model import Poisson

from DCM.Mesh import Mesh
from NN.NeuralNet import PINN_NeuralNet

from NN.PINN import PINN
from DCM.Postprocessing import View_results


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

main_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'results')
if os.path.exists(main_path):
        shutil.rmtree(main_path)
os.makedirs(main_path)

filename = os.path.join(main_path,'logfile.log')
LOG_format = '%(levelname)s - %(name)s: %(message)s'
logging.basicConfig(filename=filename, filemode='w', level=logging.INFO, format=LOG_format)
logger = logging.getLogger(__name__)

logger.info("==============================")


folder_path = os.path.join(main_path,'test')
if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
os.makedirs(folder_path)




logger.info('================================================')


class Simulation():
    
    def __init__(self):
          pass

    def setup_algorithm(self):
        logger.info("> Starting PINN Algorithm")

        
        PDE = self.PDE
        domain = PDE.set_domain(self.domain)

        self.PINN_solver = PINN()

        self.PINN_solver.adapt_PDE(PDE)


        mesh = Mesh(domain, N_b=self.N_b, N_r=self.N_r)
        mesh.create_mesh(self.borders, self.ins_domain)
        self.PINN_solver.adapt_mesh(mesh,**self.weights)

        hyperparameters = self.hyperparameters
        self.PINN_solver.create_NeuralNet(PINN_NeuralNet,self.lr,**hyperparameters)

        logger.info(json.dumps(hyperparameters, indent=4))
        self.PINN_solver.model.summary()


    def solve_algorithm(self,N_iters):
        logger.info("> Solving PINN")

        self.PINN_solver.solve(N=N_iters)


    def postprocessing(self):
        
        Post = View_results(self.PINN_solver, save=True, directory=folder_path, data=True)

        logger.info("> Ploting Solution")

        Post.plot_loss_history();
        Post.plot_u_plane();
        Post.plot_u_domain_contour();
        Post.plot_aprox_analytic();

        if Post.data:
            Post.close_file()

        logger.info('================================================')



# Inputs
###############################################


def main():

    Sim = Simulation()

    # PDE
    Sim.domain = ([-1,1],[-1,1],[-1,1])
    Sim.PDE = Poisson()
    Sim.PDE.sigma = 0.04
    Sim.PDE.epsilon = 1
    Sim.PDE.q = [(1,[0,0,0])]

    lb = {'type':'D', 'value':-1/(4*np.pi*Sim.PDE.epsilon), 'fun':None, 'dr':None, 'r':1}
    Sim.borders = {'1':lb}
    Sim.ins_domain = {'rmax': 1}

    # Mesh
    Sim.N_b = 60
    Sim.N_r = 1500

    # Neural Network
    Sim.weights = {
        'w_r': 1,
        'w_d': 1,
        'w_n': 1,
        'w_i': 1
    }

    Sim.lr = ([3000,6000],[1e-2,5e-3,5e-4])
    Sim.hyperparameters = {
                'input_shape': (None,3),
                'num_hidden_layers': 6,
                'num_neurons_per_layer': 30,
                'output_dim': 1,
                'activation': 'tanh',
                'architecture_Net': 'FCNN'
        }

    Sim.setup_algorithm()

    # Solve
    Sim.solve_algorithm(N_iters=10)
    
    Sim.postprocessing()



if __name__=='__main__':
    main()


