import tensorflow as tf
import numpy as np
from time import time
import os
import json
import logging
import shutil

from DCM.PDE_Model import Poisson
from DCM.PDE_Model import Helmholtz
from DCM.PDE_Model import PDE_2_domains


from DCM.Mesh import Mesh
from NN.NeuralNet import PINN_NeuralNet

from NN.PINN import PINN
from NN.XPINN import XPINN

from DCM.Postprocessing import View_results
from DCM.Postprocessing import View_results_X



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

        
        PDE_in = self.PDE_in
        domain_in = PDE_in.set_domain(self.domain_in)

        PDE_out = self.PDE_out
        domain_out = PDE_out.set_domain(self.domain_out)


        mesh_in = Mesh(domain_in, N_b=self.N_b, N_r=self.N_r)
        mesh_in.create_mesh(self.borders_in, self.ins_domain_in)
        
        mesh_out = Mesh(domain_out, N_b=self.N_b, N_r=self.N_r)
        mesh_out.create_mesh(self.borders_out, self.ins_domain_out)

        PDE = PDE_2_domains()
        PDE.adapt_PDEs([PDE_in,PDE_out],[PDE_in.epsilon,PDE_out.epsilon])

        self.XPINN_solver = XPINN(PINN)

        self.XPINN_solver.adapt_PDEs(PDE)

        self.XPINN_solver.adapt_meshes([mesh_in,mesh_out],[self.weights,self.weights])

        hyperparameters = self.hyperparameters
        self.XPINN_solver.create_NeuralNets(PINN_NeuralNet,[self.lr,self.lr],[hyperparameters,hyperparameters])


        logger.info(json.dumps(hyperparameters, indent=4))


    def solve_algorithm(self,N_iters):
        logger.info("> Solving XPINN")

        self.XPINN_solver.solve(N=N_iters)


    def postprocessing(self):
        
        Post = View_results_X(self.XPINN_solver, View_results, save=True, directory=folder_path, data=False)

        logger.info("> Ploting Solution")

        Post.plot_loss_history();
        Post.plot_u_plane();
        Post.plot_u_domain_contour();
        #Post.plot_aprox_analytic();

        if Post.data:
            Post.close_file()

        logger.info('================================================')



# Inputs
###############################################


def main():

    Sim = Simulation()

    # PDE
    Sim.domain_in = ([-1,1],[-1,1],[-1,1])
    Sim.PDE_in = Poisson()
    Sim.PDE_in.sigma = 0.04
    Sim.PDE_in.epsilon = 2
    Sim.PDE_in.q = [(1,[0,0,0])]

    lb = {'type':'D', 'value':-1/(4*np.pi*Sim.PDE_in.epsilon), 'fun':None, 'dr':None, 'r':1}
    Sim.borders_in = {'1':lb}
    Sim.ins_domain_in = {'rmax': 1}


    Sim.domain_out = ([-1,1],[-1,1],[-1,1])
    Sim.PDE_out = Helmholtz()
    Sim.PDE_out.epsilon = 1
    Sim.PDE_out.kappa = 0.125

    u_an = np.exp(-Sim.PDE_out.kappa*(10-1))/(4*np.pi*Sim.PDE_out.epsilon*(1+Sim.PDE_out.kappa*1)*10)
    lb = {'type':'I', 'value':None, 'fun':None, 'dr':None, 'r':1}
    lb2 = {'type':'D', 'value':u_an, 'fun':None, 'dr':None, 'r':10}
    
    Sim.borders_out = {'1':lb,'2':lb2}
    Sim.ins_domain_out = {'rmax': 10,'rmin':1}


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


