import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from time import time
import os
import types
import json
import logging
import sys
import shutil

from BINet.PDE_Model import PDE_Model

from BINet.Mesh import Mesh
from NN.NeuralNet import PINN_NeuralNet

from NN.PINN import PINN
from NN.Postprocessing import View_results
from NN.PINN import PINN_Precond

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.config.run_functions_eagerly(True)

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

logger.info("> Starting PINN Algorithm")

domain = ([-1,1],[-1,1],[-1,1])
PDE = PDE_Model()
domain = PDE.set_domain(domain)
PDE.sigma = 0.04
PDE.epsilon = 1

lb = {'type':'D', 'value':-1/(4*np.pi*PDE.epsilon), 'fun':None, 'dr':None, 'r':1}
borders = {'1':lb}
ins_domain = {'rmax': 1}

PINN_solver = PINN()



PINN_solver.adapt_PDE(PDE)
weights = {
        'w_r': 1,
        'w_d': 1,
        'w_n': 1,
        'w_i': 1
}


mesh = Mesh(domain, N_b=60, N_r=1500)
mesh.create_mesh(borders, ins_domain)
PINN_solver.adapt_mesh(mesh,**weights)


lr = ([3000,6000],[1e-2,5e-3,5e-4])
hyperparameters_FCNN = {
        'input_shape': (None,3),
        'num_hidden_layers': 6,
        'num_neurons_per_layer': 30,
        'output_dim': 1,
        'activation': 'tanh',
        'architecture_Net': 'FCNN'
}

hyperparameters = hyperparameters_FCNN

PINN_solver.create_NeuralNet(PINN_NeuralNet,lr,**hyperparameters)

logger.info(json.dumps(hyperparameters, indent=4))
PINN_solver.model.summary()

logger.info("> Solving PINN")

N_iters = 2
PINN_solver.solve(N=N_iters)

Post = View_results(PINN_solver, save=True, directory=folder_path, data=True)

logger.info("> Ploting Solution")

Post.plot_loss_history();
Post.plot_u_plane();
Post.plot_u_domain_contour();

if Post.data:
        Post.close_file()

logger.info('================================================')
logger.info('================================================')


