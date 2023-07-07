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

from DCM.PDE_Model import Poisson as PDE_Model_IN
from DCM.PDE_Model import Helmholtz as PDE_Model_OUT
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


folder_path = os.path.join(main_path,'plots')
if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
os.makedirs(folder_path)

logger.info("> Starting PINN Algorithm")

domain_in = ([-1,1],[-1,1],[-1,1])
PDE_in = PDE_Model_IN()
domain_in = PDE_in.set_domain(domain_in)
PDE_in.sigma = 0.04
PDE_in.epsilon = 2
PDE_in.q = [(1,[0,0,0])]

lb = {'type':'I', 'value':None, 'fun':None, 'dr':None, 'r':1}
borders = {'1':lb}
ins_domain = {'rmax': 1, 'rmin':0}

mesh_in = Mesh(domain_in, N_b=60, N_r=1500)
mesh_in.create_mesh(borders, ins_domain)


domain_out = ([-10,10],[-10,10],[-10,10])
PDE_out = PDE_Model_OUT()
domain_out = PDE_out.set_domain(domain_out)
PDE_out.epsilon = 80
PDE_out.kappa = 0.125

u_an = np.exp(-PDE_out.kappa*(10-1))/(4*np.pi*PDE_out.epsilon*(1+PDE_out.kappa*1)*10)
lb = {'type':'I', 'value':None, 'fun':None, 'dr':None, 'r':1}
lb2 = {'type':'D', 'value':u_an, 'fun':None, 'dr':None, 'r':10}
borders = {'1':lb,'2':lb2}
ins_domain = {'rmax': 10,'rmin':1}

mesh_out = Mesh(domain_out, N_b=60, N_r=1500)
mesh_out.create_mesh(borders, ins_domain)

PDE = PDE_2_domains()
PDE.adapt_PDEs([PDE_in,PDE_out],[1,10])

XPINN_solver = XPINN(PINN)

XPINN_solver.adapt_PDEs(PDE)
weights = {
        'w_r': 1,
        'w_d': 1,
        'w_n': 1,
        'w_i': 1
}
XPINN_solver.adapt_meshes([mesh_in,mesh_out],[weights,weights])

lr = ([1500,3500],[1e-2,5e-3,5e-4])
hyperparameters = {
        'input_shape': (None,3),
        'num_hidden_layers': 4,
        'num_neurons_per_layer': 15,
        'output_dim': 1,
        'activation': 'tanh'
}

XPINN_solver.create_NeuralNets(PINN_NeuralNet,[lr,lr],[hyperparameters,hyperparameters])



logger.info(json.dumps(hyperparameters, indent=4))

logger.info("> Solving XPINN")

N_iters = 3
XPINN_solver.solve(N=N_iters)

Post = View_results_X(XPINN_solver, View_results, save=True, directory=folder_path, data=False)

logger.info("> Ploting Solution")

Post.plot_loss_history();
Post.plot_u_plane();
Post.plot_u_domain_contour();
#Post.plot_aprox_analytic();

if Post.data:
        Post.close_file()

logger.info('================================================')
logger.info('================================================')


