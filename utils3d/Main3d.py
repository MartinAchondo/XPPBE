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

from PDE_Model import PDE_Model
from PDE_Model import PDE_Model_2
from Preconditioner import Preconditioner
from Preconditioner import change_fun

from Mesh import Mesh
from NN.NeuralNet import PINN_NeuralNet

from NN.PINN import PINN
from NN.Postprocessing import View_results
from NN.PINN import PINN_Precond

from NN.XPINN import XPINN
from NN.Postprocessing import View_results_X
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'results_sim')
if os.path.exists(folder_path):
    shutil.rmtree(folder_path)

os.makedirs(folder_path)
filename = os.path.join(folder_path,'logfile.log')
LOG_format = '%(levelname)s - %(name)s: %(message)s'
logging.basicConfig(filename=filename, filemode='w', level=logging.INFO, format=LOG_format)
logger = logging.getLogger(__name__)

print("")
print("")

print("===================================================================")
print("")
print("")
print("> Starting PINN Algorithm")
logger.info("> Starting PINN Algorithm")
print("")
print("")


domain = ([-1,1],[-1,1],[-1,1])
PDE = PDE_Model()
domain = PDE.set_domain(domain)
PDE.sigma = 0.04
PDE.epsilon = 1

lb = {'type':'D', 'value':-1/(4*np.pi*PDE.epsilon), 'fun':None, 'dr':None, 'r':1}
borders = {'1':lb}
ins_domain = {'rmax': 1}

PINN_solver = PINN()



print("")
print("===================================================================")
print("")
print("> Adapting PDE")
logger.info("> Adapting PDE")
print("")
print("")

PINN_solver.adapt_PDE(PDE)
weights = {
        'w_r': 1,
        'w_d': 1,
        'w_n': 1,
        'w_i': 1
}


print("> Adapting Mesh")
logger.info("> Adapting Mesh")
print("")
print("")

mesh1 = Mesh(domain, N_b=60, N_r=1500)
mesh1.create_mesh(borders, ins_domain)
#mesh1.plot_points_2d();
PINN_solver.adapt_mesh(mesh1,**weights)



print("> Creating NeuralNet")
logger.info("> Creating NeuralNet")

lr = ([2500,4000],[1e-2,5e-3,5e-4])
hyperparameters_FCNN = {
        'input_shape': (None,3),
        'num_hidden_layers': 8,
        'num_neurons_per_layer': 20,
        'output_dim': 1,
        'activation': 'tanh',
        'architecture_Net': 'FCNN'
}
hyperparameters_ResNet = {
        'input_shape': (None,3),
        'num_hidden_blocks': 5,
        'num_neurons_per_layer': 20,
        'output_dim': 1,
        'activation': 'tanh',
        'architecture_Net': 'ResNet'
}
 
hyperparameters = hyperparameters_ResNet

PINN_solver.create_NeuralNet(PINN_NeuralNet,lr,**hyperparameters)

print("")
print(json.dumps(hyperparameters, indent=4))
logger.info(json.dumps(hyperparameters, indent=4))
print("")
PINN_solver.model.summary()
print("")
print("")
#print("Parameters")



print("")
print("")
print("> Solving PINN")
logger.info("> Solving PINN")
print("")
print("")

N_iters = 10
PINN_solver.solve(N=N_iters)




# model_save_name = 'Test_PINN'
# print("")
# print(f'> Saving model: {model_save_name}')
# print("")
# PINN_solver.save_model('.saved_models',model_save_name) 



Post = View_results(PINN_solver, save=True, directory=folder_path, data=True)

print("")
print(f'Loss: {Post.loss_last}')
print('')
print("> Ploting Solution")
logger.info("> Ploting Solution")
print("")

Post.plot_loss_history();

Post.plot_u_plane();

if Post.data:
        Post.close_file()

print('')
print('')
print('')
