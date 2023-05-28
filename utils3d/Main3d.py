import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from time import time
import math
import os
import types
import json

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


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


print("")
print("")

print("===================================================================")
print("")
print("")
print("> Start PINN Algorithm")
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
print("")
print("")

mesh1 = Mesh(domain, N_b=60, N_r=1500)
mesh1.create_mesh(borders, ins_domain)
#mesh1.plot_points_2d();
PINN_solver.adapt_mesh(mesh1,**weights)





lr = ([2500,4000],[1e-2,5e-3,5e-4])
hyperparameters = {
        'input_shape': (None,3),
        'num_hidden_layers': 8,
        'num_neurons_per_layer': 20,
        'output_dim': 1,
        'activation': 'tanh',
        'architecture_Net': 'FFNN'
}
hyperparameters = {
        'input_shape': (None,3),
        'num_hidden_blocks': 5,
        'num_neurons_per_layer': 20,
        'output_dim': 1,
        'activation': 'tanh',
        'architecture_Net': 'ResNet'
}

PINN_solver.create_NeuralNet(PINN_NeuralNet,lr,**hyperparameters)

print("> Creating NeuralNet")
print("")
print(json.dumps(hyperparameters, indent=4))
print("")
PINN_solver.model.summary()
print("")
print("")
#print("Parameters")



print("")
print("")
print("> Solving PINN")
print("")
print("")

N_iters = 10
PINN_solver.solve(N=N_iters)




# model_save_name = 'Test_PINN'
# print("")
# print(f'> Saving model: {model_save_name}')
# print("")
# PINN_solver.save_model('.saved_models',model_save_name) 



Post = View_results(PINN_solver)

print("")
print(f'Loss: {Post.loss_last}')
print('')
print("> Ploting Solution")
print("")

Post.plot_loss_history();

Post.plot_u_plane();

print('')
print('')
print('')
