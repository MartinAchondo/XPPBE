import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from time import time
from tqdm import tqdm as log_progress
import logging

logger = logging.getLogger(__name__)

class PINN():
    
    def __init__(self):

        self.DTYPE='float32'
       
        self.loss_hist = list()
        self.loss_r = list()
        self.loss_bD = list()
        self.loss_bN = list()
        self.loss_bK = list()
        self.loss_P = list()
        self.loss_bI = list()
        self.iter = 0
        self.lr = None
 

    def adapt_mesh(self, mesh,
        w_r=1,
        w_d=1,
        w_n=1,
        w_i=1,
        w_k=1):

        logger.info("> Adapting Mesh")
        
        self.mesh = mesh
        self.lb = mesh.lb
        self.ub = mesh.ub
        self.PDE.adapt_PDE_mesh(self.mesh)

        self.w_i = w_i
        self.w = {
            'r': w_r,
            'D': w_d,
            'N': w_n,
            'K': w_k
        }
        
        self.L_names = ['r','D','N', 'K']

        logger.info("Mesh adapted")
        

    def create_NeuralNet(self,NN_class,lr,*args,**kwargs):
        logger.info("> Creating NeuralNet")
        self.model = NN_class(self.mesh.lb, self.mesh.ub,*args,**kwargs)
        self.model.build_Net()
        self.lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay(*lr)
        logger.info("Neural Network created")

    def adapt_PDE(self,PDE):
        logger.info("> Adapting PDE")
        self.PDE = PDE
        logger.info("PDE adapted")

    def load_NeuralNet(self,directory,name,lr):
        logger.info("> Adapting NeuralNet")
        path = os.path.join(os.getcwd(),directory,name)
        NN_model = tf.keras.models.load_model(path, compile=False)
        self.model = NN_model
        self.lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay(*lr)
        logger.info("Neural Network adapted")

    def save_model(self,directory,name):
        dir_path = os.path.join(os.getcwd(),directory)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        self.model.save(os.path.join(dir_path,name))
 

    def loss_fn(self, X_batch, precond=False):
        if precond:
            L = self.PDE.get_loss_preconditioner(X_batch, self.model)
        else:
            L = self.PDE.get_loss(X_batch, self.model)
        loss = 0
        for t in self.L_names:
            loss += L[t]*self.w[t]
        return loss,L
        


