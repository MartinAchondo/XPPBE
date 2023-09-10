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
        w_r=1.0,
        w_d=1.0,
        w_n=1.0,
        w_i=1.0,
        w_k=1.0):

        logger.info("> Adapting Mesh")
        
        self.mesh = mesh
        self.lb = mesh.lb
        self.ub = mesh.ub
        self.PDE.adapt_PDE_mesh(self.mesh)

        self.w = {
            'R': float(w_r),
            'D': float(w_d),
            'N': float(w_n),
            'K': float(w_k),
            'I': float(w_i)
        }

        self.w_hist = {
            'R': list(),
            'D': list(),
            'N': list(),
            'K': list(),
            'I': list()
        }
        
        self.L_names = ['R','D','N','K','I']

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
 

        


