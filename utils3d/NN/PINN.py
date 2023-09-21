import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
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
 
    def adapt_PDE(self,PDE):
        self.PDE = PDE
        self.adapt_mesh()

    def adapt_mesh(self):
        self.mesh = self.PDE.mesh
        self.lb = tf.constant(self.mesh.lb, dtype=self.DTYPE)
        self.ub = tf.constant(self.mesh.ub, dtype=self.DTYPE)

    def adapt_weights(self,
        w_r=1.0,
        w_d=1.0,
        w_n=1.0,
        w_i=1.0,
        w_k=1.0):

        self.w = {
            'R': float(w_r),
            'D': float(w_d),
            'N': float(w_n),
            'K': float(w_k),
            'I': float(w_i),
            'P': 1.0,
        }

        self.w_hist = {
            'R': list(),
            'D': list(),
            'N': list(),
            'K': list(),
            'I': list(),
            'P': list()
        }
        
        self.L_names = ['R','D','N','K','I','P']
        
 
    def create_NeuralNet(self,NN_class,lr,*args,**kwargs):
        self.model = NN_class(self.lb, self.ub,*args,**kwargs)
        self.model.build_Net()
        self.lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay(*lr)


    def load_NeuralNet(self,directory,name,lr):
        logger.info("> Adapting NeuralNet")
        path = os.path.join(os.getcwd(),directory,name)
        NN_model = tf.keras.models.load_model(path, compile=False)
        self.model = NN_model
        self.lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay(*lr)
        logger.info("Neural Network adapted")
        
        path_load = os.path.join(path,'w_hist.csv')
        df = pd.read_csv(path_load)

        for t in self.mesh.meshes_names:
            self.w_hist[t] = list(df[t])
            self.w[t] = self.w_hist[t][-1]
       

    def save_model(self,directory,name):
        dir_path = os.path.join(os.getcwd(),directory)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        self.model.save(os.path.join(dir_path,name))

        df_dict = pd.DataFrame.from_dict(self.w_hist)
        df = pd.DataFrame.from_dict(df_dict)
        path_save = os.path.join(os.path.join(dir_path,name),'w_hist.csv')
        df.to_csv(path_save)
 
