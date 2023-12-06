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
    
    DTYPE='float32'

    def __init__(self):
       
        self.loss_hist = list()

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
        w_k=1.0,
        w_e=1.0,
        w_q=1.0,
        w_g=1.0):

        self.w = {
            'R': float(w_r),
            'D': float(w_d),
            'N': float(w_n),
            'K': float(w_k),
            'Q': float(w_q),
            'I': float(w_i),
            'E': float(w_e),
            'G': float(w_g),
            'P': 1.0
        }

        self.w_hist = {
            'R': list(),
            'D': list(),
            'N': list(),
            'K': list(),
            'Q': list(),
            'I': list(),
            'E': list(),
            'G': list(),
            'P': list()
        }
        
        self.L_names = ['R','D','N','K','I','P','E','Q','G']
        

    def adapt_optimizer(self,optimizer,lr,lr_p=0.001):
        self.optimizer_name = optimizer
        self.lr = lr
        self.lr_p = lr_p

    def create_NeuralNet(self,NN_class,*args,**kwargs):
        self.model = NN_class(self.lb, self.ub,*args,**kwargs)
        self.model.build_Net()

    def load_NeuralNet(self,directory,name):
        path = os.path.join(os.getcwd(),directory,name)
        NN_model = tf.keras.models.load_model(path, compile=False)
        self.model = NN_model
        
        path_load = os.path.join(path,'w_hist.csv')
        df = pd.read_csv(path_load)

        for t in self.L_names:
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
 
