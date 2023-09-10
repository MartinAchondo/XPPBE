import numpy as np
import tensorflow as tf
from time import time
from tqdm import tqdm as log_progress
import logging
import os
import pandas as pd

logger = logging.getLogger(__name__)

class XPINN_utils():

    def __init__(self):
        self.DTYPE = 'float32'
        self.loss_hist = list()

        self.loss_r1 = list()
        self.loss_bD1 = list()
        self.loss_bN1 = list()
        self.loss_bI1 = list()
        self.loss_bK1 = list()

        self.loss_r2 = list()
        self.loss_bD2 = list()
        self.loss_bN2 = list()
        self.loss_bI2 = list()
        self.loss_bK2 = list()

        self.iter = 0
        self.lr = None


    def adapt_PDEs(self,PDE):
        self.PDE = PDE
        for solver,pde,union in zip(self.solvers,PDE.PDEs,PDE.uns):
            solver.adapt_PDE(pde)
            solver.un = union

    def adapt_meshes(self,meshes,weights):
        for solver,mesh,weight in zip(self.solvers,meshes,weights):
            solver.adapt_mesh(mesh,**weight)

    def create_NeuralNets(self,NN_class,lrs,hyperparameters):
        for solver,lr,hyperparameter in zip(self.solvers,lrs,hyperparameters):
            solver.create_NeuralNet(NN_class,lr,**hyperparameter)

    def load_NeuralNets(self,dir_load,names,lrs):   
        for solver,lr,name in zip(self.solvers,lrs,names):
            solver.load_NeuralNet(dir_load,name,lr)  
        path_load = os.path.join(dir_load,'loss.csv')
        df = pd.read_csv(path_load)
        self.loss_hist = list(df['TL'])
        self.loss_r1 = list(df['R1'])
        self.loss_bD1 = list(df['D1'])
        self.loss_bN1 = list(df['N1'])
        self.loss_bI1 = list(df['I1'])
        self.loss_bK1 = list(df['K1'])
        self.loss_r2 = list(df['R2'])
        self.loss_bD2 = list(df['D2'])
        self.loss_bN2 = list(df['N2'])
        self.loss_bI2 = list(df['I2'])
        self.loss_bK2 = list(df['K2'])
        self.iter = len(self.loss_hist) 
        self.add_losses_NN()


    def save_models(self,dir_save,names):
        for solver,name in zip(self.solvers,names):
            solver.save_model(dir_save,name)  
        df_dict = {'TL': self.loss_hist,
                   'R1': list(map(lambda tensor: tensor.numpy(),self.loss_r1)),
                   'D1': list(map(lambda tensor: tensor.numpy(),self.loss_bD1)),
                   'N1': list(map(lambda tensor: tensor.numpy(),self.loss_bN1)),
                   'K1': list(map(lambda tensor: tensor.numpy(),self.loss_bK1)),
                   'I1': list(map(lambda tensor: tensor.numpy(),self.loss_bI1)),
                   'R2': list(map(lambda tensor: tensor.numpy(),self.loss_r2)),
                   'D2': list(map(lambda tensor: tensor.numpy(),self.loss_bD2)),
                   'N2': list(map(lambda tensor: tensor.numpy(),self.loss_bN2)),
                   'K2': list(map(lambda tensor: tensor.numpy(),self.loss_bK2)),
                   'I2': list(map(lambda tensor: tensor.numpy(),self.loss_bI2))
                }
        df = pd.DataFrame.from_dict(df_dict)
        path_save = os.path.join(dir_save,'loss.csv')
        df.to_csv(path_save)
        

    def create_batches(self, N_batches):

        number_batches = N_batches

        dataset_X_r_1 = tf.data.Dataset.from_tensor_slices(self.solver1.PDE.X_r)
        dataset_X_r_1 = dataset_X_r_1.shuffle(buffer_size=len(self.solver1.PDE.X_r))
        batch_size = int(len(self.solver1.PDE.X_r)/number_batches)
        batches_X_r_1 = dataset_X_r_1.batch(batch_size)

        dataset_X_r_2 = tf.data.Dataset.from_tensor_slices(self.solver2.PDE.X_r)
        dataset_X_r_2 = dataset_X_r_2.shuffle(buffer_size=len(self.solver2.PDE.X_r))
        batch_size = int(len(self.solver2.PDE.X_r)/number_batches)
        batches_X_r_2 = dataset_X_r_2.batch(batch_size)

        if not self.precondition:
            return (batches_X_r_1, batches_X_r_2), (None,None)

    
        number_batches = N_batches

        dataset_X_r_P_1 = tf.data.Dataset.from_tensor_slices(self.solver1.PDE.X_r_P)
        dataset_X_r_P_1 = dataset_X_r_P_1.shuffle(buffer_size=len(self.solver1.PDE.X_r_P))
        batch_size = int(len(self.solver1.PDE.X_r_P)/number_batches)
        batches_X_r_P_1 = dataset_X_r_P_1.batch(batch_size)
        
        dataset_X_r_P_2 = tf.data.Dataset.from_tensor_slices(self.solver2.PDE.X_r_P)
        dataset_X_r_P_2 = dataset_X_r_P_2.shuffle(buffer_size=len(self.solver2.PDE.X_r_P))
        batch_size = int(len(self.solver2.PDE.X_r_P)/number_batches)
        batches_X_r_P_2 = dataset_X_r_P_2.batch(batch_size)

        return (batches_X_r_1, batches_X_r_2), (batches_X_r_P_1,batches_X_r_P_2)



    def add_losses_NN(self):
        self.solver1.loss_r = self.loss_r1
        self.solver1.loss_bD = self.loss_bD1
        self.solver1.loss_bN = self.loss_bN1
        self.solver1.loss_bI = self.loss_bI1
        self.solver1.loss_bK = self.loss_bK1

        self.solver2.loss_r = self.loss_r2
        self.solver2.loss_bD = self.loss_bD2
        self.solver2.loss_bN = self.loss_bN2
        self.solver2.loss_bI = self.loss_bI2
        self.solver2.loss_bK = self.loss_bK2

