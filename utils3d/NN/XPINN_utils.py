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
        self.adapt_data_batches()

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
        

    ######################################################


    def adapt_data_batches(self):
        self.L_batches_solvers = list()
        for solver in self.solvers:
            L_batches = dict()
            for t in solver.mesh.meshes_names:
                L_batches[t] = solver.mesh.data_mesh[t]
            self.L_batches_solvers.append(L_batches)
        self.N_batches = solver.mesh.N_batches
        
    
    def shuffle_batches(self,shuffle):

        def iterate_shuffled_dataset(dataset):
            for batch in dataset:
                yield batch

        L_shuffled_solver = list()
        for set_batches,solver in zip(self.L_batches_solvers,self.solvers):
            L_shuffled = dict()
            for t in set_batches:
                if t in ('I','R','P'):
                    L = list()
                    for list_batches in set_batches[t][0]:
                        if shuffle:
                            L.append(iterate_shuffled_dataset(list_batches.shuffle(buffer_size=solver.mesh.meshes_N[t])))
                        else: 
                            L.append(iterate_shuffled_dataset(list_batches))
                    L_shuffled[t] = (L,)
                elif t in ('D','K','N'):
                    L1 = list()
                    for list_batches in set_batches[t][0]:
                        if shuffle:
                            L1.append(iterate_shuffled_dataset(list_batches.shuffle(buffer_size=solver.mesh.meshes_N[t])))
                        else:
                            L1.append(iterate_shuffled_dataset(list_batches))
                    L2 = list()
                    for list_batches in set_batches[t][1]:
                        if shuffle:
                            L2.append(iterate_shuffled_dataset(list_batches.shuffle(buffer_size=solver.mesh.meshes_N[t])))  
                        else:
                            L2.append(iterate_shuffled_dataset(list_batches))             
                    L_shuffled[t] = (L1,L2)
            L_shuffled_solver.append(L_shuffled)
        return L_shuffled_solver
    

    def get_batches(self,TX_b1,TX_b2):
        X_b = (dict(),dict()) 
        TX = (TX_b1,TX_b2)
        names = (self.solver1.mesh.meshes_names,self.solver2.mesh.meshes_names)
        for k in range(len(X_b)):
            for key in names[k]:
                X_b[k][key] = list()                      
                if key in ('I','R','P'):
                    N_doms = len(TX[k][key][0])  
                    xs = TX[k][key][0]
                    for i in range(N_doms):
                        x = xs[i]
                        X_b[k][key].append(next(x))
                elif key in ('D','N','K'): 
                    N_doms = len(TX[k][key][0])  
                    xs,us = TX[k][key]
                    for i in range(N_doms):
                        x = xs[i]
                        u = us[i]
                        X_b[k][key].append((next(x),next(u)))    
        return X_b


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

