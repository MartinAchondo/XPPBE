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
        self.L_X_solvers = list()
        for solver in self.solvers:
            L_batches = dict()
            for t in solver.mesh.meshes_names:
                L_batches[t] = solver.mesh.data_mesh[t]
            self.L_X_solvers.append(L_batches)
        self.N_batches = solver.mesh.N_batches
        
    
    def create_generators_shuffle(self, shuffle):

        def generator(dataset):
            for batch in dataset:
                yield batch

        L_X_generators = list()
        for set_batches,solver in zip(self.L_X_solvers,self.solvers):
            L = dict()
            for t in set_batches:
                if shuffle:
                    L[t] = generator(set_batches[t].shuffle(buffer_size=solver.mesh.meshes_N[t]))
                elif not shuffle:
                    L[t] = generator(set_batches[t])
            L_X_generators.append(L)
        return L_X_generators


    def get_batches(self, TX_b):
        X_b = dict()
        for t in TX_b:
            X_b[t] = next(TX_b[t])
        return X_b


    def checkers_iterations(self):

        if (self.shuffle and self.iter%self.shuffle_iter ==0):
            self.shuffle_now = True
        else:
            self.shuffle_now = False

        if self.iter>=self.N_precond and self.precondition:
            self.precondition = False
        
        if self.iter % 2 == 0:
            self.pbar.set_description("Loss: {:6.4e}".format(self.current_loss))

        return [0.0, dict()],[0.0, dict()]



    def callback(self, L1,L2):

        self.loss_r1.append(L1[1]['R'])
        self.loss_bD1.append(L1[1]['D'])
        self.loss_bN1.append(L1[1]['N'])
        self.loss_bI1.append(L1[1]['I'])
        self.loss_bK1.append(L1[1]['K'])

        self.loss_r2.append(L2[1]['R'])
        self.loss_bD2.append(L2[1]['D'])
        self.loss_bN2.append(L2[1]['N'])
        self.loss_bI2.append(L2[1]['I'])
        self.loss_bK2.append(L2[1]['K'])

        loss = L1[0] + L2[0]
        self.current_loss = loss.numpy()
        self.loss_hist.append(self.current_loss)
        self.solver1.L = L1[1]
        self.solver2.L = L2[1]

        for solver in self.solvers:
            for t in solver.L_names:
                solver.w_hist[t].append(solver.w[t])

        self.iter+=1

        if self.save_model_iter > 0:
            if self.iter % self.save_model_iter == 0 and self.iter>1:
                dir_save = os.path.join(self.folder_path,f'iter_{self.iter}')
                self.save_models(dir_save, [f'model_1',f'model_2'])



    def batch_iter_callback(self,L,L_b):
        self.N_steps +=1
        L_f = list()
        for Li,Li_b in zip(L,L_b):
            Li[0] += Li_b[0]
            for t in Li_b[1]:
                if t in Li[1]:
                    Li[1][t] += Li_b[1][t]
                else:
                    Li[1][t] = Li_b[1][t]
            L_f.append(Li)
        return L_f


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

