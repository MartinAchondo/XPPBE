import numpy as np
import tensorflow as tf
from time import time
from tqdm import tqdm as log_progress
import logging
import os
import pandas as pd


logger = logging.getLogger(__name__)

class XPINN_utils():

    DTYPE='float32'

    def __init__(self):

        self.loss_hist = list()

        self.loss_r1 = list()
        self.loss_bD1 = list()
        self.loss_bN1 = list()
        self.loss_bK1 = list()
        self.loss_bQ = list()
        self.loss_P1 = list()

        self.loss_r2 = list()
        self.loss_bD2 = list()
        self.loss_bN2 = list()
        self.loss_bK2 = list()
        self.loss_P2 = list()
        
        self.loss_G = list()
        self.loss_Iu = list()
        self.loss_Id = list()
        self.loss_exp = list()

        self.G_solv_hist = dict()
        self.G_solv_hist['0'] = 0.0

        self.iter = 0
        self.lr = None

    def adapt_PDEs(self,PDE):
        self.PDE = PDE
        self.mesh = self.PDE.mesh
        for solver,pde in zip(self.solvers, PDE.get_PDEs):
            solver.adapt_PDE(pde)
        self.adapt_datasets()
        self.set_mesh_names()

    def adapt_weights(self,weights,adapt_weights=False,adapt_w_iter=2000,adapt_w_method='gradients',alpha=0.3):
        self.adapt_weights = adapt_weights
        self.adapt_w_iter = adapt_w_iter
        self.adapt_w_method = adapt_w_method
        self.alpha_w = alpha

        for solver,weight in zip(self.solvers,weights):
            solver.adapt_weights(**weight) 

    def create_NeuralNets(self,NN_class,hyperparameters):
        for solver,hyperparameter in zip(self.solvers,hyperparameters):
            solver.create_NeuralNet(NN_class,**hyperparameter)

    def adapt_optimizers(self,optimizer,lrs,lr_p=0.001):
        for solver,lr in zip(self.solvers,lrs):
            solver.adapt_optimizer(optimizer,lr,lr_p)
        self.optimizer_name = optimizer

    def set_points_methods(self, sample_method='batches', N_batches=1, sample_size=1000):
        self.sample_method = sample_method
        self.N_batches = N_batches
        self.sample_size = sample_size

    def load_NeuralNets(self,dir_load,names):   
        for solver,name in zip(self.solvers,names):
            solver.adapt_weights()
            solver.load_NeuralNet(dir_load,name)  
        path_load = os.path.join(dir_load,'loss.csv')
        df = pd.read_csv(path_load)
        self.loss_hist = list(df['TL'])
        self.loss_r1 = list(df['R1'])
        self.loss_bD1 = list(df['D1'])
        self.loss_bN1 = list(df['N1'])
        self.loss_bK1 = list(df['K1'])
        self.loss_bQ = list(df['Q'])
        self.loss_r2 = list(df['R2'])
        self.loss_bD2 = list(df['D2'])
        self.loss_bN2 = list(df['N2'])
        self.loss_bK2 = list(df['K2'])
        self.loss_G = list(df['G'])
        self.loss_Iu = list(df['Iu'])
        self.loss_Id = list(df['Id'])
        self.loss_exp = list(df['E'])
        self.loss_P1 = list(df['P1'])
        self.loss_P2 = list(df['P2'])
        self.iter = len(self.loss_hist) 
        self.add_losses_NN()

        path_load = os.path.join(dir_load,'G_solv.csv')
        df_2 = pd.read_csv(path_load)
        for iters,G_solv in zip(list(df_2['iter']),list(df_2['G_solv'])):
            self.G_solv_hist[str(iters)] = G_solv

    def save_models(self,dir_save,names):
        for solver,name in zip(self.solvers,names):
            solver.save_model(dir_save,name)  
        df_dict = {'TL': self.loss_hist,
                   'R1': list(map(lambda tensor: tensor.numpy(),self.loss_r1)),
                   'D1': list(map(lambda tensor: tensor.numpy(),self.loss_bD1)),
                   'N1': list(map(lambda tensor: tensor.numpy(),self.loss_bN1)),
                   'K1': list(map(lambda tensor: tensor.numpy(),self.loss_bK1)),
                   'Q': list(map(lambda tensor: tensor.numpy(),self.loss_bQ)),
                   'R2': list(map(lambda tensor: tensor.numpy(),self.loss_r2)),
                   'D2': list(map(lambda tensor: tensor.numpy(),self.loss_bD2)),
                   'N2': list(map(lambda tensor: tensor.numpy(),self.loss_bN2)),
                   'K2': list(map(lambda tensor: tensor.numpy(),self.loss_bK2)),
                   'Iu': list(map(lambda tensor: tensor.numpy(),self.loss_Iu)),
                   'Id': list(map(lambda tensor: tensor.numpy(),self.loss_Id)),
                   'G': list(map(lambda tensor: tensor.numpy(),self.loss_G)),
                   'E': list(map(lambda tensor: tensor.numpy(),self.loss_exp)),
                   'P1': list(map(lambda tensor: tensor.numpy(),self.loss_P1)),
                   'P2': list(map(lambda tensor: tensor.numpy(),self.loss_P2))
                }
        df = pd.DataFrame.from_dict(df_dict)
        path_save = os.path.join(dir_save,'loss.csv')
        df.to_csv(path_save)

        df_2 = pd.DataFrame(self.G_solv_hist.items())
        df_2.columns = ['iter','G_solv']
        path_save = os.path.join(dir_save,'G_solv.csv')
        df_2.to_csv(path_save)        

    ######################################################

    def set_mesh_names(self):
        for solver in self.solvers:
            solver.Mesh_names = solver.mesh.solver_mesh_names.union(self.mesh.domain_mesh_names)
        if 'E' in self.solver1.Mesh_names:
            self.solver1.Mesh_names.remove('E')

    def adapt_datasets(self):
        self.L_X_solvers = list()
        for solver in self.solvers:
            L_batches = dict()
            for t in solver.mesh.solver_mesh_names:
                L_batches[t] = solver.mesh.solver_mesh_data[t]
            self.L_X_solvers.append(L_batches)

        self.L_X_domain = dict()
        for t in self.mesh.domain_mesh_names:
            if t in ('Iu','Id'):
                self.L_X_domain['I'] = self.mesh.domain_mesh_data['I']
            else:
                self.L_X_domain[t] = self.mesh.domain_mesh_data[t]

    # Full batch
    def get_all_batches(self):
        ((TX_b1, TX_b2),TX_d) = self.create_generators()
        X_b1 = self.get_batches(TX_b1)
        X_b2 = self.get_batches(TX_b2)   
        X_d = self.get_batches(TX_d) 
        return X_b1,X_b2,X_d

    def create_generators(self):
        def generator(dataset):
            for batch in dataset:
                yield batch
        L_SV = list()
        for set_batches in self.L_X_solvers:
            L = dict()
            for t in set_batches:
                new_dataset = set_batches[t]
                L[t] = generator(self.create_batches(new_dataset,self.N_batches))
            L_SV.append(L)
        L_D = dict()
        for t in self.L_X_domain:
            if t in ('E'):
                L_D[t] = self.L_X_domain[t]
            elif t in ('I'):
                new_dataset = self.L_X_domain[t]
                L_D[t] = generator(self.create_batches(new_dataset,self.N_batches))
        return L_SV,L_D
    
    def create_batches(self, dataset, num_batches=1):
        batch_size = int(dataset.cardinality().numpy()/num_batches)
        batches = dataset.batch(batch_size=batch_size)
        return batches

    def get_batches(self, TX_b):
        X_b = dict()
        for t in TX_b:
            if t in ('E'):
                X_b[t] = TX_b[t]
            else:
                X_b[t] = next(TX_b[t])
        return X_b


    #utils
    def checkers_iterations(self):

        # solvation energy
        if (self.iter+1)%self.G_solv_iter==0 and self.iter>1:
            self.calc_Gsolv_now = True
        else:
            self.calc_Gsolv_now = False

        # shuffle batches
        if self.shuffle and self.iter%self.shuffle_iter==0 and self.iter>1:
            self.shuffle_now = True
        else:
            self.shuffle_now = False

        # adapt losses weights
        if self.adapt_weights and self.iter%self.adapt_w_iter==0 and self.iter>1 and not self.precondition:
            self.adapt_w_now = True
        else:
            self.adapt_w_now = False  

        # check precondition
        if self.iter>=self.N_precond and self.precondition:
            self.precondition = False
            for data,solver in zip(self.L_X_solvers,self.solvers):
                del data['P']
                solver.mesh.solver_mesh_data['P'] = None
                solver.mesh.solver_mesh_names.remove('P')
                solver.Mesh_names.remove('P')
        
        if self.iter % 2 == 0:
            self.pbar.set_description("Loss: {:6.4e}".format(self.current_loss))                

    def callback(self, L1,L2):

        self.loss_r1.append(L1[1]['R'])
        self.loss_bD1.append(L1[1]['D'])
        self.loss_bN1.append(L1[1]['N'])
        self.loss_bK1.append(L1[1]['K'])
        self.loss_bQ.append(L1[1]['Q'])

        self.loss_r2.append(L2[1]['R'])
        self.loss_bD2.append(L2[1]['D'])
        self.loss_bN2.append(L2[1]['N'])
        self.loss_bK2.append(L2[1]['K'])

        self.loss_Iu.append(L1[1]['Iu'])
        self.loss_Id.append(L1[1]['Id'])
        self.loss_G.append(L1[1]['G'])
        self.loss_exp.append(L1[1]['E'])

        self.loss_P1.append(L1[1]['P'])
        self.loss_P2.append(L2[1]['P'])

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


    def add_losses_NN(self):
        self.solver1.loss_r = self.loss_r1
        self.solver1.loss_bD = self.loss_bD1
        self.solver1.loss_bN = self.loss_bN1
        self.solver1.loss_bK = self.loss_bK1
        self.solver1.loss_bQ = self.loss_bQ

        self.solver2.loss_r = self.loss_r2
        self.solver2.loss_bD = self.loss_bD2
        self.solver2.loss_bN = self.loss_bN2
        self.solver2.loss_bK = self.loss_bK2

        losses1 = [self.loss_r1,self.loss_bD1,self.loss_bN1,self.loss_bK1,self.loss_bQ,self.loss_G,self.loss_Iu,self.loss_Id,self.loss_P1]

        losses2 = [self.loss_r2,self.loss_bD2,self.loss_bN2,self.loss_bK2,self.loss_G,self.loss_Iu,self.loss_Id,self.loss_exp,self.loss_P2]
        
        self.solver1.loss_hist = [0]*len(losses1[0])
        for subloss in losses1:
            self.solver1.loss_hist = [a + b for a, b in zip(self.solver1.loss_hist, subloss)]

        self.solver2.loss_hist = [0]*len(losses2[0])
        for subloss in losses2:
            self.solver2.loss_hist = [a + b for a, b in zip(self.solver2.loss_hist, subloss)]



