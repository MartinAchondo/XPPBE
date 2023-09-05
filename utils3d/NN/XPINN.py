import numpy as np
import tensorflow as tf
from time import time
from tqdm import tqdm as log_progress
import logging
import os
import pandas as pd

logger = logging.getLogger(__name__)


class XPINN():
    
    def __init__(self, PINN):

        self.DTYPE = 'float32'

        self.solver1, self.solver2 = PINN(), PINN()
        self.solvers = [self.solver1,self.solver2]
        
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
        
    
    def get_loss(self,X_batch, s1,s2, precond):
        loss,L = s1.loss_fn(X_batch,precond=precond)
        if not precond:    
            loss_I = self.PDE.loss_I(s1,s2)
            L['I'] = loss_I
            loss += s1.w_i*loss_I
        return loss,L


    def get_grad(self,X_batch,solver,solver_ex, precond=False):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(solver.model.trainable_variables)
            loss,L = self.get_loss(X_batch,solver,solver_ex, precond)
        g = tape.gradient(loss, solver.model.trainable_variables)
        del tape
        return loss, L, g
    
    
    def main_loop(self, N=1000, N_precond=10, N_batches=1):
        
        optimizer1,optimizer2 = self.create_optimizers()
        if self.precondition:
            optimizer1P,optimizer2P = self.create_optimizers(precond=True)

        @tf.function
        def train_step(X_batch, precond=False):
            X_batch1, X_batch2 = X_batch
            loss1, L_loss1, grad_theta1 = self.get_grad(X_batch1, self.solver1,self.solver2, precond)
            loss2, L_loss2, grad_theta2 = self.get_grad(X_batch2, self.solver2,self.solver1, precond)

            optimizer1.apply_gradients(zip(grad_theta1, self.solver1.model.trainable_variables))
            optimizer2.apply_gradients(zip(grad_theta2, self.solver2.model.trainable_variables))

            L1 = [loss1,L_loss1]
            L2 = [loss2,L_loss2]
            return L1,L2

        @tf.function
        def train_step_precond(X_batch, precond=True):
            X_batch1, X_batch2 = X_batch
            loss1, L_loss1, grad_theta1 = self.get_grad(X_batch1, self.solver1,self.solver2, precond)
            loss2, L_loss2, grad_theta2 = self.get_grad(X_batch2, self.solver2,self.solver1, precond)

            optimizer1P.apply_gradients(zip(grad_theta1, self.solver1.model.trainable_variables))
            optimizer2P.apply_gradients(zip(grad_theta2, self.solver2.model.trainable_variables))

            L1 = [loss1,L_loss1]
            L2 = [loss2,L_loss2]
            return L1,L2
    
        
        batches_r, batches_r_P = self.create_batches(N_batches)
        
        self.N_iters = N
        N_j = 0
        pbar = log_progress(range(N))
        pbar.set_description("Loss: %s " % 100)
        for i in pbar:

            if not self.precondition:
                b1,b2 = batches_r
                shuff_b1 = b1.shuffle(buffer_size=len(self.solver1.PDE.X_r))
                shuff_b2 = b2.shuffle(buffer_size=len(self.solver2.PDE.X_r))
                
                for X_b1, X_b2 in zip(shuff_b1,shuff_b2):
                    N_j += 1
                    L1,L2 = train_step((X_b1,X_b2), self.precondition)
            
            elif self.precondition:
                b1,b2 = batches_r_P
                shuff_b1 = b1.shuffle(buffer_size=len(self.solver1.PDE.X_r_P))
                shuff_b2 = b2.shuffle(buffer_size=len(self.solver2.PDE.X_r_P))
                
                for X_b1, X_b2 in zip(shuff_b1,shuff_b2):
                    N_j += 1
                    L1,L2 = train_step_precond((X_b1,X_b2), self.precondition)

            self.callback(L1,L2)

            if self.iter>N_precond and self.precondition:
                self.precondition = False
            
            if self.iter % 5 == 0:
                pbar.set_description("Loss: {:6.4e}".format(self.current_loss))

            if self.save_model_iter > 0:
                if self.iter % self.save_model_iter == 0:
                    dir_save = os.path.join(self.folder_path,f'iter_{self.iter}')
                    self.save_models(dir_save, [f'model_1',f'model_2'])
        
        logger.info(f' Iterations: {N}')
        logger.info(f' Total steps: {N_j}')
        logger.info(" Loss: {:6.4e}".format(self.current_loss))
    

    def create_optimizers(self, precond=False):
        if not precond:
            optim1 = tf.keras.optimizers.Adam(learning_rate=self.solver1.lr)
            optim2 = tf.keras.optimizers.Adam(learning_rate=self.solver2.lr)
            optimizers = [optim1,optim2]
            return optimizers
        elif precond:           
            lr = 0.001
            optim1P = tf.keras.optimizers.Adam(learning_rate=lr)
            optim2P = tf.keras.optimizers.Adam(learning_rate=lr)
            optimizers_p = [optim1P,optim2P]
            return optimizers_p


    def solve(self,N=1000, precond=False, N_precond=10, N_batches=1, save_model=0):

        self.precondition = precond
        self.save_model_iter = save_model

        t0 = time()
        self.main_loop(N, N_precond, N_batches=N_batches)
        logger.info('Computation time: {} minutes'.format(int((time()-t0)/60)))

        self.add_losses_NN()


    def create_batches(self, N_batches):

        number_batches = 1

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



    def callback(self, L1,L2):
        self.loss_r1.append(L1[1]['r'])
        self.loss_bD1.append(L1[1]['D'])
        self.loss_bN1.append(L1[1]['N'])
        self.loss_bI1.append(L1[1]['I'])
        self.loss_bK1.append(L1[1]['K'])

        self.loss_r2.append(L2[1]['r'])
        self.loss_bD2.append(L2[1]['D'])
        self.loss_bN2.append(L2[1]['N'])
        self.loss_bI2.append(L2[1]['I'])
        self.loss_bK2.append(L2[1]['K'])

        loss = L1[0] + L2[0]
        self.current_loss = loss.numpy()
        self.loss_hist.append(self.current_loss)
        self.iter+=1

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



if __name__=='__main__':
    pass

