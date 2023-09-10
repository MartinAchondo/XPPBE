import numpy as np
import tensorflow as tf
from time import time
from tqdm import tqdm as log_progress
import logging
import os
import pandas as pd

from NN.XPINN_utils import XPINN_utils

logger = logging.getLogger(__name__)


############################################################################################### 

        
class XPINN(XPINN_utils):
    
    def __init__(self, PINN):

        self.solver1, self.solver2 = PINN(), PINN()
        self.solvers = [self.solver1,self.solver2]
        self.alpha_w = 0.0        
        super().__init__()       
    
    def loss_PINN(self, pinn, X_batch, precond=False):
        if precond:
            L = pinn.PDE.get_loss_preconditioner(X_batch, pinn.model)
        elif not precond:
            L = pinn.PDE.get_loss(X_batch, pinn.model)
        return L


    def get_loss(self, X_batch, s1, s2, w, precond=False):
        L = self.loss_PINN(s1,X_batch,precond=precond)
        if not precond:    
            L['I'] += self.PDE.loss_I(s1,s2)

        loss = 0
        for t in s1.L_names:
            loss += w[t]*L[t]

        return loss,L
    

    def get_grad(self,X_batch,solver,solver_ex, w, precond):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(solver.model.trainable_variables)
            loss,L = self.get_loss(X_batch,solver,solver_ex, w, precond)
        g = tape.gradient(loss, solver.model.trainable_variables)
        del tape
        return loss, L, g
    
    
    def main_loop(self, N=1000, N_precond=10, N_batches=1):
        
        optimizer1,optimizer2 = self.create_optimizers()
        if self.precondition:
            optimizer1P,optimizer2P = self.create_optimizers(precond=True)

        @tf.function
        def train_step(X_batch, ws,precond=False):
            X_batch1, X_batch2 = X_batch
            loss1, L_loss1, grad_theta1 = self.get_grad(X_batch1, self.solver1,self.solver2, ws[0],precond)
            loss2, L_loss2, grad_theta2 = self.get_grad(X_batch2, self.solver2,self.solver1, ws[1],precond)

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
        pbar.set_description("Loss: {:6.4e}".format(100))
        for i in pbar:

            if not self.precondition:

                self.check_get_new_weights()

                b1,b2 = batches_r
                shuff_b1 = b1.shuffle(buffer_size=len(self.solver1.PDE.X_r))
                shuff_b2 = b2.shuffle(buffer_size=len(self.solver2.PDE.X_r))

                for X_b1, X_b2 in zip(shuff_b1,shuff_b2):
                    N_j += 1
                    L1,L2 = train_step((X_b1,X_b2), ws=[self.solver1.w,self.solver2.w], precond=self.precondition)
            
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
            
            if self.iter % 2 == 0:
                pbar.set_description("Loss: {:6.4e}".format(self.current_loss))

            if self.save_model_iter > 0:
                if self.iter % self.save_model_iter == 0:
                    dir_save = os.path.join(self.folder_path,f'iter_{self.iter}')
                    self.save_models(dir_save, [f'model_1',f'model_2'])

        
        logger.info(f' Iterations: {N}')
        logger.info(f' Total steps: {N_j}')
        logger.info(" Loss: {:6.4e}".format(self.current_loss))
    

    def check_get_new_weights(self):
        if self.adapt_weights and (self.iter % self.adapt_w_iter)==0 and self.iter>1:        
            for solver in self.solvers:
                loss_wo_w = sum(solver.L.values())
                for t in solver.L_names:
                    if t in solver.mesh.meshes_names:
                        eps = 1e-10
                        w = float(loss_wo_w/(solver.L[t]+eps))
                        solver.w[t] = self.alpha_w*solver.w[t] + (1-self.alpha_w)*w


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


    def solve(self,N=1000, precond=False, N_precond=10, N_batches=1, save_model=0, adapt_weights=False, adapt_w_iter=1):

        self.precondition = precond
        self.save_model_iter = save_model

        self.adapt_weights = adapt_weights
        self.adapt_w_iter = adapt_w_iter

        t0 = time()
        self.main_loop(N, N_precond, N_batches=N_batches)
        logger.info('Computation time: {} minutes'.format(int((time()-t0)/60)))

        self.add_losses_NN()


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



if __name__=='__main__':
    pass

