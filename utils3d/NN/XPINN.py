import numpy as np
import tensorflow as tf
from time import time
from tqdm import tqdm as log_progress
import logging
import os
import pandas as pd

from NN.XPINN_utils import XPINN_utils


logger = logging.getLogger(__name__)


        
class XPINN(XPINN_utils):
    
    def __init__(self, PINN):

        self.solver1, self.solver2 = PINN(), PINN()
        self.solvers = [self.solver1,self.solver2]
          
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
            L['I'] += self.PDE.get_loss_I(s1,s2,X_batch['I'])
            L['E'] += self.PDE.get_loss_experimental(s1,s2,self.PDE.mesh.domain_meshes_data['E'])

        loss = 0
        for t in s1.mesh.meshes_names:
            loss += w[t]*L[t]
        for t in self.PDE.mesh.domain_meshes_names:
            loss += w[t]*L[t]
        return loss,L
    

    def get_grad(self,X_batch,solver,solver_ex, w, precond=False):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(solver.model.trainable_variables)
            loss,L = self.get_loss(X_batch,solver,solver_ex, w, precond)
        g = tape.gradient(loss, solver.model.trainable_variables)
        del tape
        return loss, L, g
    
    
    def main_loop(self, N=1000, N_precond=10):
        
        optimizer1,optimizer2 = self.create_optimizers()
        if self.precondition:
            optimizer1P,optimizer2P = self.create_optimizers(precond=True)

        @tf.function
        def train_step(X_batch, ws,precond=False):
            X_batch1, X_batch2 = X_batch
            loss1, L_loss1, grad_theta1 = self.get_grad(X_batch1, self.solver1,self.solver2, ws[0], precond)
            loss2, L_loss2, grad_theta2 = self.get_grad(X_batch2, self.solver2,self.solver1, ws[1], precond)

            optimizer1.apply_gradients(zip(grad_theta1, self.solver1.model.trainable_variables))
            optimizer2.apply_gradients(zip(grad_theta2, self.solver2.model.trainable_variables))

            L1 = [loss1,L_loss1]
            L2 = [loss2,L_loss2]
            return L1,L2

        @tf.function
        def train_step_precond(X_batch, ws, precond=True):
            X_batch1, X_batch2 = X_batch
            loss1, L_loss1, grad_theta1 = self.get_grad(X_batch1, self.solver1,self.solver2, ws[0], precond)
            loss2, L_loss2, grad_theta2 = self.get_grad(X_batch2, self.solver2,self.solver1, ws[1], precond)

            optimizer1P.apply_gradients(zip(grad_theta1, self.solver1.model.trainable_variables))
            optimizer2P.apply_gradients(zip(grad_theta2, self.solver2.model.trainable_variables))

            L1 = [loss1,L_loss1]
            L2 = [loss2,L_loss2]
            return L1,L2

        self.N_iters = N
        self.N_precond = N_precond
        self.N_steps = 0
        self.current_loss = 100

        self.pbar = log_progress(range(N))

        for i in self.pbar:

            L1,L2 = self.checkers_iterations()
            
            self.check_adapt_new_weights(self.adapt_w_now)
            TX_b1, TX_b2 = self.create_generators_shuffle(self.shuffle_now)

            for n_b in range(self.N_batches):

                X_b1 = self.get_batches(TX_b1)
                X_b2 = self.get_batches(TX_b2)    

                if not self.precondition:
                    L1_b,L2_b = train_step((X_b1,X_b2), ws=[self.solver1.w,self.solver2.w])   

                elif self.precondition:        
                    L1_b,L2_b = train_step_precond((X_b1,X_b2), ws=[self.solver1.w,self.solver2.w])

                L1,L2 = self.batch_iter_callback((L1,L2),(L1_b,L2_b)) 

            self.callback(L1,L2)
    

    def check_adapt_new_weights(self,adapt_now):
        
        if adapt_now:
            TX_b1, TX_b2 = self.create_generators_shuffle(True)
            X_b1 = self.get_batches(TX_b1)
            X_b2 = self.get_batches(TX_b2) 

            self.modify_weights_by(self.solver1,self.solver2,X_b1) 
            self.modify_weights_by(self.solver2,self.solver1,X_b2) 
            

    def modify_weights_by(self,solver,solver_ex,X_b):
        
        flag = 'gradients'
        self.alpha_w = 0.7  

        L = dict()
        if flag == 'gradients':
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(solver.model.trainable_variables)
                _,L_loss = self.get_loss(X_b,solver,solver_ex, solver.w)

            for t in solver.mesh.meshes_names:
                loss = L_loss[t]
                grads = tape.gradient(loss, solver.model.trainable_variables)
                grads = [grad if grad is not None else tf.zeros_like(var) for grad, var in zip(grads, solver.model.trainable_variables)]
                gradient_norm = tf.sqrt(sum([tf.reduce_sum(tf.square(g)) for g in grads]))
                L[t] = gradient_norm
            del tape

        elif flag == 'values':
            for t in solver.mesh.meshes_names:
                _,L = self.get_loss(X_b,solver,solver_ex, solver.w) 

        loss_wo_w = sum(L.values())
        for t in solver.mesh.meshes_names:
            eps = 1e-9
            w = float(loss_wo_w/(L[t]+eps))
            solver.w[t] = self.alpha_w*solver.w[t] + (1-self.alpha_w)*w     


    def create_optimizers(self, precond=False):
        if self.optimizer_name == 'Adam':
            if not precond:
                optim1 = tf.keras.optimizers.Adam(learning_rate=self.solver1.lr)
                optim2 = tf.keras.optimizers.Adam(learning_rate=self.solver2.lr)
                optimizers = [optim1,optim2]
                return optimizers
            elif precond:           
                optim1P = tf.keras.optimizers.Adam(learning_rate=self.solver1.lr_p)
                optim2P = tf.keras.optimizers.Adam(learning_rate=self.solver2.lr_p)
                optimizers_p = [optim1P,optim2P]
                return optimizers_p


    def solve(self,N=1000, precond=False, N_precond=10, save_model=0, adapt_weights=False, adapt_w_iter=1, shuffle = True, shuffle_iter = 500):

        self.precondition = precond
        self.save_model_iter = save_model if save_model != 0 else N

        self.adapt_weights = adapt_weights
        self.adapt_w_iter = adapt_w_iter

        self.shuffle = shuffle
        self.shuffle_iter = shuffle_iter 

        t0 = time()

        self.main_loop(N, N_precond)

        logger.info(f' Iterations: {self.N_iters}')
        logger.info(f' Total steps: {self.N_steps}')
        logger.info(" Loss: {:6.4e}".format(self.current_loss))
        logger.info('Computation time: {} minutes'.format(int((time()-t0)/60)))

        self.add_losses_NN()




if __name__=='__main__':
    pass

