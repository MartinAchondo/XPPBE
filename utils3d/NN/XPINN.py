import numpy as np
import tensorflow as tf
from time import time
from tqdm import tqdm as log_progress
import logging
import os

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

    def load_NeuralNets(self,dirs,names,lrs):   
        for solver,lr,name in zip(self.solvers,lrs,names):
            solver.load_NeuralNet(dirs,name,lr)  

    def save_models(self,dirs,names):
        for solver,name in zip(self.solvers,names):
            solver.save_model(dirs,name)   

    
    def get_loss(self,X_batch, s1,s2, precond):
        if precond:
            loss,L = s1.loss_fn(X_batch,precond=True)
        else:
            loss,L = s1.loss_fn(X_batch,precond=False)
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
    
    
    def solve_TF_optimizer(self, optimizer, N=1000, N_precond=10, N_batches=1):
        optimizer1,optimizer2 = optimizer

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
            
            if self.precondition:
                b1,b2 = batches_r_P
                shuff_b1 = b1.shuffle(buffer_size=len(self.solver1.PDE.X_r_P))
                shuff_b2 = b2.shuffle(buffer_size=len(self.solver2.PDE.X_r_P))
                
                for X_b1, X_b2 in zip(shuff_b1,shuff_b2):
                    N_j += 1
                    L1,L2 = train_step((X_b1,X_b2), self.precondition)


            self.callback(L1,L2)

            if self.iter>N_precond:
                self.precondition = False
            
            if self.iter % 10 == 0:
                pbar.set_description("Loss: {:6.4e}".format(self.current_loss))

            if self.save_model_iter > 0:
                if self.iter % self.save_model_iter == 0:
                    self.save_models(self.folder_path, [f'model_1_{self.iter}',f'model_2_{self.iter}'])
        
        logger.info(f' Iterations: {N}')
        logger.info(f' Total steps: {N_j}')
        logger.info(" Loss: {:6.4e}".format(self.current_loss))
    

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


    def solve(self,N=1000, precond=False, N_precond=10, N_batches=1, save_model=0):

        self.precondition = precond
        self.save_model_iter = save_model
        optim1 = tf.keras.optimizers.Adam(learning_rate=self.solver1.lr)
        optim2 = tf.keras.optimizers.Adam(learning_rate=self.solver2.lr)
        optim = [optim1,optim2]

        t0 = time()
        self.solve_TF_optimizer(optim, N, N_precond, N_batches=N_batches)
        logger.info('Computation time: {} minutes'.format(int((time()-t0)/60)))

        self.add_losses_NN()


if __name__=='__main__':
    pass

