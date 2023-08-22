import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
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
 

    def adapt_mesh(self, mesh,
        w_r=1,
        w_d=1,
        w_n=1,
        w_i=1,
        w_k=1):

        logger.info("> Adapting Mesh")
        
        self.mesh = mesh
        self.lb = mesh.lb
        self.ub = mesh.ub
        self.PDE.adapt_PDE_mesh(self.mesh)

        self.w_i = w_i
        self.w = {
            'r': w_r,
            'D': w_d,
            'N': w_n,
            'K': w_k
        }
        
        self.L_names = ['r','D','N', 'K']

        logger.info("Mesh adapted")
        

    def create_NeuralNet(self,NN_class,lr,*args,**kwargs):
        logger.info("> Creating NeuralNet")
        self.model = NN_class(self.mesh.lb, self.mesh.ub,*args,**kwargs)
        self.model.build_Net()
        self.lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay(*lr)
        logger.info("Neural Network created")

    def adapt_PDE(self,PDE):
        logger.info("> Adapting PDE")
        self.PDE = PDE
        logger.info("PDE adapted")

    def load_NeuralNet(self,directory,name,lr):
        logger.info("> Adapting NeuralNet")
        path = os.path.join(os.getcwd(),directory,name)
        NN_model = tf.keras.models.load_model(path, compile=False)
        self.model = NN_model
        self.lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay(*lr)
        logger.info("Neural Network adapted")

    def save_model(self,directory,name):
        dir_path = os.path.join(os.getcwd(),directory)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        self.model.save(os.path.join(dir_path,name))
 

    def loss_fn(self, X_batch, precond=False):
        if precond:
            L = self.PDE.get_loss_preconditioner(X_batch, self.model)
        else:
            L = self.PDE.get_loss(X_batch, self.model)
        loss = 0
        for t in self.L_names:
            loss += L[t]*self.w[t]
        return loss,L
        
    def get_grad(self, X_batch, precond=False):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.model.trainable_variables)
            loss,L = self.loss_fn(X_batch, precond)
        g = tape.gradient(loss, self.model.trainable_variables)
        del tape
        return loss, L, g
    
    def solve_TF_optimizer(self, optimizer, N=1001, N_precond=10, N_batches=1):
        @tf.function
        def train_step(X_batch, precond=False):
            loss, L_loss, grad_theta = self.get_grad(X_batch, precond)
            optimizer.apply_gradients(zip(grad_theta, self.model.trainable_variables))
            return loss, L_loss
        
        batches_X_r, batches_X_r_P = self.create_batches(N_batches)

        N_j = 0
        pbar = log_progress(range(N))
        pbar.set_description("Loss: %s " % 100)
        for i in pbar:

            if not self.precondition:
                shuffled_batches_X_r = batches_X_r.shuffle(buffer_size=len(self.PDE.X_r))
                for X_batch in shuffled_batches_X_r:
                    N_j += 1
                    loss,L_loss = train_step(X_batch, self.precondition)
            
            if self.precondition:
                shuffled_batches_X_r_P = batches_X_r_P.shuffle(buffer_size=len(self.PDE.X_r_P))
                for X_batch in shuffled_batches_X_r_P:
                    N_j += 1
                    loss,L_loss = train_step(X_batch, self.precondition)

            self.callback(loss,L_loss)

            if self.iter>N_precond:
                self.precondition = False

            if self.iter % 10 == 0:
                pbar.set_description("Loss: {:6.4e}".format(self.current_loss))

            if self.save_model_iter > 0:
                if self.iter % self.save_model_iter == 0:
                    self.save_model(self.folder_path, f'model_{self.iter}')
            
        logger.info(f' Iterations: {N}')
        logger.info(f' Total steps: {N_j}')
        logger.info(" Loss: {:6.4e}".format(self.current_loss))


    def create_batches(self, N_batches):

        number_batches = N_batches

        dataset_X_r = tf.data.Dataset.from_tensor_slices(self.PDE.X_r)
        dataset_X_r = dataset_X_r.shuffle(buffer_size=len(self.PDE.X_r))

        batch_size = int(len(self.PDE.X_r)/number_batches)
        batches_X_r = dataset_X_r.batch(batch_size)


        dataset_X_r_P = tf.data.Dataset.from_tensor_slices(self.PDE.X_r_P)
        dataset_X_r_P = dataset_X_r_P.shuffle(buffer_size=len(self.PDE.X_r_P))

        batch_size = int(len(self.PDE.X_r_P)/number_batches)
        batches_X_r_P = dataset_X_r_P.batch(batch_size)

        return batches_X_r, batches_X_r_P
 

    def callback(self,loss,L_loss):
        self.loss_r.append(L_loss['r'])
        self.loss_bD.append(L_loss['D'])
        self.loss_bK.append(L_loss['K'])
        self.loss_bN.append(L_loss['N'])
        self.current_loss = loss.numpy()
        self.loss_hist.append(self.current_loss)
        self.iter+=1

    def solve(self,N=1000, precond=False, N_precond=10, N_batches=1, save_model=0):
        
        self.precondition = precond
        self.save_model_iter = save_model
        optim = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.N_iters = N

        t0 = time()
        self.solve_TF_optimizer(optim, N, N_precond, N_batches=N_batches)
        logger.info('Computation time: {} minutes'.format(int((time()-t0)/60)))



