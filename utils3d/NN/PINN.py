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
        self.loss_P = list()
        self.loss_bI = list()
        self.iter = 0
        self.lr = None

    def adapt_mesh(self, mesh,
        w_r=1,
        w_d=1,
        w_n=1,
        w_i=1):

        logger.info("> Adapting Mesh")
        
        self.mesh = mesh
        self.lb = mesh.lb
        self.ub = mesh.ub
        self.PDE.adapt_PDE_mesh(self.mesh)

        self.w_i = w_i
        self.w = {
            'r': w_r,
            'D': w_d,
            'N': w_n
        }
        
        self.L_names = ['r','D','N']

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
 

    def loss_fn(self):
        L = self.PDE.get_loss(self.model)
        loss = 0
        for t in self.L_names:
            loss += L[t]*self.w[t]
        return loss,L
        
    def get_grad(self):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.model.trainable_variables)
            loss,L = self.loss_fn()
        g = tape.gradient(loss, self.model.trainable_variables)
        del tape
        return loss, L, g
    
    def solve_with_TFoptimizer(self, optimizer, N=1001):
        @tf.function
        def train_step():
            loss, L_loss, grad_theta = self.get_grad()
            optimizer.apply_gradients(zip(grad_theta, self.model.trainable_variables))
            return loss, L_loss
        
        pbar = log_progress(range(N))
        pbar.set_description("Loss: %s " % 100)
        for i in pbar:
            loss,L_loss = train_step()
            self.callback(loss,L_loss)
            if self.iter % 10 == 0:
                pbar.set_description("Loss: {:6.4e}".format(self.current_loss))
        logger.info(f' Iterations: {N}')
        logger.info(" Loss: {:6.4e}".format(self.current_loss))


    def callback(self,loss,L_loss):
        self.loss_r.append(L_loss['r'])
        self.loss_bD.append(L_loss['D'])
        self.loss_bN.append(L_loss['N'])
        self.current_loss = loss.numpy()
        self.loss_hist.append(self.current_loss)
        self.iter+=1

    def solve(self,N=1000,flag_time=True):
        self.flag_time = flag_time
        optim = tf.keras.optimizers.Adam(learning_rate=self.lr)

        self.N_iters = N

        t0 = time()
        self.solve_with_TFoptimizer(optim, N)
        #print('\nComputation time: {6.4e} seconds'.format(time()-t0))
        logger.info('Computation time: {} minutes'.format(int((time()-t0)/60)))





class PINN_Precond(PINN):

    def __init__(self):
        super().__init__()

    def load_preconditioner(self,precond):
        self.precond = precond
        self.precond.X_r = self.X_r
        self.precond.x = self.x
        self.precond.y = self.y
        self.precond.z = self.z

    def get_precond_grad(self):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.model.trainable_variables)
            loss = self.precond.loss_fn(self.model,self.mesh)
        g = tape.gradient(loss, self.model.trainable_variables)
        del tape
        return loss, g

    def precond_with_TFoptimizer(self, optimizer, N=1001):
        @tf.function
        def train_step_precond():
            loss, grad_theta = self.get_precond_grad()
            optimizer.apply_gradients(zip(grad_theta, self.model.trainable_variables))
            return loss

        pbar = log_progress(range(N))
        pbar.set_description("Loss: %s " % 100)
        for i in pbar:
            loss = train_step_precond()
            self.callback(loss)

            if self.iter % 10 == 0:
                pbar.set_description("Loss: %s" % self.current_loss)

    def callback(self,loss):
        self.current_loss = loss.numpy()
        self.loss_hist.append(self.current_loss)
        self.iter+=1

    def preconditionate(self,N=2000):
        optim = tf.keras.optimizers.Adam(learning_rate=self.lr)

        t0 = time()
        self.precond_with_TFoptimizer(optim, N)
        print('\nComputation time: {} seconds'.format(time()-t0))


