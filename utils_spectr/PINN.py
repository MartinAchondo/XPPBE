import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from time import time
from tqdm import tqdm as log_progress

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
        
        self.mesh = mesh
        self.lb = mesh.lb
        self.ub = mesh.ub

        self.X_r = self.mesh.data_mesh['residual']
        self.w_r = w_r
        self.XD_data,self.UD_data = self.mesh.data_mesh['dirichlet']
        self.w_d = w_d
        self.XN_data,self.UN_data,self.derN = self.mesh.data_mesh['neumann']
        self.w_n = w_n
        self.XI_data,self.derI = self.mesh.data_mesh['interface']
        self.w_i = w_i

        self.x,self.y,self.z = self.mesh.get_X(self.X_r)
        

    def create_NeuralNet(self,NN_class,lr,*args,**kwargs):
        self.model = NN_class(self.mesh.lb, self.mesh.ub,*args,**kwargs)
        self.model.build_Net()
        self.lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay(*lr)


    def adapt_PDE(self,PDE):
        self.PDE = PDE

    def load_NeuralNet(self,directory,name,lr):
        path = os.path.join(os.getcwd(),directory,name)
        NN_model = tf.keras.models.load_model(path, compile=False)
        self.model = NN_model
        self.lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay(*lr)

    def save_model(self,directory,name):
        dir_path = os.path.join(os.getcwd(),directory)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        self.model.save(os.path.join(dir_path,name))

    def get_r(self):

        R = self.mesh.stack_X(self.x,self.y,self.z)
        uk = self.model(R)

        return self.PDE.fun_r(self.x, self.y, self.z, uk)
    

    def loss_fn(self):
        
        L = dict()
        L['r'] = 0
        L['D'] = 0
        L['N'] = 0

        #residual
        r = self.get_r()
        phi_r = tf.reduce_mean(tf.square(r))
        loss = self.w_r*phi_r
        L['r'] += loss

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
 
        t0 = time()
        self.solve_with_TFoptimizer(optim, N)
        print('\nComputation time: {} seconds'.format(time()-t0))


