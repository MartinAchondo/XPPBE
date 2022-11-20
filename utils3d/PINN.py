import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from time import time


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

    def adapt_mesh(self,mesh,
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
        with tf.GradientTape(persistent=True) as tape:
           
            tape.watch(self.x)
            tape.watch(self.y)
            tape.watch(self.z)
            R = self.mesh.stack_X(self.x,self.y,self.z)
            u = self.model(R)
            u_x = tape.gradient(u, self.x)
            u_y = tape.gradient(u, self.y)
            u_z = tape.gradient(u, self.z)
            
        u_xx = tape.gradient(u_x, self.x)
        u_yy = tape.gradient(u_y, self.y)
        u_zz = tape.gradient(u_z, self.z)

        del tape
        return self.PDE.fun_r(self.x, u_x, u_xx, self.y, u_y, u_yy, self.z, u_z, u_zz)
    

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

        #dirichlet
        for i in range(len(self.XD_data)):
            u_pred = self.model(self.XD_data[i])
            loss += self.w_d*tf.reduce_mean(tf.square(self.UD_data[i] - u_pred))
            L['D'] += self.w_d*tf.reduce_mean(tf.square(self.UD_data[i] - u_pred))

        #neumann
        for i in range(len(self.XN_data)):
            x_n,y_n,z_n = self.mesh.get_X(self.XN_data[i])
            if self.derN[i]=='x':
                with tf.GradientTape() as tapex:
                    tapex.watch(x_n)
                    R = self.mesh.stack_X(x_n,y_n,z_n)
                    u_pred = self.model(R)
                ux_pred = tapex.gradient(u_pred,x_n)
                loss += self.w_n*tf.reduce_mean(tf.square(self.UN_data[i] - ux_pred))
                L['N'] += self.w_n*tf.reduce_mean(tf.square(self.UN_data[i] - ux_pred))
                del tapex
            elif self.derN[i]=='y':
                with tf.GradientTape() as tapey:
                    tapey.watch(y_n)
                    R = self.mesh.stack_X(x_n,y_n,z_n)
                    u_pred = self.model(R)
                uy_pred = tapey.gradient(u_pred,y_n)
                loss += self.w_n*tf.reduce_mean(tf.square(self.UN_data[i] - uy_pred))
                L['N'] += self.w_n*tf.reduce_mean(tf.square(self.UN_data[i] - uy_pred))
                del tapey
            elif self.derN[i]=='z':
                with tf.GradientTape() as tapez:
                    tapez.watch(z_n)
                    R = self.mesh.stack_X(x_n,y_n,z_n)
                    u_pred = self.model(R)
                uz_pred = tapez.gradient(u_pred,z_n)
                loss += self.w_n*tf.reduce_mean(tf.square(self.UN_data[i] - uz_pred))
                L['N'] += self.w_n*tf.reduce_mean(tf.square(self.UN_data[i] - uz_pred))
                del tapez
            elif self.derN[i]=='r':
                with tf.GradientTape(persistent=True) as taper:
                    taper.watch(x_n)
                    taper.watch(y_n)
                    taper.watch(z_n)
                    R = self.mesh.stack_X(x_n,y_n,z_n)
                    u_pred = self.model(R)
                ux_pred = taper.gradient(u_pred,x_n)
                uy_pred = taper.gradient(u_pred,y_n)
                uz_pred = taper.gradient(u_pred,z_n)
                
                norm_vn = tf.sqrt(x_n**2+y_n**2+z_n**2)
                un_pred = (x_n*ux_pred + y_n*uy_pred + z_n*uz_pred)/norm_vn

                loss += self.w_n*tf.reduce_mean(tf.square(self.UN_data[i] - un_pred))
                L['N'] += self.w_n*tf.reduce_mean(tf.square(self.UN_data[i] - un_pred))
                del taper


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
        
        for i in range(N):
            loss,L_loss = train_step()
            self.loss_r.append(L_loss['r'])
            self.loss_bD.append(L_loss['D'])
            self.loss_bN.append(L_loss['N'])
            self.current_loss = loss.numpy()
            self.callback()

    def callback(self, xr=None):
        if self.iter % 50 == 0:
            if self.flag_time:
                print('It {:05d}: loss = {:10.8e}'.format(self.iter,self.current_loss))
        self.loss_hist.append(self.current_loss)
        self.iter+=1

    def solve(self,N=1000,flag_time=True):
        self.flag_time = flag_time
        optim = tf.keras.optimizers.Adam(learning_rate=self.lr)
        if self.flag_time:
            t0 = time()
            self.solve_with_TFoptimizer(optim, N)
            print('\nComputation time: {} seconds'.format(time()-t0))
        else:
            self.solve_with_TFoptimizer(optim, N)


class PINN_Precond(PINN):

    def __init__(self):
        super().__init__()

    def load_preconditioner(self,precond):
        self.precond = precond
        self.precond.X_r = self.X_r
        self.precond.x = self.x
        self.precond.y = self.y

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

        for i in range(N):
            loss = train_step_precond()
            self.current_loss = loss.numpy()
            self.callback()

    def preconditionate(self,N=2000,flag_time=True):
        self.flag_time = flag_time
        optim = tf.keras.optimizers.Adam(learning_rate=self.lr)
        if self.flag_time:
            t0 = time()
            self.precond_with_TFoptimizer(optim, N)
            print('\nComputation time: {} seconds'.format(time()-t0))
        else:
            self.precond_with_TFoptimizer(optim, N)

