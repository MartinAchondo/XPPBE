import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from time import time


class XPINN():
    
    def __init__(self,PINN):

        self.DTYPE = 'float32'

        self.solver1 = PINN()
        self.solver2 = PINN()

        self.solvers = [self.solver1,self.solver2]
        
        self.loss_hist = list()

        self.loss_r1 = list()
        self.loss_bD1 = list()
        self.loss_bN1 = list()
        self.loss_bI1 = list()

        self.loss_r2 = list()
        self.loss_bD2 = list()
        self.loss_bN2 = list()
        self.loss_bI2 = list()

        self.iter = 0
        self.lr = None


    def adapt_PDEs(self,PDEs,unions):
        for solver,pde,union in zip(self.solvers,PDEs,unions):
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


    def loss_I(self,solver,solver_ex):
        for j in range(len(solver.XI_data)):
            x_i,y_i = solver.mesh.get_X(solver.XI_data[j])
            
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch(x_i)
                tape1.watch(y_i)
                R = self.mesh.stack_X(x_i,y_i)
                u_pred_1 = solver.model(R)
                ux_pred_1 = tape1.gradient(u_pred_1,x_i)
                uy_pred_1 = tape1.gradient(u_pred_1,y_i)
            uxx_1 = tape1.gradient(ux_pred_1,x_i)
            uyy_1 = tape1.gradient(ux_pred_1,y_i)

            with tf.GradientTape(persistent=True) as tape2:
                tape2.watch(x_i)
                tape2.watch(y_i)
                R = self.mesh.stack_X(x_i,y_i)
                u_pred_2 = solver_ex.model(R)
                ux_pred_2 = tape2.gradient(u_pred_2,x_i)
                uy_pred_2 = tape2.gradient(u_pred_2,y_i)
            uxx_2 = tape2.gradient(ux_pred_2,x_i)
            uyy_2 = tape2.gradient(ux_pred_2,y_i)

            u_prom = (u_pred_1+u_pred_2)/2
            loss = tf.reduce_mean(tf.square(u_pred_1 - u_prom)) 

            res = solver.PDE.fun_r(x_i,ux_pred_1,uxx_1,y_i,uy_pred_1,uyy_1)-solver_ex.PDE.fun_r(x_i,ux_pred_2,uxx_2,y_i,uy_pred_2,uyy_2)
            loss += tf.reduce_mean(tf.square(res))
            
            norm_vn = tf.sqrt(x_i**2+y_i**2)
            v1 = (x_i*ux_pred_1+y_i*uy_pred_1)/norm_vn
            v2 = (x_i*ux_pred_2+y_i*uy_pred_2)/norm_vn
            loss += tf.reduce_mean(tf.square(v1*solver.un - v2*solver_ex.un))

            del tape1
            del tape2
            
        return loss

    def get_grad(self,solver,solver_ex):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(solver.model.trainable_variables)
            loss,L = solver.loss_fn()
            loss_I = self.loss_I(solver,solver_ex)
            loss += loss_I 
            L['I'] = loss_I
        g = tape.gradient(loss, solver.model.trainable_variables)
        del tape
        return loss, L, g
    
    
    def solve_with_TFoptimizer(self, optimizer, N=1001):

        optimizer1,optimizer2 = optimizer

        @tf.function
        def train_step():
            loss1, L_loss1, grad_theta1 = self.get_grad(self.solver1,self.solver2)
            loss2, L_loss2, grad_theta2 = self.get_grad(self.solver2,self.solver1)

            optimizer1.apply_gradients(zip(grad_theta1, self.solver1.model.trainable_variables))
            optimizer2.apply_gradients(zip(grad_theta2, self.solver2.model.trainable_variables))

            L1 = [loss1,L_loss1]
            L2 = [loss2,L_loss2]
            return L1,L2
        
        for i in range(N):
            L1,L2 = train_step()
            self.loss_r1.append(L1[1]['r'])
            self.loss_bD1.append(L1[1]['D'])
            self.loss_bN1.append(L1[1]['N'])
            self.loss_bI1.append(L1[1]['I'])

            self.loss_r2.append(L2[1]['r'])
            self.loss_bD2.append(L2[1]['D'])
            self.loss_bN2.append(L2[1]['N'])
            self.loss_bI2.append(L2[1]['I'])

            loss = L1[0] + L2[0]

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
        lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay([1000,3000],[1e-2,1e-3,5e-4])
        optim1 = tf.keras.optimizers.Adam(learning_rate=lr)
        optim2 = tf.keras.optimizers.Adam(learning_rate=lr)
        optim = [optim1,optim2]
        if self.flag_time:
            t0 = time()
            self.solve_with_TFoptimizer(optim, N)
            print('\nComputation time: {} seconds'.format(time()-t0))
        else:
            self.solve_with_TFoptimizer(optim, N)


