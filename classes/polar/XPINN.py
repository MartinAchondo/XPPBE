import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from time import time
from matplotlib import cm

class XPINN():
    
    def __init__(self, NN, L_mesh, L_PDE, conds,PINN):

        self.solver1 = PINN(NN, L_mesh[0], L_PDE[0])
        self.solver2 = PINN(NN, L_mesh[1], L_PDE[1])

        self.solver1.un = conds[0]
        self.solver2.un = conds[1]

        self.solvers = [self.solver1,self.solver2]
        
        self.loss_hist = list()
        self.loss_r1 = list()
        self.loss_bD1 = list()
        self.loss_bN1 = list()
        self.loss_bI1 = list()
        self.loss_p1 = list()
        self.loss_r2 = list()
        self.loss_bD2 = list()
        self.loss_bN2 = list()
        self.loss_bI2 = list()
        self.loss_p2 = list()

        self.iter = 0

        self.DTYPE = 'float32'
    

    def loss_I(self):
        for j in range(len(self.solver2.XI_data)):
            x_i = self.solver2.XI_data[j][:,0:1]
            y_i = self.solver2.XI_data[j][:,1:2]
            
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch(x_i)
                tape1.watch(y_i)
                u_pred_1 = self.solver1.model(tf.stack([x_i[:,0],y_i[:,0]], axis=1))
                ux_pred_1 = tape1.gradient(u_pred_1,x_i)
                uy_pred_1 = tape1.gradient(u_pred_1,y_i)
            uxx_1 = tape1.gradient(ux_pred_1,x_i)
            uyy_1 = tape1.gradient(ux_pred_1,y_i)

            with tf.GradientTape(persistent=True) as tape2:
                tape2.watch(x_i)
                tape2.watch(y_i)
                u_pred_2 = self.solver2.model(tf.stack([x_i[:,0],y_i[:,0]], axis=1))
                ux_pred_2 = tape2.gradient(u_pred_2,x_i)
                uy_pred_2 = tape2.gradient(u_pred_2,y_i)
            uxx_2 = tape2.gradient(ux_pred_2,x_i)
            uyy_2 = tape2.gradient(ux_pred_2,y_i)

            loss = tf.reduce_mean(tf.square(u_pred_1 - u_pred_2)) 
            #loss += tf.reduce_mean(tf.square(self.solver1.PDE.fun_r(x_i,y_i,ux_pred_1,uxx_1,uyy_1)-self.solver1.PDE.fun_r(x_i,y_i,ux_pred_2,uxx_2,uyy_2)))
            loss += tf.reduce_mean(tf.square(ux_pred_1*self.solver1.un - ux_pred_2*self.solver2.un))

            del tape1
            del tape2
            
        return loss

    def get_grad(self,solver,solver_ex):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(solver.model.trainable_variables)
            loss,L = solver.loss_fn()
            loss_I = self.loss_I()
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
            self.loss_p1.append(L1[1]['P'])

            self.loss_r2.append(L2[1]['r'])
            self.loss_bD2.append(L2[1]['D'])
            self.loss_bN2.append(L2[1]['N'])
            self.loss_bI2.append(L2[1]['I'])
            self.loss_p2.append(L2[1]['P'])

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


    def plot_solution(self,angle1=35, angle2=35):
        N = 600
        tspace = np.linspace(self.solver1.lb[0], self.solver1.ub[0], N + 1)
        xspace = np.linspace(self.solver1.lb[1], self.solver1.ub[1], N + 1)
        T1, X1 = np.meshgrid(tspace, xspace)
        Xgrid = np.vstack([T1.flatten(),X1.flatten()]).T
        upred = self.solver1.model(tf.cast(Xgrid,self.solver1.DTYPE))
        U1 = upred.numpy().reshape(N+1,N+1)

        tspace = np.linspace(self.solver2.lb[0], self.solver2.ub[0], N + 1)
        xspace = np.linspace(self.solver2.lb[1], self.solver2.ub[1], N + 1)
        T2, X2 = np.meshgrid(tspace, xspace)
        Xgrid = np.vstack([T2.flatten(),X2.flatten()]).T
        upred = self.solver2.model(tf.cast(Xgrid,self.solver2.DTYPE))
        U2= upred.numpy().reshape(N+1,N+1)
       
        fig = plt.figure(figsize=(9,6))
        ax = fig.add_subplot(111, projection='3d')

        max_v = np.max(np.array([np.max(U1),np.max(U2)]))
        min_v = np.min(np.array([np.min(U1),np.min(U2)]))

        ax.plot_surface(T1, X1, U1, cmap=cm.viridis, vmin=min_v, vmax=max_v);
        ax.plot_surface(T2, X2, U2,  cmap=cm.viridis, vmin=min_v, vmax=max_v);
        
        ax.view_init(angle1,angle2)
        ax.set_xlabel('$t$')
        ax.set_ylabel('$x$')
        ax.set_zlabel('$u_\\theta(t,x)$')
        ax.set_title('Solution of Heat equation');

    def get_u(self,N=600):
        xspace = tf.constant(np.linspace(self.solver1.lb[0], self.solver1.ub[0], N + 1))
        yspace = tf.constant(np.linspace(self.solver1.lb[1], self.solver1.ub[1], N + 1))
        X, Y = np.meshgrid(xspace, yspace)
        Xgrid = np.vstack([X.flatten(),Y.flatten()]).T
        upred = self.solver1.model(tf.cast(Xgrid,self.DTYPE))
        U = upred.numpy().reshape(N+1,N+1)

        xspace = tf.constant(np.linspace(self.solver2.lb[0], self.solver2.ub[0], N + 1))
        yspace = tf.constant(np.linspace(self.solver2.lb[1], self.solver2.ub[1], N + 1))
        X2, Y2 = np.meshgrid(xspace, yspace)
        Xgrid = np.vstack([X.flatten(),Y.flatten()]).T
        upred = self.solver2.model(tf.cast(Xgrid,self.DTYPE))
        U2 = upred.numpy().reshape(N+1,N+1)
        return ((X,Y,U),(X2,Y2,U2))

    def plot_loss_history1(self, flag=True, ax=None):
        if not ax:
            fig = plt.figure(figsize=(7,5))
            ax = fig.add_subplot(111)
        ax.semilogy(range(len(self.loss_hist)), self.loss_hist,'k-',label='Loss')
        if flag: 
            ax.semilogy(range(len(self.loss_r1)), self.loss_r1,'r-',label='Loss_r')
            ax.semilogy(range(len(self.loss_bD1)), self.loss_bD1,'b-',label='Loss_bD')
            ax.semilogy(range(len(self.loss_bN1)), self.loss_bN1,'g-',label='Loss_bN')
            ax.semilogy(range(len(self.loss_bI1)), self.loss_bI1,'m-',label='Loss_bI')
            ax.semilogy(range(len(self.loss_p1)), self.loss_p1,'c-',label='Loss_P')
        ax.legend()
        ax.set_xlabel('$n_{epoch}$')
        ax.set_ylabel('$\\phi^{n_{epoch}}$')
        return ax

    def plot_loss_history2(self, flag=True, ax=None):
        if not ax:
            fig = plt.figure(figsize=(7,5))
            ax = fig.add_subplot(111)
        ax.semilogy(range(len(self.loss_hist)), self.loss_hist,'k-',label='Loss')
        if flag: 
            ax.semilogy(range(len(self.loss_r2)), self.loss_r2,'r-',label='Loss_r')
            ax.semilogy(range(len(self.loss_bD2)), self.loss_bD2,'b-',label='Loss_bD')
            ax.semilogy(range(len(self.loss_bN2)), self.loss_bN2,'g-',label='Loss_bN')
            ax.semilogy(range(len(self.loss_bI2)), self.loss_bI2,'m-',label='Loss_bI')
            ax.semilogy(range(len(self.loss_p2)), self.loss_p2,'c-',label='Loss_P')
        ax.legend()
        ax.set_xlabel('$n_{epoch}$')
        ax.set_ylabel('$\\phi^{n_{epoch}}$')
        return ax