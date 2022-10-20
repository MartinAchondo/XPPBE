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
        self.iter = 0

    def adapt_mesh(self,mesh):
        self.mesh = mesh
        self.lb = mesh.lb
        self.ub = mesh.ub

        self.X_r = self.mesh.data_mesh['residual']
        self.XD_data,self.UD_data = self.mesh.data_mesh['dirichlet']
        self.XN_data,self.UN_data,self.derN = self.mesh.data_mesh['neumann']
        self.XI_data,self.derI = self.mesh.data_mesh['interface']

        self.x = self.X_r[:,0:1]
        self.y = self.X_r[:,1:2]

    def create_NeuralNet(self,NN_class):
        self.model = NN_class(self.mesh.lb, self.mesh.ub)
        self.model.build_Net()

    def adapt_PDE(self,PDE):
        self.PDE = PDE


    def load_NeuralNet(self,directory,name):
        path = os.path.join(os.getcwd(),directory,name)
        NN_model = tf.keras.models.load_model(path, compile=False)
        self.model = NN_model

    def save_model(self,directory,name):
        dir_path = os.path.join(os.getcwd(),directory)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        self.model.save(os.path.join(dir_path,name))


    def get_r(self):
        with tf.GradientTape(persistent=True) as tape:
           
            tape.watch(self.x)
            tape.watch(self.y)
            u = self.model(tf.stack([self.x[:,0], self.y[:,0]], axis=1))
            u_x = tape.gradient(u, self.x)
            u_y = tape.gradient(u, self.y)
            
        u_xx = tape.gradient(u_x, self.x)
        u_yy = tape.gradient(u_y, self.y)

        del tape
        return self.PDE.fun_r(self.x, self.y, u_x, u_xx, u_yy)
    

    def loss_fn(self):
        
        r = self.get_r()
        phi_r = tf.reduce_mean(tf.square(r))
        loss = phi_r
        L = dict()
        L['r'] = phi_r
        L['D'] = 0
        L['N'] = 0
        L['P'] = 0
        
        for i in range(len(self.XD_data)):
            u_pred = self.model(self.XD_data[i])
            loss += tf.reduce_mean(tf.square(self.UD_data[i] - u_pred))
            L['D'] += tf.reduce_mean(tf.square(self.UD_data[i] - u_pred))


        for i in range(len(self.XN_data)):
            x_n = self.XN_data[i][:,0:1]
            y_n = self.XN_data[i][:,1:2]
            if self.derN[i]=='x':
                with tf.GradientTape() as tapex:
                    tapex.watch(x_n)
                    u_pred = self.model(tf.stack([x_n[:,0],y_n[:,0]], axis=1))
                ux_pred = tapex.gradient(u_pred,x_n)
                loss += tf.reduce_mean(tf.square(self.UN_data[i] - ux_pred))
                del tapex
                L['N'] += tf.reduce_mean(tf.square(self.UN_data[i] - ux_pred))
            elif self.derN[i]=='y':
                with tf.GradientTape() as tapey:
                    tapey.watch(y_n)
                    u_pred = self.model(tf.stack([x_n[:,0],y_n[:,0]], axis=1))
                uy_pred = tapey.gradient(u_pred,y_n)
                loss += tf.reduce_mean(tf.square(self.UN_data[i] - uy_pred))
                del tapey
                L['N'] += tf.reduce_mean(tf.square(self.UN_data[i] - uy_pred))

        loss_p = self.add_periodicity()
        loss += loss_p
        L['P'] += loss_p

        return loss,L
    
    def add_periodicity(self):
        Xr_pts = self.mesh.X_r
        pts = int(np.sqrt(self.mesh.N_r))+1
        Xp_1 = Xr_pts[:pts]
        Xp_2 = Xr_pts[len(Xr_pts)-pts:]

        x_1 = Xp_1[:,0:1]
        y_1 = Xp_1[:,1:2]
        x_2 = Xp_2[:,0:1]
        y_2 = Xp_2[:,1:2]
            
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(y_1)
            tape1.watch(y_2)
            u_pred_1 = self.model(tf.stack([x_1[:,0],y_1[:,0]], axis=1))
            u_pred_2 = self.model(tf.stack([x_2[:,0],y_2[:,0]], axis=1))
        uy_pred_1 = tape1.gradient(u_pred_1,y_1)
        uy_pred_2 = tape1.gradient(u_pred_2,y_2)

        loss = tf.reduce_mean(tf.square(u_pred_1-u_pred_2))
        loss += tf.reduce_mean(tf.square(uy_pred_1-uy_pred_2))
        
        del tape_xp
        return loss

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
            self.loss_P.append(L_loss['P'])
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
        optim = tf.keras.optimizers.Adam(learning_rate=lr)
        if self.flag_time:
            t0 = time()
            self.solve_with_TFoptimizer(optim, N)
            print('\nComputation time: {} seconds'.format(time()-t0))
        else:
            self.solve_with_TFoptimizer(optim, N)

    def get_u(self,N=600):
        xspace = tf.constant(np.linspace(self.lb[0], self.ub[0], N + 1))
        yspace = tf.constant(np.linspace(self.lb[1], self.ub[1], N + 1))
        X, Y = np.meshgrid(xspace, yspace)
        Xgrid = np.vstack([X.flatten(),Y.flatten()]).T
        upred = self.model(tf.cast(Xgrid,self.DTYPE))
        U = upred.numpy().reshape(N+1,N+1)
        return X,Y,U


    def plot_loss_history(self, flag=True, ax=None):
        if not ax:
            fig = plt.figure(figsize=(7,5))
            ax = fig.add_subplot(111)
        ax.semilogy(range(len(self.loss_hist)), self.loss_hist,'k-',label='Loss')
        if flag: 
            ax.semilogy(range(len(self.loss_r)), self.loss_r,'r-',label='Loss_r')
            ax.semilogy(range(len(self.loss_bD)), self.loss_bD,'b-',label='Loss_bD')
            ax.semilogy(range(len(self.loss_bN)), self.loss_bN,'g-',label='Loss_bN')
            ax.semilogy(range(len(self.loss_P)), self.loss_P,'c-',label='Loss_P')
        ax.legend()
        ax.set_xlabel('$n_{epoch}$')
        ax.set_ylabel('$\\phi^{n_{epoch}}$')
        return ax