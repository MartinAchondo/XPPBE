import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import logging
import os
import pandas as pd

logger = logging.getLogger(__name__)


class View_results():

    def __init__(self,NN,save=False,directory=None, data=False, X=False):

        self.DTYPE='float32'
        self.pi = tf.constant(np.pi, dtype=self.DTYPE)
        self.save = save
        self.directory = directory
        self.data = data
        if self.data:
            self.Excel_writer = pd.ExcelWriter(os.path.join(directory,'results.xlsx'), engine='openpyxl')

        self.NN = NN
        self.model = NN.model
        self.mesh = NN.mesh

        self.lb = self.NN.mesh.lb
        self.ub = self.NN.mesh.ub

        if not X:
            self.loss_last = np.format_float_scientific(self.NN.loss_hist[-1], unique=False, precision=3)

    def close_file(self):
        self.Excel_writer.close()
        logger.info(f'Excel created: {"results.xlsx"}')

    def plot_loss_history(self, flag=True, ax=None):
        if not ax:
            fig = plt.figure(figsize=(7, 5))
            ax = fig.add_subplot(111)

        ax.semilogy(range(len(self.NN.loss_hist)), self.NN.loss_hist, 'k-', label='Loss')

        if flag:
            ax.semilogy(range(len(self.NN.loss_r)), self.NN.loss_r, 'r-', label='Loss_r')
            ax.semilogy(range(len(self.NN.loss_bD)), self.NN.loss_bD, 'b-', label='Loss_bD')
            ax.semilogy(range(len(self.NN.loss_bN)), self.NN.loss_bN, 'g-', label='Loss_bN')
            ax.semilogy(range(len(self.NN.loss_bK)), self.NN.loss_bK, 'm-', label='Loss_bK')

        ax.legend()
        ax.set_xlabel('$n: iterations$')
        ax.set_ylabel(r'$\mathcal{L}: Losses$')
        text_l = r'$\phi_{\theta}$'
        ax.set_title(f'Solution {text_l} of PDE, Iterations: {self.NN.N_iters}, Loss: {self.loss_last}')
        ax.grid()

        if self.save:
            path = 'loss_history.png'
            path_save = os.path.join(self.directory,path)
            fig.savefig(path_save)
            logger.info(f'Loss history Plot saved: {path}')

            if self.data:
                d = {'Residual': list(map(lambda tensor: tensor.numpy(), self.NN.loss_r)),
                     'Dirichlet': list(map(lambda tensor: tensor.numpy(), self.NN.loss_bD)),
                     'Neumann': list(map(lambda tensor: tensor.numpy(), self.NN.loss_bN))
                     }
                df = pd.DataFrame(d)
                df.to_excel(self.Excel_writer, sheet_name='Losses', index=False)
                
        return ax
    
    def plot_u_plane(self,N=60,tv=0):
        N_r = N
        xspace = np.linspace(self.lb[0], self.ub[0], N_r, dtype=self.DTYPE)
        yspace = np.linspace(self.lb[1], self.ub[1], N_r, dtype=self.DTYPE)
        tspace = np.linspace(tv, tv, N_r, dtype=self.DTYPE)
        X, Y, time = np.meshgrid(xspace, yspace, tspace)

        X_r = tf.constant(np.vstack([X.flatten(),Y.flatten(), time.flatten()]).T)

        O = self.model(X_r)
        u = O[:,0]
        v = O[:,1]
        p = O[:,2]
        fig, ax = plt.subplots()

        ax.scatter(X, Y, c=u, label='Solution of PDE')
        ax.set_xlabel('r')
        ax.set_ylabel(r'$\phi_{\theta}$')

        loss = np.format_float_scientific(self.NN.loss_hist[-1], unique=False, precision=3)
        text_l = r'$\phi_{\theta}$'
        ax.set_title(f'Solution {text_l} of PDE, Iterations: {self.NN.N_iters}, Loss: {loss}')

        ax.grid()
        ax.legend()

        if self.save:
            path = f'solution_{tv}.png'
            path_save = os.path.join(self.directory,path)
            fig.savefig(path_save)
            logger.info(f'Solution Plot saved: {path}')

            if self.data:
                x=None
                U=x
                d = {'x': x[:,0],
                    'u': U[:,0]}
                df = pd.DataFrame(d)
                df.to_excel(self.Excel_writer, sheet_name='Solution', index=False)


    def plot_u_domain_contour(self,N=100):
        x,y,z,u = self.get_u_domain(N)
        plane = np.abs(z)<10**-4
    
        fig, ax = plt.subplots()

        s = ax.scatter(x[plane], y[plane], c=u[plane])
        fig.colorbar(s, ax=ax)

        loss = np.format_float_scientific(self.NN.loss_hist[-1], unique=False, precision=3)
        text_l = r'$\phi_{\theta}$'
        ax.set_title(f'Solution {text_l} of PDE, Iterations: {self.NN.N_iters}, Loss: {loss}')

        if self.save:
            path = 'contour.png'
            path_save = os.path.join(self.directory,path)
            fig.savefig(path_save)
            logger.info(f'Contour Plot saved: {path}')


    def plot_aprox_analytic(self,N=200):
        x = tf.constant(np.linspace(1, self.ub[0], 200, dtype=self.DTYPE))
        x = tf.reshape(x,[x.shape[0],1])
        y = tf.ones((N,1), dtype=self.DTYPE)*0
        z = tf.ones((N,1), dtype=self.DTYPE)*0
        X = tf.concat([x, y, z], axis=1)
        U = self.model(X)

        fig, ax = plt.subplots() 

        ax.plot(x[:,0],U[:,0], c='b', label='Aproximated')

        U2 = self.NN.PDE.analytic(x,y,z)
        ax.plot(x[:,0],U2[:,0], c='r', label='Analytic')

        ax.set_xlabel('r')
        ax.set_ylabel(r'$\phi_{\theta}$')
        if np.max(U[:,0]) > 0:
            ur = np.max(U[:,0])*1.2
        else:
            ur = 1
        #ax.set_ylim([np.min(U[:,0])*1.2,ur])

        loss = np.format_float_scientific(self.NN.loss_hist[-1], unique=False, precision=3)
        text_l = r'$\phi_{\theta}$'
        ax.set_title(f'Solution {text_l} of PDE, Iterations: {self.NN.N_iters}, Loss: {loss}')

        ax.grid()
        ax.legend()


        if self.save:
            path = 'solution_analytic.png'
            path_save = os.path.join(self.directory,path)
            fig.savefig(path_save)
            logger.info(f'Solution Plot saved: {path}')

     
    def get_grid(self,N=100):
        xspace = np.linspace(self.lb[0], self.ub[0], N + 1, dtype=self.DTYPE)
        yspace = np.linspace(self.lb[1], self.ub[1], N + 1, dtype=self.DTYPE)
        zspace = np.linspace(self.lb[2], self.ub[2], N + 1, dtype=self.DTYPE)
        X, Y, Z = np.meshgrid(xspace, yspace, zspace)
        
        if 'rmin' not in self.mesh.ins_domain:
            self.mesh.ins_domain['rmin'] = -0.1

        r = np.sqrt(X**2 + Y**2 + Z**2)
        inside1 = r < self.mesh.ins_domain['rmax']
        X1 = X[inside1]
        Y1 = Y[inside1]
        Z1 = Z[inside1]
        r = np.sqrt(X1**2 + Y1**2 + Z1**2)
        inside = r > self.mesh.ins_domain['rmin']

        Xgrid = tf.constant(np.vstack([X1[inside].flatten(),Y1[inside].flatten(), Z1[inside].flatten()]).T)

        return Xgrid,X1[inside],Y1[inside],Z1[inside]


    def get_loss(self,N=100):
        
        Xgrid,_,_ = self.get_grid(N)
        self.NN.x,self.NN.y,self.NN.z = self.mesh.get_X(Xgrid)
        loss,L = self.NN.loss_fn()

        L['r'] = L['r'].numpy()
        if len(self.NN.XD_data)!=0:
            L['D'] = L['D'].numpy()
        if len(self.NN.XN_data)!=0:
            L['N'] = L['N'].numpy()
        
        return loss.numpy(),L


    def plot_loss(self,N=100):
        Xgrid,x,y,z = self.get_grid(N)
        self.NN.x,self.NN.y,self.NN.z = self.mesh.get_X(Xgrid)
        loss = self.NN.get_r()
        plane = np.abs(z)<10**-4
        plt.scatter(x.flatten()[plane],y.flatten()[plane],c=tf.square(loss).numpy()[plane], norm=matplotlib.colors.LogNorm())
        plt.colorbar();


    def evaluate_u_point(self,X):
        X_input = tf.constant([X])
        U_output = self.model(X_input)
        return U_output.numpy()[0][0]

    
    def evaluate_u_array(self,X):
        x,y,z = X
        xt = tf.constant(x)
        yt = tf.constant(y)
        zt = tf.constant(z)
        x = tf.reshape(xt,[xt.shape[0],1])
        y = tf.reshape(yt,[yt.shape[0],1])
        z = tf.reshape(zt,[zt.shape[0],1])
        
        X = tf.concat([x, y, z], axis=1)
        U = self.model(X)

        return x[:,0],y[:,0],z[:,0],U[:,0]


    def get_u_domain(self,N=100):
        Xgrid,X,Y,Z = self.get_grid(N)
        upred = self.model(tf.cast(Xgrid,self.DTYPE))
        return X.flatten(),Y.flatten(),Z.flatten(),upred.numpy()

    def plot_u_domain_surface(self,N=100,alpha1=35,alpha2=135):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x,y,z,u = self.get_u_domain(N)
        plane = np.abs(z)<10**-4
        ax.scatter(x[plane], y[plane],u[plane], c=u[plane])
        ax.view_init(alpha1,alpha2)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$');


    def plot_loss_analytic(self,N=200):
        x = tf.constant(np.linspace(0, self.ub[0], 200, dtype=self.DTYPE))
        x = tf.reshape(x,[x.shape[0],1])
        y = tf.ones((N,1), dtype=self.DTYPE)*0
        z = tf.ones((N,1), dtype=self.DTYPE)*0
        X = tf.concat([x, y, z], axis=1)
        U = self.model(X)
        U2 = self.NN.PDE.analytic(x,y,z)

        error = tf.square(U-U2)
        plt.plot(x[:,0],error[:,0], c='r', label='Error')

        plt.legend()
        plt.xlabel('x')
        plt.ylabel('Error')
        plt.yscale('log');






