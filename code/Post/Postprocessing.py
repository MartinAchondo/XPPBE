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

        self.NN = NN
        self.model = NN.model
        self.mesh = NN.mesh

        self.lb = self.NN.mesh.lb
        self.ub = self.NN.mesh.ub

        if not X:
            self.loss_last = np.format_float_scientific(self.NN.loss_hist[-1], unique=False, precision=3)

     
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



class View_results_X():

    def __init__(self,XPINN,Post, save=False,directory=None, data=False, last=False):

        self.DTYPE='float32'
        self.pi = tf.constant(np.pi, dtype=self.DTYPE)
        self.save = save
        self.directory = directory
        self.data = data

        self.XPINN = XPINN
        self.NN = [XPINN.solver1,XPINN.solver2]
        self.model = [XPINN.solver1.model,XPINN.solver2.model]
        self.mesh = [XPINN.solver1.mesh,XPINN.solver2.mesh]

        self.Post = list()

        for NN in self.NN:
            self.Post.append(Post(NN,X=True))

        if not last:
            self.loss_last = np.format_float_scientific(self.XPINN.loss_hist[-1], unique=False, precision=3)
        else:
            self.loss_last = 0

    def close_file(self):
        self.Excel_writer.close()
        logger.info(f'Excel created: {"results.xlsx"}')

    def plot_loss_history(self, flag=True, plot_w=False, ax=None):
        if not ax:
            fig = plt.figure(figsize=(7,5))
            ax = fig.add_subplot(111)
        ax.semilogy(range(len(self.XPINN.loss_hist)), self.XPINN.loss_hist,'k-',label='Loss')
        if flag: 
            iter = 1
            c = [['r','b','g','c','m','lime'],['salmon','royalblue','springgreen','aqua', 'pink','yellowgreen']]
            for NN in self.NN:
                if not plot_w:
                    w = {
                    'R': 1.0,
                    'D': 1.0,
                    'N': 1.0,
                    'K': 1.0,
                    'I': 1.0,
                    'E': 1.0,
                    'Q': 1.0
                    }
                elif plot_w:
                    w = NN.w_hist
                meshes_names = NN.Mesh_names
                if 'R' in meshes_names:
                    ax.semilogy(range(len(NN.loss_r)), w['R']*np.array(NN.loss_r),c[iter-1][0],label=f'Loss_R_{iter}')
                if 'D' in meshes_names:
                    ax.semilogy(range(len(NN.loss_bD)), w['D']*np.array(NN.loss_bD),c[iter-1][1],label=f'Loss_D_{iter}')
                if 'N' in meshes_names:
                    ax.semilogy(range(len(NN.loss_bN)), w['N']*np.array(NN.loss_bN),c[iter-1][2],label=f'Loss_N_{iter}')
                if 'K' in meshes_names:
                    ax.semilogy(range(len(NN.loss_bK)), w['K']*np.array(NN.loss_bK),c[iter-1][3],label=f'Loss_K_{iter}')
                if 'Q' in meshes_names:
                    ax.semilogy(range(len(NN.loss_bQ)), w['Q']*np.array(NN.loss_bQ),'gold',label=f'Loss_Q_{iter}')
                if 'I' in meshes_names:
                    ax.semilogy(range(len(NN.loss_bI)), w['I']*np.array(NN.loss_bI),c[iter-1][4],label=f'Loss_I_{iter}')
                if 'E' in meshes_names:
                    ax.semilogy(range(len(self.XPINN.loss_exp)), w['E']*np.array(self.XPINN.loss_exp),c[iter-1][5],label=f'Loss_E_{iter}')
                iter += 1      
        ax.legend()
        ax.set_xlabel('$n: iterations$')
        ax.set_ylabel(r'$\mathcal{L}: Losses$')
        text_l = r'$\phi_{\theta}$'
        ax.set_title(f'Solution {text_l} of PDE, Iterations: {self.XPINN.N_iters}, Loss: {self.loss_last}')
        ax.grid()

        if self.save:
            path = 'loss_history.png' if not plot_w  else 'loss_history_w.png' 
            path_save = os.path.join(self.directory,path)
            fig.savefig(path_save)
            logger.info(f'Loss history Plot saved: {path}')


    def plot_weights_history(self, ax=None):
        if not ax:
            fig = plt.figure(figsize=(7,5))
            ax = fig.add_subplot(111)

        iter = 1
        c = [['r','b','g','c','m','lime'],['salmon','royalblue','springgreen','aqua', 'pink','yellowgreen']]
        for NN in self.NN:
            if True:
                w = NN.w_hist
                meshes_names = NN.Mesh_names
                if 'R' in meshes_names:
                    ax.semilogy(range(len(w['R'])), w['R'], c[iter-1][0],label=f'w_R_{iter}')
                if 'D' in meshes_names:
                    ax.semilogy(range(len(w['D'])), w['D'], c[iter-1][1],label=f'w_D_{iter}')
                if 'N' in meshes_names:
                    ax.semilogy(range(len(w['N'])), w['N'], c[iter-1][2],label=f'w_N_{iter}')
                if 'K' in meshes_names:
                    ax.semilogy(range(len(w['K'])), w['K'], c[iter-1][3],label=f'w_K_{iter}')
                if 'Q' in meshes_names:
                    ax.semilogy(range(len(w['Q'])), w['Q'], 'gold',label=f'w_Q_{iter}')
                if 'I' in meshes_names:
                    ax.semilogy(range(len(w['I'])), w['I'], c[iter-1][4],label=f'w_I_{iter}')
                if 'E' in meshes_names:
                    ax.semilogy(range(len(w['E'])), w['E'],c[iter-1][5],label=f'w_E_{iter}')
            iter += 1
        ax.legend()
        ax.set_xlabel('$n: iterations$')
        ax.set_ylabel('w: weights')
        text_l = r'$\phi_{\theta}$'
        ax.set_title(f'Solution {text_l} of PDE, Iterations: {self.XPINN.N_iters}, Loss: {self.loss_last}')
        ax.grid()

        if self.save:
            path = 'weights_history.png'
            path_save = os.path.join(self.directory,path)
            fig.savefig(path_save)
            logger.info(f'Loss history Plot saved: {path}')



    def plot_u_plane(self,N=200, theta=np.pi/2, phi=0 ):
        fig, ax = plt.subplots()
        labels = ['Inside', 'Outside']
        colr = ['r','b']
        i = 0

        self.Post[0].rmin = 0
        self.Post[0].rmax = 1
        self.Post[1].rmin = 1
        self.Post[1].rmax = 8

        for post_obj in self.Post:
            r = tf.constant(np.linspace(post_obj.rmin,post_obj.rmax, N, dtype=self.DTYPE))
            r = tf.reshape(r,[r.shape[0],1])
            theta = tf.ones((N,1), dtype=self.DTYPE)*theta
            phi = tf.ones((N,1), dtype=self.DTYPE)*phi

            x = r*np.sin(theta)*np.cos(phi)
            y = r*np.sin(theta)*np.sin(phi)
            z = r*np.cos(theta)

            X = tf.concat([x, y, z], axis=1)
            U = post_obj.model(X)
            
            ax.plot(r[:,0],U[:,0], label=labels[i], c=colr[i])
            i += 1

        ax.set_xlabel('r')
        ax.set_ylabel(r'$\phi_{\theta}$')

        text_l = r'$\phi_{\theta}$'
        ax.set_title(f'Solution {text_l} of PDE, Iterations: {self.XPINN.N_iters}, Loss: {self.loss_last}')

        ax.grid()
        ax.legend()

        if self.save:
            path = 'solution.png'
            path_save = os.path.join(self.directory,path)
            fig.savefig(path_save)
            logger.info(f'Solution Plot saved: {path}')

    def plot_u_domain_contour(self, N=35):
        fig, ax = plt.subplots()
        vmax,vmin = self.get_max_min()
        for post_obj in self.Post:
            x,y,z,u = post_obj.get_u_domain(N)
            plane = np.abs(z)<10**-4
            s = ax.scatter(x[plane], y[plane], c=u[plane],vmin=vmin,vmax=vmax)
            
        ax.set_xlim([-10,10])
        ax.set_ylim([-10,10])
        fig.colorbar(s, ax=ax)

        text_l = r'$\phi_{\theta}$'
        ax.set_title(f'Solution {text_l} of PDE, Iterations: {self.XPINN.N_iters}, Loss: {self.loss_last}')

        if self.save:
            path = 'contour.png'
            path_save = os.path.join(self.directory,path)
            fig.savefig(path_save)
            logger.info(f'Contour Plot saved: {path}')


    def plot_aprox_analytic(self,N=200, theta=np.pi/2, phi=0, lims=None):
        
        flag = True
        fig, ax = plt.subplots() 

        for post_obj,NN in zip(self.Post,self.NN):
            r = tf.constant(np.linspace(post_obj.rmin,post_obj.rmax, N, dtype=self.DTYPE))
            r = tf.reshape(r,[r.shape[0],1])
            theta = tf.ones((N,1), dtype=self.DTYPE)*theta
            phi = tf.ones((N,1), dtype=self.DTYPE)*phi

            x = r*np.sin(theta)*np.cos(phi)
            y = r*np.sin(theta)*np.sin(phi)
            z = r*np.cos(theta)

            X = tf.concat([x, y, z], axis=1)
            U = post_obj.model(X)
            
            if flag:
                ax.plot(r[:,0],U[:,0], c='b', label='Aprox')
                flag = False
            else:
                ax.plot(r[:,0],U[:,0], c='b')
        r = np.linspace(0.2, 8, 200, dtype=self.DTYPE)

        U2 = self.XPINN.PDE.analytic_Born_Ion(r)
        ax.plot(r,U2, c='r', label='Analytic', linestyle='--')
            
        ax.set_xlabel('r')
        ax.set_ylabel(r'$\phi_{\theta}$')
        
        if lims != None:
            ax.set_ylim(lims)
            
        ax.grid()
        ax.legend()

        text_l = r'$\phi_{\theta}$'
        ax.set_title(f'Solution {text_l} of PDE, Iterations: {self.XPINN.N_iters}, Loss: {self.loss_last}')

        if self.save:
            path = 'analytic.png'
            path_save = os.path.join(self.directory,path)
            fig.savefig(path_save)
            logger.info(f'Analytic Plot saved: {path}')   


    def plot_interface(self,N=200):

        labels = ['Inside', 'Outside']
        colr = ['r','b']
        i = 0

        rr = 1
        self.pi = np.pi
        
        r_bl = np.linspace(rr, rr, N + 1, dtype=self.DTYPE)
        theta_bl = np.linspace(np.pi/2, np.pi/2, N + 1, dtype=self.DTYPE)
        phi_bl = np.linspace(0, 2*self.pi, N + 1, dtype=self.DTYPE)
        
        x_bl = tf.constant(r_bl*np.sin(theta_bl)*np.cos(phi_bl))
        y_bl = tf.constant(r_bl*np.sin(theta_bl)*np.sin(phi_bl))
        z_bl = tf.constant(r_bl*np.cos(theta_bl))
        
        x_bl = tf.reshape(x_bl,[x_bl.shape[0],1])
        y_bl = tf.reshape(y_bl,[y_bl.shape[0],1])
        z_bl = tf.reshape(z_bl,[z_bl.shape[0],1])

        phi_bl = tf.constant(phi_bl)
        phi_bl = tf.reshape(phi_bl,[phi_bl.shape[0],1])

        XX_bl = tf.concat([x_bl, y_bl, z_bl], axis=1)

        fig, ax = plt.subplots() 

        for post_obj in self.Post:
            U = post_obj.model(XX_bl)
            ax.plot(phi_bl[:,0],U[:,0], label=labels[i], c=colr[i])
            i += 1

        r = np.linspace(1, 1, 200, dtype=self.DTYPE)

        U2 = self.XPINN.PDE.analytic_Born_Ion(1.0)
        u2 = phi_bl/phi_bl*U2
        ax.plot(phi_bl, u2, c='g', label='Analytic', linestyle='--')
        
        ax.set_xlabel(r'$\beta$')
        ax.set_ylabel(r'$\phi_{\theta}$')

        text_l = r'$\phi_{\theta}$'
        ax.set_title(f'Solution {text_l} of PDE, Iterations: {self.XPINN.N_iters}, Loss: {self.loss_last}')

        ax.grid()
        ax.legend()

        if self.save:
            path = 'interface.png'
            path_save = os.path.join(self.directory,path)
            fig.savefig(path_save)
            logger.info(f'Interface Plot saved: {path}')


    ########################################################################################################################
 
 
    def plot_u_domain_surface(self,N=100,alpha1=35,alpha2=135):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        vmax,vmin = self.get_max_min()
        for post_obj in self.Post:
            x,y,z,u = post_obj.get_u_domain(N)
            ax.scatter(x,y,u,c=u, vmin=vmin,vmax=vmax)
        ax.view_init(alpha1,alpha2)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$');


    def plot_loss(self,N=100):
        vmax,vmin = self.get_max_min_loss()
        for post_obj in self.Post:
            Xgrid,x,y = post_obj.get_grid(N)
            post_obj.NN.x,post_obj.NN.y = post_obj.mesh.get_X(Xgrid)
            loss = post_obj.NN.get_r()
            plt.scatter(x.flatten(),y.flatten(),c=tf.square(loss).numpy(), norm=matplotlib.colors.LogNorm(vmax=vmax, vmin=vmin))
        plt.colorbar();


    def plot_u_plane_direction(self,theta=0,phi=0,N=200):
        for post_obj in self.Post:
            r = tf.constant(np.linspace(post_obj.mesh.ins_domain['rmin'], post_obj.mesh.ins_domain['rmax'], 200, dtype=self.DTYPE))
            x = r*tf.sin(theta)*tf.cos(phi)
            x = tf.reshape(x,[x.shape[0],1])
            y = r*tf.sin(theta)*tf.sin(phi)
            y = tf.reshape(y,[y.shape[0],1])
            z = r*tf.cos(theta)
            z = tf.reshape(z,[z.shape[0],1])
            X = tf.concat([x, y, z], axis=1)
            U = post_obj.model(X)

            r = tf.reshape(r,[r.shape[0],1])
            plt.plot(r[:,0],U[:,0])
        plt.xlabel('r')
        plt.ylabel('u');



    def plot_loss_analytic(self,N=200):
        c = ['b','r']
        i=0
        for post_obj,NN in zip(self.Post,self.NN):
            x = tf.constant(np.linspace(post_obj.mesh.ins_domain['rmin'], post_obj.mesh.ins_domain['rmax'], 200, dtype=self.DTYPE))
            x = tf.reshape(x,[x.shape[0],1])
            y = tf.ones((N,1), dtype=self.DTYPE)*0
            X = tf.concat([x, y], axis=1)
            U = post_obj.model(X)

            U2 = NN.PDE.analytic(x,y)

            error = tf.square(U-U2)

            plt.plot(x[:,0],error[:,0], c=c[i], label='Error')
            i += 1

        plt.legend()
        plt.xlabel('x')
        plt.ylabel('Error')
        plt.yscale('log');


    def get_max_min(self,N=100):
        U = list()
        for post_obj in self.Post:
            _,_,_,u = post_obj.get_u_domain(N)
            U.append(u)
        Umax = list(map(np.max,U))
        vmax = np.max(np.array(Umax))
        Umin = list(map(np.min,U))
        vmin = np.min(np.array(Umin))
        return vmax,vmin

    def get_max_min_loss(self,N=100):
        L = list()
        for post_obj in self.Post:
            Xgrid,_,_,_ = post_obj.get_grid(N)
            post_obj.x,post_obj.y,post_obj.z = post_obj.mesh.get_X(Xgrid)
            loss = post_obj.NN.get_r()
            L.append(loss.numpy()**2)
        Lmax = list(map(np.max,L))
        vmax = np.max(np.array(Lmax))
        Lmin = list(map(np.min,L))
        vmin = np.min(np.array(Lmin))
        
        return vmax,vmin    


if __name__=='__main__':
    pass