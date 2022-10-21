import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class view_results():

    def __init__(self,NN):

        self.DTYPE='float32'
        self.pi = tf.constant(np.pi, dtype=self.DTYPE)

        self.NN = NN
        self.model = NN.model
        self.mesh = NN.mesh

        self.lb = self.model.lb
        self.ub = self.model.ub


    def plot_loss_history(self, flag=True, ax=None):
        if not ax:
            fig = plt.figure(figsize=(7,5))
            ax = fig.add_subplot(111)
        ax.semilogy(range(len(self.NN.loss_hist)), self.NN.loss_hist,'k-',label='Loss')
        if flag: 
            ax.semilogy(range(len(self.NN.loss_r)), self.NN.loss_r,'r-',label='Loss_r')
            ax.semilogy(range(len(self.NN.loss_bD)), self.NN.loss_bD,'b-',label='Loss_bD')
            ax.semilogy(range(len(self.NN.loss_bN)), self.NN.loss_bN,'g-',label='Loss_bN')
        ax.legend()
        ax.set_xlabel('$n_{epoch}$')
        ax.set_ylabel('$\\phi^{n_{epoch}}$')
        return ax


    def evaluate_u_point(self,X):
        X_input = tf.constant([X])
        U_output = self.model(X_input)
        return U_output.numpy()[0][0]

    
    def evaluate_u_array(self,X):
        x,y = X
        xt = tf.constant(x)
        yt = tf.constant(y)
        x = tf.reshape(xt,[xt.shape[0],1])
        y = tf.reshape(yt,[yt.shape[0],1])
        
        X = tf.concat([x, y], axis=1)
        U = self.model(X)

        return x[:,0],y[:,0],U[:,0]


    def get_u_domain(self,N=100):
        xspace = np.linspace(self.lb[0], self.ub[0], N + 1, dtype=self.DTYPE)
        yspace = np.linspace(self.lb[1], self.ub[1], N + 1, dtype=self.DTYPE)
        X, Y = np.meshgrid(xspace, yspace)

        if 'rmin' not in self.mesh.ins_domain:
            self.mesh.ins_domain['rmin'] = -0.1

        r = np.sqrt(X**2 + Y**2)
        inside1 = r < self.mesh.ins_domain['rmax']
        X1 = X[inside1]
        Y1 = Y[inside1]
        r = np.sqrt(X1**2 + Y1**2)
        inside = r > self.mesh.ins_domain['rmin']

        Xgrid = tf.constant(np.vstack([X1[inside].flatten(),Y1[inside].flatten()]).T)
        upred = self.model(tf.cast(Xgrid,self.DTYPE))
        
        return X1[inside].flatten(),Y1[inside].flatten(),upred.numpy()


    def plot_u_domain_countour(self,N=100):

        x,y,u = self.get_u_domain(N)
        plt.scatter(x,y,c=u)
        plt.colorbar();


    def plot_u_domain_surface(self,N=100,alpha1=35,alpha2=135):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x,y,u = self.get_u_domain(N)
        ax.scatter(x,y,u, c=u)
        ax.view_init(alpha1,alpha2)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$');


    def plot_u_plane(self,N=200):
        x = tf.constant(np.linspace(0, self.ub[0], 200, dtype=self.DTYPE))
        x = tf.reshape(x,[x.shape[0],1])
        y = tf.ones((N,1), dtype=self.DTYPE)*0
        X = tf.concat([x, y], axis=1)
        U = self.model(X)

        plt.plot(x[:,0],U[:,0])
        plt.xlabel('x')
        plt.ylabel('u');


if __name__=='__main__':
    pass