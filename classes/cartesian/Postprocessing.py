import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

class View_results():

    def __init__(self,NN):

        self.DTYPE='float32'
        self.pi = tf.constant(np.pi, dtype=self.DTYPE)

        self.NN = NN
        self.model = NN.model
        self.mesh = NN.mesh

        self.lb = self.model.lb
        self.ub = self.model.ub

    def get_grid(self,N=100):
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

        return Xgrid,X1[inside],Y1[inside]


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

    def get_loss(self,N=100):
        
        Xgrid,_,_ = self.get_grid(N)
        self.NN.x,self.NN.y = self.mesh.get_X(Xgrid)
        loss,L = self.NN.loss_fn()

        L['r'] = L['r'].numpy()
        if len(self.NN.XD_data)!=0:
            L['D'] = L['D'].numpy()
        if len(self.NN.XN_data)!=0:
            L['N'] = L['N'].numpy()
        
        return loss.numpy(),L


    def plot_loss(self,N=100):
        Xgrid,x,y = self.get_grid(N)
        self.NN.x,self.NN.y = self.mesh.get_X(Xgrid)
        loss = self.NN.get_r()
        plt.scatter(x.flatten(),y.flatten(),c=tf.square(loss).numpy(), norm=matplotlib.colors.LogNorm())
        plt.colorbar();


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
        Xgrid,X,Y = self.get_grid(N)
        upred = self.model(tf.cast(Xgrid,self.DTYPE))
        return X.flatten(),Y.flatten(),upred.numpy()


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



class View_results_X():

    def __init__(self,XPINN,Post):

        self.DTYPE='float32'
        self.pi = tf.constant(np.pi, dtype=self.DTYPE)

        self.XPINN = XPINN
        self.NN = [XPINN.solver1,XPINN.solver2]
        self.model = [XPINN.solver1.model,XPINN.solver2.model]
        self.mesh = [XPINN.solver1.mesh,XPINN.solver2.mesh]

        self.lb = [self.model[0].lb,self.model[1].lb]
        self.ub = [self.model[0].ub,self.model[1].ub]

        self.Post = list()
        for NN in self.NN:
            self.Post.append(Post(NN))


    def plot_loss_history(self, flag=True, ax=None):
        if not ax:
            fig = plt.figure(figsize=(7,5))
            ax = fig.add_subplot(111)
        ax.semilogy(range(len(self.XPINN.loss_hist)), self.XPINN.loss_hist,'k-',label='Loss')
        if flag: 
            iter = 1
            for NN in self.NN:
                ax.semilogy(range(len(NN.loss_r)), NN.loss_r,'r-',label='Loss_r_NN_'+str(iter))
                ax.semilogy(range(len(NN.loss_bD)), NN.loss_bD,'b-',label='Loss_bD_NN_'+str(iter))
                ax.semilogy(range(len(NN.loss_bN)), NN.loss_bN,'g-',label='Loss_bN_NN_'+str(iter))
                iter += 1
            ax.semilogy(range(len(NN.loss_bI)), NN.loss_bI,'m-',label='Loss_bI')
        ax.legend()
        ax.set_xlabel('$n_{epoch}$')
        ax.set_ylabel('$\\phi^{n_{epoch}}$')


    def plot_u_domain_countour(self, N=100):
        vmax,vmin = self.get_max_min()
        for post_obj in self.Post:
            x,y,u = post_obj.get_u_domain(N)
            plt.scatter(x,y,c=u, vmin=vmin,vmax=vmax)
        plt.colorbar();

 
    def plot_u_domain_surface(self,N=100,alpha1=35,alpha2=135):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        vmax,vmin = self.get_max_min()
        for post_obj in self.Post:
            x,y,u = post_obj.get_u_domain(N)
            ax.scatter(x,y,u,c=u, vmin=vmin,vmax=vmax)
        ax.view_init(alpha1,alpha2)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$');


    def plot_loss(self,N=100):
        vmax,vmin = self.get_max_min_loss()
        for post_obj in self.Post:
            Xgrid,x,y = post_obj.get_grid(N)
            post_obj.x,post_obj.y = post_obj.mesh.get_X(Xgrid)
            loss = post_obj.NN.get_r()
            plt.scatter(x.flatten(),y.flatten(),c=tf.square(loss).numpy(), norm=matplotlib.colors.LogNorm(), vmin=vmin, vmax=vmax)
        plt.colorbar();


    def plot_u_plane(self,N=200):
        for post_obj in self.Post:
            x = tf.constant(np.linspace(post_obj.mesh.ins_domain['rmin'], post_obj.mesh.ins_domain['rmax'], 200, dtype=self.DTYPE))
            x = tf.reshape(x,[x.shape[0],1])
            y = tf.ones((N,1), dtype=self.DTYPE)*0
            X = tf.concat([x, y], axis=1)
            U = post_obj.model(X)

            plt.plot(x[:,0],U[:,0])
        plt.xlabel('x')
        plt.ylabel('u');


    def get_max_min(self,N=100):
        U = list()
        for post_obj in self.Post:
            _,_,u = post_obj.get_u_domain(N)
            U.append(u)
        Umax = list(map(np.max,U))
        vmax = np.max(np.array(Umax))
        Umin = list(map(np.min,U))
        vmin = np.min(np.array(Umin))
        return vmax,vmin

    def get_max_min_loss(self,N=100):
        L = list()
        for post_obj in self.Post:
            Xgrid,_,_ = post_obj.get_grid(N)
            post_obj.x,post_obj.y = post_obj.mesh.get_X(Xgrid)
            loss = post_obj.NN.get_r()
            L.append(loss.numpy())
        Lmax = list(map(np.max,L))
        vmax = np.abs(np.max(np.array(Lmax)))
        Lmin = list(map(np.min,L))
        vmin = np.abs(np.min(np.array(Lmin)))
        
        return vmax,vmin    


if __name__=='__main__':
    pass