import tensorflow as tf
import numpy as np
from DCM.PDE_utils import PDE_utils


class Poisson(PDE_utils):

    def __init__(self):

        self.sigma = 0.04
        self.epsilon = None
        self.epsilon_G = None
        self.q = None
        super().__init__()

    # Define loss of the PDE
    def residual_loss(self,mesh,model,X):
        x,y,z = X
        r = self.laplacian(mesh,model,X)*self.epsilon     
        Loss_r = tf.reduce_mean(tf.square(r))
        return Loss_r

    def G_Fun(self,x,y,z):
        # epsilon es del interior
        sum = 0
        for qk,Xk in self.q:
            xk,yk,zk = Xk
            r_1 = 1/((x-xk)**2+(y-yk)**2+(z-zk)**2)
            sum += qk*tf.sqrt(r_1)
        return (1/(self.epsilon_G*4*self.pi))*sum


    def analytic(self,x,y,z):
        q = 0
        for qk,Xk in self.q:
            xk,yk,zk = Xk
            #r_1 = 1/((x-xk)**2+(y-yk)**2+(z-zk)**2)
            q += qk
        r = tf.sqrt(x**2 + y**2 + z**2)
        return q/(4*self.pi)*(1/(2*r)-1/(2*1)+1/(80*(1+0.125*1)*1)) - self.G_Fun(x,y,z)



class Helmholtz(PDE_utils):

    def __init__(self):

        self.sigma = 0.04
        self.epsilon = None
        self.epsilon_2 = None
        self.epsilon_G = None
        self.q = None
        super().__init__()


    # Define loss of the PDE
    def residual_loss(self,mesh,model,X):
        x,y,z = X
        R = mesh.stack_X(x,y,z)
        u = model(R)
        r = self.laplacian(mesh,model,X) - self.kappa**2*(self.G_Fun(x,y,z)+u)    
        Loss_r = tf.reduce_mean(tf.square(r))
        return Loss_r
    

    def G_Fun(self,x,y,z):
        # epsilon es del interior
        sum = 0
        for qk,Xk in self.q:
            xk,yk,zk = Xk
            r_1 = 1/((x-xk)**2+(y-yk)**2+(z-zk)**2)
            sum += qk*tf.sqrt(r_1)
        return (1/(self.epsilon_G*4*self.pi))*sum
    

    def outer_border(self,x,y,z):
        sum = 0
        for qk,Xk in self.q:
            xk,yk,zk = Xk
            r_1 = tf.sqrt((x-xk)**2+(y-yk)**2+(z-zk)**2)
            sum += qk/r_1*tf.exp(-self.kappa*r_1)
        return (1/(self.epsilon*4*self.pi))*sum
      



    def analytic(self,x,y,z):
        q = 0
        for qk,Xk in self.q:
            xk,yk,zk = Xk
            #r_1 = 1/((x-xk)**2+(y-yk)**2+(z-zk)**2)
            q += qk
        r = tf.sqrt(x**2 + y**2 + z**2)
        return q/(4*self.pi)*(tf.exp(-0.125*(r-1))/(80*(1+0.125*1)*r)) - self.G_Fun(x,y,z)



class Non_Linear(PDE_utils):

    def __init__(self):

        self.sigma = 0.04
        self.epsilon = None
        self.epsilon_G = None
        self.q = None
        super().__init__()


    # Define loss of the PDE
    def residual_loss(self,mesh,model,X):
        x,y,z = X
        R = mesh.stack_X(x,y,z)
        u = model(R)
        r = self.laplacian(mesh,model,X) - self.kappa**2*tf.math.sinh(self.G_Fun(x,y,z)+u)      
        Loss_r = tf.reduce_mean(tf.square(r))
        return Loss_r
    
    def G_Fun(self,x,y,z):
        # epsilon es del interior
        sum = 0
        for qk,Xk in self.q:
            xk,yk,zk = Xk
            r_1 = 1/((x-xk)**2+(y-yk)**2+(z-zk)**2)
            sum += qk*r_1
        return (-1/self.epsilon_G*4*self.pi)*sum




class PDE_2_domains(PDE_utils):

    def __init__(self):
        super().__init__()


    def adapt_PDEs(self,PDEs,unions):
        self.PDEs = PDEs
        self.uns = unions


    def dG_n(self,X):
        # epsilon es del interior
        x,y,z = X
        n_v = self.normal_vector(X)
        sum = 0
        for qk,Xk in self.q:
            xk,yk,zk = Xk
            r_1 = 1/((x-xk)**2+(y-yk)**2+(z-zk)**2)
            sum += qk*r_1 
        return (-1/self.epsilon_G*4*self.pi)*sum



    def loss_I(self,solver,solver_ex):
        loss = 0
        for j in range(len(solver.PDE.XI_data)):
            X = solver.mesh.get_X(solver.PDE.XI_data[j])
            x_i,y_i,z_i = X

            R = solver.mesh.stack_X(x_i,y_i,z_i)
            u_1 = solver.model(R)
            u_2 = solver_ex.model(R)

            n_v = self.normal_vector(X)
            du_1 = self.directional_gradient(solver.mesh,solver.model,X,n_v)
            du_2 = self.directional_gradient(solver_ex.mesh,solver_ex.model,X,n_v)

            u_prom = (u_1+u_2)/2
            
            loss += tf.reduce_mean(tf.square(u_1 - u_2)) 
            loss += tf.reduce_mean(tf.square((du_1*solver.un - du_2*solver_ex.un)-(solver_ex.un-solver.un)*self.dG_n(x_i,y_i,z_i)))
            
        return loss

    