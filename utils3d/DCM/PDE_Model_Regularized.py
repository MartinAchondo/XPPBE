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
    

    def preconditioner(self,mesh,model,X):
        x,y,z = X

        rI = self.problem['rI']
        epsilon_1 = self.problem['epsilon_1']
        epsilon_2 = self.problem['epsilon_2']
        kappa = self.problem['kappa']
        q = self.q[0][0]

        G = (q/(4*self.pi))*(1/(epsilon_1))

        R = mesh.stack_X(x,y,z)
        u = model(R)
        r = tf.sqrt(x**2+y**2+z**2)
        upred = (q/(4*self.pi)) * ( 1/(epsilon_1*r) - 1/(epsilon_1*rI) + 1/(epsilon_2*(1+kappa*rI)*rI) ) - G/r
        loss = tf.reduce_mean(tf.square(upred-u))

        return loss
    
    def analytic(self,x,y,z):
        rI = self.problem['rI']
        epsilon_1 = self.problem['epsilon_1']
        epsilon_2 = self.problem['epsilon_2']
        kappa = self.problem['kappa']
        q = self.q[0][0]

        G = (q/(4*self.pi))*(1/(epsilon_1))

        r = tf.sqrt(x**2+y**2+z**2)
        upred = (q/(4*self.pi)) * ( 1/(epsilon_1*r) - 1/(epsilon_1*rI) + 1/(epsilon_2*(1+kappa*rI)*rI) ) - G/r

        return upred


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
    
    def border_value(self,x,y,z,R):
        q = 0
        for qk,Xk in self.q:
            xk,yk,zk = Xk
            #r_1 = 1/((x-xk)**2+(y-yk)**2+(z-zk)**2)
            q += qk
        r = np.sqrt(x**2 + y**2 + z**2)
        return q/(4*self.pi)*(np.exp(-self.kappa*(r-R))/(self.epsilon*(1+self.kappa*R)*r))
    
    def preconditioner(self,mesh,model,X):
        x,y,z = X

        rI = self.problem['rI']
        epsilon_1 = self.problem['epsilon_1']
        epsilon_2 = self.problem['epsilon_2']
        kappa = self.problem['kappa']
        q = self.q[0][0]

        G = (q/(4*self.pi))*(1/(epsilon_1))

        R = mesh.stack_X(x,y,z)
        u = model(R)
        r = tf.math.sqrt(x**2+y**2+z**2)

        upred = (q/(4*self.pi)) * (tf.math.exp(-kappa*(r-rI))/(epsilon_2*(1+kappa*rI)*r)) - G/r

        loss = tf.reduce_mean(tf.square(upred-u))

        return loss
    

    def analytic(self,x,y,z):
        rI = self.problem['rI']
        epsilon_1 = self.problem['epsilon_1']
        epsilon_2 = self.problem['epsilon_2']
        kappa = self.problem['kappa']
        q = self.q[0][0]

        G = (q/(4*self.pi))*(1/(epsilon_1))

        r = tf.math.sqrt(x**2+y**2+z**2)

        upred = (q/(4*self.pi)) * (tf.math.exp(-kappa*(r-rI))/(epsilon_2*(1+kappa*rI)*r)) - G/r

        return upred


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
        return (1/self.epsilon_G*4*self.pi)*sum



class PBE_Interface(PDE_utils):

    def __init__(self):
        super().__init__()

    def adapt_PDEs(self,PDEs,unions):
        self.PDEs = PDEs
        self.uns = unions

    def dG_n(self,x,y,z):
        # epsilon es del interior
        #x,y,z = X
        #n_v = self.normal_vector(X)
        sum = 0
        for qk,Xk in self.q:
            xk,yk,zk = Xk
            r_1 = 1/((x-xk)**2+(y-yk)**2+(z-zk)**2)
            sum += qk*r_1 
        return (-1/self.epsilon_G*4*self.pi)*sum

    def get_loss_I(self,solver,solver_ex,XI_data):
        loss = 0
        
        X = solver.mesh.get_X(XI_data)
        x_i,y_i,z_i = X

        R = solver.mesh.stack_X(x_i,y_i,z_i)
        u_1 = solver.model(R)
        u_2 = solver_ex.model(R)

        n_v = self.normal_vector(X)
        du_1 = self.directional_gradient(solver.mesh,solver.model,X,n_v)
        du_2 = self.directional_gradient(solver_ex.mesh,solver_ex.model,X,n_v)
        dG = self.dG_n(x_i,y_i,z_i)

        u_prom = (u_1+u_2)/2
        dun_prom = (solver.un*(du_1+dG) + solver_ex.un*(du_2+dG))/2
        
        loss += tf.reduce_mean(tf.square(u_1 - u_prom)) 
        loss += tf.reduce_mean(tf.square(solver.un*(du_1+dG)-dun_prom))
        
        return loss
    

    def analytic(self,r):
        rI = self.problem['rI']
        epsilon_1 = self.problem['epsilon_1']
        epsilon_2 = self.problem['epsilon_2']
        kappa = self.problem['kappa']
        q = self.q[0][0]

        G = (q/(4*self.pi))*(1/(epsilon_1))

        f_IN = lambda r: (q/(4*self.pi)) * ( 1/(epsilon_1*r) - 1/(epsilon_1*rI) + 1/(epsilon_2*(1+kappa*rI)*rI) ) - G/r
        f_OUT = lambda r: (q/(4*self.pi)) * (np.exp(-kappa*(r-rI))/(epsilon_2*(1+kappa*rI)*r)) - G/r

        y = np.piecewise(r, [r<=rI, r>rI], [f_IN, f_OUT])

        return y
    

    