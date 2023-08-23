import tensorflow as tf
import numpy as np
from DCM.PDE_utils import PDE_utils


class Navier_Stokes(PDE_utils):

    def __init__(self):

        self.rho = None
        self.mu = None
        super().__init__()

    # Define loss of the PDE
    def residual_loss(self,mesh,model,X):
        x,y,t = X

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(y)
            tape.watch(t)
            R = mesh.stack_X(x,y,t)
            O = model(R)
            u = O[:,0]
            v = O[:,1]
            p = O[:,2]
            u_x = tape.gradient(u,x)
            u_y = tape.gradient(u,y)
            u_t = tape.gradient(u,t)
            v_x = tape.gradient(v,x)
            v_y = tape.gradient(v,y)
            v_t = tape.gradient(v,t)
            p_x = tape.gradient(p,x)
            p_y = tape.gradient(p,y)
        u_xx = tape.gradient(u_x,x)
        u_yy = tape.gradient(u_y,y)
        v_xx = tape.gradient(v_x,x)
        v_yy = tape.gradient(v_y,y)
        del tape

        rx = self.rho*(u_t + u*u_x + v*u_y) + p_x - self.mu*(u_xx+u_yy)
        ry = self.rho*(v_t + u*v_x + v*v_y) + p_y - self.mu*(v_xx+v_yy)
        rc = u_x + u_y     
        Loss_r = tf.reduce_mean(tf.square(rx)) + tf.reduce_mean(tf.square(ry)) + tf.reduce_mean(tf.square(rc))

        return Loss_r
    

    def analytic(self,x,y,t):
        rho = self.problem['rho']
        mu = self.problem['mu']
        nu = mu/rho

        u = np.exp(-2*nu*t)*np.cos(x)*np.sin(y)
        v = np.exp(-2*nu*t)*np.sin(x)*np.cos(y)
        p = -rho/4 * np.exp(4*nu*t)*(np.cos(2*x)+np.cos(2*y)) 

        return u,v,p


    def initial(self,x,y,t,n):
        rho = self.problem['rho']
        mu = self.problem['mu']
        nu = mu/rho
        A = np.ones((n,3), dtype=self.DTYPE)
        
        A[:,0:1] = np.exp(-2*nu*t)*np.cos(x)*np.sin(y)
        A[:,1:2] = np.exp(-2*nu*t)*np.sin(x)*np.cos(y)
        A[:,2:-1] = -rho/4 * np.exp(4*nu*t)*(np.cos(2*x)+np.cos(2*y))

        return A

