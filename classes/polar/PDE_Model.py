import tensorflow as tf
import numpy as np


class PDE_Model():

    def __init__(self):

        self.DTYPE='float32'
        self.pi = tf.constant(np.pi, dtype=self.DTYPE)
   
    def set_domain(self,X):
        x,y = X
        self.xmin = x[0]
        self.xmax = x[1]
        self.ymin = y[0]
        self.ymax = y[1]

        lb = tf.constant([self.xmin, self.ymin], dtype=self.DTYPE)
        ub = tf.constant([self.xmax, self.ymax], dtype=self.DTYPE)

        return (lb,ub)

    # Define boundary condition
    def fun_u_b(self,x, y, value):
        n = x.shape[0]
        return tf.ones((n,1), dtype=self.DTYPE)*value

    def fun_ux_b(self,x, y, value):
        n = x.shape[0]
        return tf.ones((n,1), dtype=self.DTYPE)*value

    def rhs(self,r,o):
        res = (1/(0.04*(2*self.pi)**0.5))*tf.exp((-1/(2*0.04**2))*((r*tf.cos(o))**2+(r*tf.sin(o))**2))
        return res

    # Define residual of the PDE
    def fun_r(self,r,o,u_r,u_rr,u_o):
        return u_rr*r*r + u_r*r + u_o - self.rhs(r,o)
        