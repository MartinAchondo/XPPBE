import tensorflow as tf
import numpy as np


class PDE_Model():

    def __init__(self):

        self.DTYPE='float32'
        self.pi = tf.constant(np.pi, dtype=self.DTYPE)
        self.sigma = 0.04
        self.epsilon = None
    
    def set_domain(self,X):
        x,y,z = X
        self.xmin,self.xmax = x
        self.ymin,self.ymax = y
        self.zmin,self.zmax = z

        lb = tf.constant([self.xmin, self.ymin,self.zmin], dtype=self.DTYPE)
        ub = tf.constant([self.xmax, self.ymax,self.zmax], dtype=self.DTYPE)

        return (lb,ub)

    # Define boundary condition
    def fun_u_b(self,x, y, z, value):
        n = x.shape[0]
        return tf.ones((n,1), dtype=self.DTYPE)*value

    def fun_ux_b(self,x, y, z, value):
        n = x.shape[0]
        return tf.ones((n,1), dtype=self.DTYPE)*value

    def source(self,kx,ky,kz):
        return 1

    # Define residual of the PDE
    def fun_r(self,kx,ky,kz,uk):
        return (kx**2+ky**2+kz**2)*uk + self.source(kx,ky,kz)

