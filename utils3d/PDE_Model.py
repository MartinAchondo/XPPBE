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

    def source(self,x,y,z):
        return (1/((2*self.pi)**(3.0/2)*self.sigma**3))*tf.exp((-1/(2*self.sigma**2))*(x**2+y**2+z**2))

    # Define residual of the PDE
    def fun_r(self,x,u_x,u_xx,y,u_y,u_yy,z,u_z,u_zz):
        return u_xx + u_yy + u_zz - self.source(x,y,z)


    def analytic(self,x,y):
        return (1/(self.epsilon*2*self.pi))*tf.math.log(tf.sqrt(x**2+y**2))
        

class PDE_Model_2():

    def __init__(self):

        self.DTYPE='float32'
        self.pi = tf.constant(np.pi, dtype=self.DTYPE)
        self.sigma = 0.04
        self.epsilon = None
    
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

    def source(self,x,y):
        return 0

    # Define residual of the PDE
    def fun_r(self,x,u_x,u_xx,y,u_y,u_yy):
        return u_xx + u_yy - self.source(x,y)

    
    def analytic(self,x,y):
        return (1/(self.epsilon*2*self.pi))*tf.math.log(tf.sqrt(x**2+y**2))
        