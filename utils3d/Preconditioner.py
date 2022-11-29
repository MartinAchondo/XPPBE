import tensorflow as tf
import numpy as np
import types


class Preconditioner():

    def __init__(self):
        
        self.DTYPE='float32'
        self.pi = tf.constant(np.pi, dtype=self.DTYPE)

        self.epsilon = None

    def set_domain(self,X):
        x,y,z = X
        self.xmin,self.xmax = x
        self.ymin,self.ymax = y
        self.zmin,self.zmax = z

        lb = tf.constant([self.xmin, self.ymin,self.zmin], dtype=self.DTYPE)
        ub = tf.constant([self.xmax, self.ymax,self.zmax], dtype=self.DTYPE)

        return (lb,ub)

    def fun_r(self,x,y,z):
        z = (1/(4*self.pi))*(1/self.epsilon)*1/(tf.sqrt(x**2+y**2+z**2))
        return z

    def loss_fn(self,model,mesh):
        R = mesh.stack_X(self.x,self.y,self.z)
        upred = model(R)
        u = self.fun_r(self.x,self.y,self.z)
        loss = tf.reduce_mean(tf.square(upred-u))
        return loss
        
def change_fun(cls,new_func):
    cls.fun_r = types.MethodType(new_func,cls)