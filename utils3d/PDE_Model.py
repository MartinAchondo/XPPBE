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
    

    # Boundary Surface

    def surface(r,theta,phi):
        x = r*np.sin(theta)*np.cos(phi)
        y = r*np.sin(theta)*np.sin(phi)
        z = r*np.cos(theta)

        return (x,y,z)
    

    def normal_vector(self,X):
        x,y,z = X
        norm_vn = tf.sqrt(x**2+y**2+z**2)
        n = np.array([x,y,z])/norm_vn
        return n

    # Differential operators

    def laplacian(self,mesh,model,X):
        x,y,z = X
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(y)
            tape.watch(z)
            R = mesh.stack_X(x,y,z)
            u = model(R)
            u_x = tape.gradient(u,x)
            u_y = tape.gradient(u,y)
            u_z = tape.gradient(u,z)
        u_xx = tape.gradient(u_x,x)
        u_yy = tape.gradient(u_y,y)
        u_zz = tape.gradient(u_z,z)
        del tape

        return u_xx + u_yy + u_zz

    def gradient(self,mesh,model,X):
        x,y,z = X
        with tf.GradientTape(persistent=True,watch_accessed_variables=False) as tape:
            tape.watch(x)
            tape.watch(y)
            tape.watch(z)
            R = mesh.stack_X(x,y,z)
            u = model(R)
        u_x = tape.gradient(u,x)
        u_y = tape.gradient(u,y)
        u_z = tape.gradient(u,z)
        del tape

        return (u_x,u_y,u_z)
    
    def directional_gradient(self,mesh,model,X,n):
        gradient = self.gradient(mesh,model,X)
        dir_deriv = 0
        norm = 0
        for x_i,dx_i in zip(n,gradient):
            dir_deriv += x_i*dx_i
            norm += x_i**2
        norm = tf.sqrt(norm)
        dir_deriv *= (1/norm)

        return dir_deriv

    ###################################################

    # Define loss of the PDE
    def residual_loss(self,mesh,model,X):
        x,y,z = X
        r = self.laplacian(mesh,model,X) - self.source(x,y,z)        
        Loss_r = tf.reduce_mean(tf.square(r))
        return Loss_r


    # Other losses

    # Dirichlet
    def dirichlet_loss(self,mesh,model,XD,UD):
        Loss_d = 0
        for i in range(len(XD)):
            u_pred = model(XD[i])
            loss = tf.reduce_mean(tf.square(UD[i] - u_pred)) 
            Loss_d += loss
        return Loss_d


    def neumann_loss(self,mesh,model,XN,UN,V=None):
        Loss_n = 0
        for i in range(len(XN)):
            X = mesh.get_X(XN[i])
            grad = self.directional_gradient(mesh,model,X,self.normal_vector(X))
            loss = tf.reduce_mean(tf.square(UN[i]-grad))
            Loss_n += loss
        return Loss_n



    def analytic(self,x,y,z):
        return (-1/(self.epsilon*4*self.pi))*1/(tf.sqrt(x**2+y**2+z**2))




class PDE_Model_2():

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

        lb = tf.constant([self.xmin, self.ymin, self.zmin], dtype=self.DTYPE)
        ub = tf.constant([self.xmax, self.ymax, self.zmax], dtype=self.DTYPE)

        return (lb,ub)

    # Define boundary condition
    def fun_u_b(self,x, y, z, value):
        n = x.shape[0]
        return tf.ones((n,1), dtype=self.DTYPE)*value

    def fun_ux_b(self,x, y, z, value):
        n = x.shape[0]
        return tf.ones((n,1), dtype=self.DTYPE)*value

    def source(self,x,y,z):
        return 0

    # Define residual of the PDE
    def fun_r(self,x,u_x,u_xx,y,u_y,u_yy,z,u_z,u_zz):
        return u_xx + u_yy + u_zz - self.source(x,y,z)

    
    def analytic(self,x,y,z):
        return (-1/(self.epsilon*4*self.pi))*1/(tf.sqrt(x**2+y**2+z**2))
        