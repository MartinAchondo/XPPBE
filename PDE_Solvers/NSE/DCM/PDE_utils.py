import tensorflow as tf
import numpy as np


class PDE_utils():

    def __init__(self):

        self.DTYPE='float32'
        self.pi = tf.constant(np.pi, dtype=self.DTYPE)
    
    def set_domain(self,X):
        x,y,t = X
        self.xmin,self.xmax = x
        self.ymin,self.ymax = y
        self.tmin,self.tmax = t

        lb = tf.constant([self.xmin, self.ymin,self.tmin], dtype=self.DTYPE)
        ub = tf.constant([self.xmax, self.ymax,self.tmax], dtype=self.DTYPE)

        return (lb,ub)
    

    def adapt_PDE_mesh(self,mesh):
        self.mesh = mesh
        self.lb = mesh.lb
        self.ub = mesh.ub

        self.X_r = self.mesh.data_mesh['residual']
        self.XD_data,self.UD_data = self.mesh.data_mesh['dirichlet']
        self.XN_data,self.UN_data,self.derN = self.mesh.data_mesh['neumann']
        self.XI_data,self.derI = self.mesh.data_mesh['interface']
        self.XK_data,self.UK_data = self.mesh.data_mesh['data_known']
        self.X_r_P = self.mesh.data_mesh['precondition']

        self.x,self.y,self.t = self.mesh.get_X(self.X_r)
        if self.X_r_P != None:
            self.xP,self.yP,self.tP = self.mesh.get_X(self.X_r_P)

    
    def get_loss(self, X_batch, model):
        L = dict()
        L['r'] = 0
        L['D'] = 0
        L['N'] = 0
        L['K'] = 0

        #residual
        X = (self.x,self.y,self.t)
        #X = self.mesh.get_X(X_batch)
        loss_r = self.residual_loss(self.mesh,model,X)
        L['r'] += loss_r

        #dirichlet 
        loss_d = self.dirichlet_loss(self.mesh,model,self.XD_data,self.UD_data)
        L['D'] += loss_d

        #neumann
        loss_n = self.neumann_loss(self.mesh,model,self.XN_data,self.UN_data)
        L['N'] += loss_n

        # data known
        loss_k = self.data_known_loss(self.mesh,model,self.XK_data,self.UK_data)
        L['K'] += loss_k    

        return L


    # Dirichlet
    def dirichlet_loss(self,mesh,model,XD,UD):
        Loss_d = 0
        for i in range(len(XD)):
            O = model(XD[i])
            u = O[:,0]
            v = O[:,1]
            p = O[:,2]
            loss = tf.reduce_mean(tf.square(UD[i][:,0] - u)) + tf.reduce_mean(tf.square(UD[i][:,1] - v)) + tf.reduce_mean(tf.square(UD[i][:,2] - p)) 
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
    
        # Dirichlet
    def data_known_loss(self,mesh,model,XK,UK):
        Loss_d = 0
        for i in range(len(XK)):
            u_pred = model(XK[i])
            loss = tf.reduce_mean(tf.square(UK[i] - u_pred)) 
            Loss_d += loss
        return Loss_d


    def get_loss_preconditioner(self, X_batch, model):
        L = dict()
        L['r'] = 0
        L['D'] = 0
        L['N'] = 0
        L['I'] = 0
        L['K'] = 0

        #residual
        X = self.mesh.get_X(X_batch)
        loss_r = self.preconditioner(self.mesh,model,X)
        L['r'] += loss_r

        return L


    ####################################################################################################################################################

    # Define boundary condition
    def fun_u_b(self,x, y, t, value):
        n = x.shape[0]
        v1,v2,v3 = value
        u = tf.ones((n,1), dtype=self.DTYPE)*v1
        v = tf.ones((n,1), dtype=self.DTYPE)*v2
        p = tf.ones((n,1), dtype=self.DTYPE)*v3
        return u,v,p

    def fun_ux_b(self,x, y, t, value):
        n = x.shape[0]
        return tf.ones((n,1), dtype=self.DTYPE)*value


    ####################################################################################################################################################
