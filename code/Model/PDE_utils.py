import numpy as np
import tensorflow as tf


class PDE_utils():

    DTYPE = 'float32'
    pi = tf.constant(np.pi, dtype=DTYPE)

    def __init__(self):
        pass
    
    def get_loss(self, X_batches, model, validation=False):
        L = self.create_L()

        #residual
        if 'R1' in X_batches: 
            ((X,SU),flag) = X_batches['R1']
            loss_r = self.PDE_in.residual_loss(self.mesh,model,self.mesh.get_X(X),SU,flag)
            L['R1'] += loss_r   

        if 'R2' in X_batches: 
            ((X,SU),flag) = X_batches['R2']
            loss_r = self.PDE_out.residual_loss(self.mesh,model,self.mesh.get_X(X),SU,flag)
            L['R2'] += loss_r   

        if 'Q1' in X_batches: 
            ((X,SU),flag) = X_batches['Q1']
            loss_q = self.PDE_in.residual_loss(self.mesh,model,self.mesh.get_X(X),SU,flag)
            L['Q1'] += loss_q 

        #dirichlet 
        if 'D2' in X_batches:
            ((X,U),flag) = X_batches['D2']
            loss_d = self.dirichlet_loss(self.mesh,model,X,U,flag)
            L['D2'] += loss_d

        # data known
        if 'K1' in X_batches and not validation:
            ((X,U),flag) = X_batches['K1']
            loss_k = self.data_known_loss(self.mesh,model,X,U,flag)
            L['K1'] += loss_k   

        if 'K2' in X_batches and not validation:
            ((X,U),flag) = X_batches['K2']
            loss_k = self.data_known_loss(self.mesh,model,X,U,flag)
            L['K2'] += loss_k 

        #neumann
        if 'N1' in X_batches:
            ((X,U),flag) = X_batches['N1']
            loss_n = self.neumann_loss(self.mesh,model,X,U,flag)
            L['N1'] += loss_n

        if 'I' in X_batches:
            L['Iu'] += self.get_loss_I(model,X_batches['I'], [True,False])
            L['Id'] += self.get_loss_I(model,X_batches['I'], [False,True])

        if 'E2' in X_batches and not validation:
            L['E2'] += self.get_loss_experimental(model,X_batches['E2'])

        if 'G' in X_batches and not validation:
            L['G'] += self.get_loss_Gauss(model,X_batches['G'])

        return L


    def dirichlet_loss(self,mesh,model,XD,UD,flag):
        num = 0 if flag=='molecule' else 1
        Loss_d = 0
        u_pred = model(XD,flag)[:,num]
        loss = tf.reduce_mean(tf.square(UD - u_pred)) 
        Loss_d += loss
        return Loss_d

    def neumann_loss(self,mesh,model,XN,UN,flag,V=None):
        num = 0 if flag=='molecule' else 1
        Loss_n = 0
        X = mesh.get_X(XN)[:,num]
        grad = self.directional_gradient(mesh,model,X,self.normal_vector(X),flag)
        loss = tf.reduce_mean(tf.square(UN-grad))
        Loss_n += loss
        return Loss_n
    
    def data_known_loss(self,mesh,model,XK,UK,flag):
        num = 0 if flag=='molecule' else 1
        Loss_d = 0
        u_pred = model(XK,flag)[:,num]
        loss = tf.reduce_mean(tf.square(UK - u_pred)) 
        Loss_d += loss
        return Loss_d
    


    def get_loss_preconditioner(self, X_batches, model):
        L = self.create_L()

        #residual
        if 'P1' in X_batches:
            ((X,U),flag) = X_batches['P1']
            loss_p = self.data_known_loss(self.mesh,model,X,U,flag)
            L['P1'] += loss_p  

        if 'P2' in X_batches:
            ((X,U),flag) = X_batches['P2']
            loss_p = self.data_known_loss(self.mesh,model,X,U,flag)
            L['P2'] += loss_p  
            
        return L
    
    @classmethod
    def create_L(cls):
        cls.names = ['R1','D1','N1','K1','Q1','R2','D2','N2','K2','G','Iu','Id','E2','P1','P2']
        L = dict()
        for t in cls.names:
            L[t] = tf.constant(0.0, dtype=cls.DTYPE)
        return L
    
    ####################################################################################################################################################

    # Define boundary condition
    def fun_u_b(self,x, y, z, value):
        n = x.shape[0]
        return tf.ones((n,1), dtype=self.DTYPE)*value

    def fun_ux_b(self,x, y, z, value):
        n = x.shape[0]
        return tf.ones((n,1), dtype=self.DTYPE)*value

    ####################################################################################################################################################

    def aprox_exp(self,x):
        aprox = 1.0 + x + x**2/2.0 + x**3/6.0 + x**4/24.0
        return aprox


    # Differential operators

    def laplacian(self,mesh,model,X,flag):
        num = 0 if flag=='molecule' else 1
        x,y,z = X
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(y)
            tape.watch(z)
            R = mesh.stack_X(x,y,z)
            u = model(R,flag)[:,num]
            u_x = tape.gradient(u,x)
            u_y = tape.gradient(u,y)
            u_z = tape.gradient(u,z)
        u_xx = tape.gradient(u_x,x)
        u_yy = tape.gradient(u_y,y)
        u_zz = tape.gradient(u_z,z)
        del tape
        return u_xx + u_yy + u_zz

    def gradient(self,mesh,model,X,flag):
        num = 0 if flag=='molecule' else 1
        x,y,z = X
        with tf.GradientTape(persistent=True,watch_accessed_variables=False) as tape:
            tape.watch(x)
            tape.watch(y)
            tape.watch(z)
            R = mesh.stack_X(x,y,z)
            u = model(R,flag)[:,num]
        u_x = tape.gradient(u,x)
        u_y = tape.gradient(u,y)
        u_z = tape.gradient(u,z)
        del tape
        return (u_x,u_y,u_z)
    
    def directional_gradient(self,mesh,model,X,n_v,flag):
        gradient = self.gradient(mesh,model,X,flag)
        dir_deriv = 0
        for j in range(3):
            dir_deriv += n_v[j]*gradient[j]
        return dir_deriv