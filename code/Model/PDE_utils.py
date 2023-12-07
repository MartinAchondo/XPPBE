import tensorflow as tf
import numpy as np


class PDE_utils():

    DTYPE = 'float32'
    pi = tf.constant(np.pi, dtype=DTYPE)

    def __init__(self):
        pass
    
    def get_loss_PINN(self, X_batches, model):
        L = self.create_L()

        #residual
        if 'R' in self.mesh.solver_mesh_names: 
            X,SU = X_batches['R']
            loss_r = self.residual_loss(self.mesh,model,self.mesh.get_X(X),SU)
            L['R'] += loss_r   

        if 'Q' in self.mesh.solver_mesh_names: 
            X,SU = X_batches['Q']
            loss_q = self.residual_loss(self.mesh,model,self.mesh.get_X(X),SU)
            L['Q'] += loss_q 

        #dirichlet 
        if 'D' in self.mesh.solver_mesh_names:
            X,U = X_batches['D']
            loss_d = self.dirichlet_loss(self.mesh,model,X,U)
            L['D'] += loss_d

        #neumann
        if 'N' in self.mesh.solver_mesh_names:
            X,U = X_batches['K']
            loss_n = self.neumann_loss(self.mesh,model,X,U)
            L['N'] += loss_n

        # data known
        if 'K' in self.mesh.solver_mesh_names:
            X,U = X_batches['K']
            loss_k = self.data_known_loss(self.mesh,model,X,U)
            L['K'] += loss_k    

        return L


    def dirichlet_loss(self,mesh,model,XD,UD):
        Loss_d = 0
        u_pred = model(XD)
        loss = tf.reduce_mean(tf.square(UD - u_pred)) 
        Loss_d += loss
        return Loss_d

    def neumann_loss(self,mesh,model,XN,UN,V=None):
        Loss_n = 0
        X = mesh.get_X(XN)
        grad = self.directional_gradient(mesh,model,X,self.normal_vector(X))
        loss = tf.reduce_mean(tf.square(UN-grad))
        Loss_n += loss
        return Loss_n
    
    def data_known_loss(self,mesh,model,XK,UK):
        Loss_d = 0
        u_pred = model(XK)
        loss = tf.reduce_mean(tf.square(UK - u_pred)) 
        Loss_d += loss
        return Loss_d
    
    def get_loss_XPINN(self,solvers_t,solvers_i,X_domain):
        L = self.create_L()

        if 'Iu' and 'Id' in self.mesh.domain_mesh_names:
            L['Iu'] += self.get_loss_I(solvers_i[0],solvers_i[1],X_domain['I'], [True,False])
            L['Id'] += self.get_loss_I(solvers_i[0],solvers_i[1],X_domain['I'], [False,True])

        if 'E' in self.mesh.domain_mesh_names:
            L['E'] += self.get_loss_experimental(solvers_t,X_domain['E'])

        if 'G' in self.mesh.domain_mesh_names:
            L['G'] += self.get_loss_Gauss(solvers_t,X_domain['I'])
        return L

    def get_loss_preconditioner_PINN(self, X_batches, model):
        L = self.create_L()

        #residual
        if 'P' in self.mesh.solver_mesh_names:
            X,U = X_batches['P']
            loss_p = self.data_known_loss(self.mesh,model,X,U)
            L['P'] += loss_p  
            
        return L
    
    @classmethod
    def create_L(cls):
        L = dict()
        L['R'] = tf.constant(0.0, dtype=cls.DTYPE)
        L['D'] = tf.constant(0.0, dtype=cls.DTYPE)
        L['N'] = tf.constant(0.0, dtype=cls.DTYPE)
        L['I'] = tf.constant(0.0, dtype=cls.DTYPE)
        L['Iu'] = tf.constant(0.0, dtype=cls.DTYPE)
        L['Id'] = tf.constant(0.0, dtype=cls.DTYPE)
        L['Q'] = tf.constant(0.0, dtype=cls.DTYPE)
        L['K'] = tf.constant(0.0, dtype=cls.DTYPE)
        L['P'] = tf.constant(0.0, dtype=cls.DTYPE)
        L['E'] = tf.constant(0.0, dtype=cls.DTYPE)
        L['G'] = tf.constant(0.0, dtype=cls.DTYPE)
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
    
    def directional_gradient(self,mesh,model,X,n_v):
        gradient = self.gradient(mesh,model,X)
        dir_deriv = 0
        for j in range(3):
            dir_deriv += n_v[j]*gradient[j]

        return dir_deriv