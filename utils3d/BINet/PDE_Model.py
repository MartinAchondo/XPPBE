import tensorflow as tf
import numpy as np
import bempp.api


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
    

    def adapt_PDE_mesh(self,mesh):
        self.mesh = mesh
        self.lb = mesh.lb
        self.ub = mesh.ub

        self.X_r = self.mesh.data_mesh['residual']
        self.XD_data,self.UD_data = self.mesh.data_mesh['dirichlet']
        self.XN_data,self.UN_data,self.derN = self.mesh.data_mesh['neumann']
        self.XI_data,self.derI = self.mesh.data_mesh['interface']
        self.x,self.y,self.z = self.mesh.get_X(self.X_r)

        self.create_potentials()


    
    def get_loss(self,model):
        L = dict()
        L['r'] = 0
        L['D'] = 0
        L['N'] = 0

        #residual
        X = (self.x,self.y,self.z)
        loss_r = self.residual_loss(self.mesh,model,X)
        L['r'] += loss_r

        return L


    # Define loss of the PDE
    def residual_loss(self,mesh,model,X): 
        x,y,z = X
        R = self.mesh.stack_X(x,y,z)
        ud = tf.transpose(model(R)).numpy()

        self.slp_grid = bempp.api.GridFunction(self.slp_space, coefficients=ud)
        self.dlp_grid = bempp.api.GridFunction(self.dlp_space, coefficients=ud)

        r =  self.slp_pot*self.slp_grid - self.dlp_pot*self.dlp_grid      
        Loss_r = tf.reduce_mean(tf.square(r))
        return Loss_r



    ####################################################################################################################################################
    
    def analytic(self,x,y,z):
        return (-1/(self.epsilon*4*self.pi))*1/(tf.sqrt(x**2+y**2+z**2))


    ####################################################################################################################################################


    def create_potentials(self):
        self.slp_space = bempp.api.function_space(self.mesh.grid, "P", 1)
        self.dlp_space = bempp.api.function_space(self.mesh.grid, "P", 1)
        self.ident_space = bempp.api.function_space(self.mesh.grid, "P", 1)

        X = self.mesh.to_bempp(self.X_r)
        self.slp_pot = bempp.api.operators.potential.laplace.single_layer(
            self.slp_space, X)
        self.dlp_pot = bempp.api.operators.potential.laplace.double_layer(
            self.dlp_space, X)
        self.ident_pot = bempp.api.operators.boundary.sparse.identity(
            self.ident_space, self.ident_space, X)
