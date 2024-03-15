import os
import tensorflow as tf
import numpy as np


from Model.PDE_Model import PBE

class PBE_Reg(PBE):

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

        self.PDE_in = Laplace(self,self.inputs)
        if self.eq=='linear':
            self.PDE_out = Linear(self,self.inputs)
        elif self.eq=='nonlinear':
            self.PDE_out = Non_Linear(self,self.inputs)


    def get_phi(self,X,flag,model,value='phi'):
        if flag=='molecule':
            phi = tf.reshape(model(X,flag)[:,0], (-1,1)) 
        elif flag=='solvent':
            phi = tf.reshape(model(X,flag)[:,1], (-1,1))
        elif flag=='interface':
            phi = model(X,flag)
        if value =='react':
            return phi
        
        if value == 'phi':
            if flag != 'interface':
                phi += self.G(*self.mesh.get_X(X))
            elif flag =='interface':
                G_val = self.G(*self.mesh.get_X(X))
                return tf.stack([tf.reshape(phi[:,0],(-1,1))+G_val,tf.reshape(phi[:,1],(-1,1))+G_val], axis=1)
            return phi 

    def get_dphi(self,X,Nv,flag,model,value='phi'):
        x = self.mesh.get_X(X)
        nv = self.mesh.get_X(Nv)
        du_1 = self.directional_gradient(self.mesh,model,x,nv,'molecule',value='react')
        du_2 = self.directional_gradient(self.mesh,model,x,nv,'solvent',value='react')
        if value=='phi':
            du_1 += self.dG_n(*x,Nv)
            du_2 += self.dG_n(*x,Nv)
        return du_1,du_2

    def get_dphi_interface(self,model, value='phi'): 
        verts = tf.constant(self.mesh.mol_verts, dtype=self.DTYPE)     
        X = self.mesh.get_X(verts)
        n_v = self.mesh.get_X(self.mesh.mol_normal)
        du_1 = self.directional_gradient(self.mesh,model,X,n_v,'molecule',value='react')
        du_2 = self.directional_gradient(self.mesh,model,X,n_v,'solvent',value='react')
        if value=='phi':
            du_1 += self.dG_n(*X,self.mesh.mol_normal)
            du_2 += self.dG_n(*X,self.mesh.mol_normal)
        du_prom = (du_1*self.PDE_in.epsilon + du_2*self.PDE_out.epsilon)/2
        return du_prom.numpy(),du_1.numpy(),du_2.numpy()
    
    
    def get_loss_I(self,model,XI_data,loss_type=[True,True]):
        
        loss = 0
        ((XI,N_v),flag) = XI_data
        X = self.mesh.get_X(XI)

        if loss_type[0]:
            u = self.get_phi(XI,flag,model,value='react')
            loss += tf.reduce_mean(tf.square(u[:,0]-u[:,1])) 

        if loss_type[1]:
            n_v = self.mesh.get_X(N_v)
            du_1 = self.directional_gradient(self.mesh,model,X,n_v,'molecule')
            du_2 = self.directional_gradient(self.mesh,model,X,n_v,'solvent')

            loss += tf.reduce_mean(tf.square(du_1*self.PDE_in.epsilon - du_2*self.PDE_out.epsilon -(self.PDE_out.epsilon-self.PDE_in.epsilon)*self.dG_n(*X,N_v)))
            
        return loss

    
    def get_loss_Gauss(self,model,XI_data):
        loss = 0
        ((XI,N_v,areas),flag) = XI_data
        X = self.mesh.get_X(XI)
        n_v = self.mesh.get_X(N_v)
        du_1 = self.directional_gradient(self.mesh,model,X,n_v,'molecule',value='react')+self.dG_n(*X,N_v)
        du_2 = self.directional_gradient(self.mesh,model,X,n_v,'solvent',value='react')+self.dG_n(*X,N_v)
        du_prom = (du_1*self.PDE_in.epsilon + du_2*self.PDE_out.epsilon)/2

        integral = tf.reduce_sum(du_prom * areas)
        loss += tf.reduce_mean(tf.square(integral - self.total_charge))

        return loss
    


class Laplace():

    def __init__(self, PBE, inputs):
        
        self.PBE = PBE
        for key, value in inputs.items():
            setattr(self, key, value)
        self.epsilon = self.epsilon_1
        super().__init__()

    def residual_loss(self,mesh,model,X,SU,flag):
        x,y,z = X
        r = self.PBE.laplacian(mesh,model,X,flag,value='react')     
        Loss_r = tf.reduce_mean(tf.square(r))
        return Loss_r


class Linear():

    def __init__(self, PBE, inputs):

        self.PBE = PBE
        for key, value in inputs.items():
            setattr(self, key, value)
        self.epsilon = self.epsilon_2
        super().__init__()

    def residual_loss(self,mesh,model,X,SU,flag):
        x,y,z = X
        R = mesh.stack_X(x,y,z)
        u = self.PBE.get_phi(R,flag,model,value='react')
        r = self.PBE.laplacian(mesh,model,X,flag,value='react') - self.kappa**2*(u)      
        Loss_r = tf.reduce_mean(tf.square(r))
        return Loss_r  


class Non_Linear():

    def __init__(self, PBE, inputs):

        self.PBE = PBE
        for key, value in inputs.items():
            setattr(self, key, value)
        self.epsilon = self.epsilon_2
        super().__init__()

    def residual_loss(self,mesh,model,X,SU,flag):
        x,y,z = X
        R = mesh.stack_X(x,y,z)
        u = self.PBE.get_phi(R,flag,model,value='react')
        r = self.PBE.laplacian(mesh,model,X,flag,value='react') - self.kappa**2*tf.math.sinh(u)     
        Loss_r = tf.reduce_mean(tf.square(r))
        return Loss_r
    