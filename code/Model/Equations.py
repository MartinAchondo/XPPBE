import numpy as np
import tensorflow as tf
import bempp.api

from Model.PDE_Model import PBE

# PBE Equations and schemes 

class PBE_Std(PBE):

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

        self.PDE_in = Poisson(self, field='phi')
        if self.equation=='linear':
            self.PDE_out = Helmholtz(self, field='phi')
        elif self.equation=='nonlinear':
            self.PDE_out = Non_Linear(self, field='phi')

    def get_phi(self,X,flag,model,value='phi'):
        if flag=='molecule':
            phi = tf.reshape(model(X,flag)[:,0], (-1,1))
        elif flag=='solvent':
            phi = tf.reshape(model(X,flag)[:,1], (-1,1))
        elif flag=='interface':
            phi = model(X,flag)

        if value=='phi':
            return phi 

        if value == 'react':
            if flag != 'interface':
                phi_r = phi - self.G(*self.mesh.get_X(X))
            elif flag =='interface':
                G_val = self.G(*self.mesh.get_X(X))
                phi_r = tf.stack([tf.reshape(phi[:,0],(-1,1))-G_val,tf.reshape(phi[:,1],(-1,1))-G_val], axis=1)
        return phi_r

    def get_dphi(self,X,Nv,flag,model,value='phi'):
        x = self.mesh.get_X(X)
        nv = self.mesh.get_X(Nv)
        du_1 = self.directional_gradient(self.mesh,model,x,nv,'molecule',value='phi')
        du_2 = self.directional_gradient(self.mesh,model,x,nv,'solvent',value='phi')
        if value=='react':
            du_1 -= self.dG_n(*x,Nv)
            du_2 -= self.dG_n(*x,Nv)
        return du_1,du_2

    def get_solvation_energy(self,model):

        u_interface,_,_ = self.get_phi_interface(model)
        u_interface = u_interface.flatten()
        _,du_1,du_2 = self.get_dphi_interface(model)
        du_1 = du_1.flatten()
        du_2 = du_2.flatten()
        du_1_interface = (du_1+du_2*self.PDE_out.epsilon/self.PDE_in.epsilon)/2

        phi = bempp.api.GridFunction(self.space, coefficients=u_interface)
        dphi = bempp.api.GridFunction(self.space, coefficients=du_1_interface)

        phi_q = self.slp_q * dphi - self.dlp_q * phi
        
        G_solv = 0.5*np.sum(self.qs * phi_q).real
        G_solv *= self.to_V*self.qe*self.Na*(10**-3/self.cal2j)   
        
        return G_solv


class PBE_Reg_1(PBE):

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

        self.PDE_in = Laplace(self, field='react')
        if self.equation=='linear':
            self.PDE_out = Helmholtz(self, field='react')
        elif self.equation=='nonlinear':
            self.PDE_out = Non_Linear(self, field='react')

    def get_phi(self,X,flag,model,value='phi'):
        if flag=='molecule':
            phi_r = tf.reshape(model(X,flag)[:,0], (-1,1)) 
        elif flag=='solvent':
            phi_r = tf.reshape(model(X,flag)[:,1], (-1,1))
        elif flag=='interface':
            phi_r = model(X,flag)
        
        if value =='react':
            return phi_r
        
        if value == 'phi':
            if flag != 'interface':
                phi = phi_r + self.G(*self.mesh.get_X(X))
            elif flag =='interface':
                G_val = self.G(*self.mesh.get_X(X))
                phi = tf.stack([tf.reshape(phi_r[:,0],(-1,1))+G_val,tf.reshape(phi_r[:,1],(-1,1))+G_val], axis=1)
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

    def get_solvation_energy(self,model):
        X = self.x_qs
        phi_q = self.get_phi(X,'molecule',model,'react')
        G_solv = 0.5*np.sum(self.qs * phi_q)
        G_solv *= self.to_V*self.qe*self.Na*(10**-3/self.cal2j)   
        return G_solv


class PBE_Reg_2(PBE):

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

        self.PDE_in = Laplace(self, field='react')
        if self.equation=='linear':
            self.PDE_out = Helmholtz(self, field='phi')
        elif self.equation=='nonlinear':
            self.PDE_out = Non_Linear(self, field='phi')

    def get_phi(self,X,flag,model,value='phi'):
        if flag=='molecule':
            phi_r = tf.reshape(model(X,flag)[:,0], (-1,1)) 
        elif flag=='solvent':
            phi_r = tf.reshape(model(X,flag)[:,1], (-1,1)) - self.G(*self.mesh.get_X(X))
        elif flag=='interface':
            phi_t = model(X,flag)
            G_val = self.G(*self.mesh.get_X(X))
            phi_r = tf.stack([tf.reshape(phi_t[:,0],(-1,1)),tf.reshape(phi_t[:,1],(-1,1))-G_val], axis=1)

        if value =='react':
            return phi_r
        
        if value == 'phi':
            if flag != 'interface':
                phi = phi_r + self.G(*self.mesh.get_X(X))
            elif flag =='interface':
                G_val = self.G(*self.mesh.get_X(X))
                phi = tf.stack([tf.reshape(phi_r[:,0],(-1,1))+G_val,tf.reshape(phi_r[:,1],(-1,1))+G_val], axis=1)
        return phi 

    def get_dphi(self,X,Nv,flag,model,value='phi'):
        x = self.mesh.get_X(X)
        nv = self.mesh.get_X(Nv)
        du_1 = self.directional_gradient(self.mesh,model,x,nv,'molecule',value='react')
        du_2 = self.directional_gradient(self.mesh,model,x,nv,'solvent',value='phi')
        if value=='phi':
            du_1 += self.dG_n(*x,Nv)
        elif value =='react':
            du_2 -= self.dG_n(*x,Nv)
        return du_1,du_2
    
    def get_solvation_energy(self,model):
        X = self.x_qs
        phi_q = self.get_phi(X,'molecule',model,'react')
        G_solv = 0.5*np.sum(self.qs * phi_q)
        G_solv *= self.to_V*self.qe*self.Na*(10**-3/self.cal2j)   
        return G_solv
    

# Residuals
class Equations_utils():

    def __init__(self, PBE, field):
        
        self.PBE = PBE
        self.field = field

    def residual_loss(self,mesh,model,X,SU,flag):
        r = self.get_r(mesh,model,X,SU,flag)     
        Loss_r = tf.reduce_mean(tf.square(r))
        return Loss_r


class Laplace(Equations_utils):

    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)

    def get_r(self,mesh,model,X,SU,flag):
        r = self.PBE.laplacian(mesh,model,X,flag,value=self.field)     
        return r


class Poisson(Equations_utils):

    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)

    def get_r(self,mesh,model,X,SU,flag):
        x,y,z = X
        source = self.PBE.source(x,y,z) if SU==None else SU
        r = self.PBE.laplacian(mesh,model,X,flag, value=self.field) - source   
        return r


class Helmholtz(Equations_utils):

    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)

    def get_r(self,mesh,model,X,SU,flag):
        x,y,z = X
        R = mesh.stack_X(x,y,z)
        u = self.PBE.get_phi(R,flag,model,value=self.field)
        r = self.PBE.laplacian(mesh,model,X,flag,value=self.field) - self.PBE.kappa**2*u      
        return r  


class Non_Linear(Equations_utils):

    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)

    def get_r(self,mesh,model,X,SU,flag):
        x,y,z = X
        R = mesh.stack_X(x,y,z)
        u = self.PBE.get_phi(R,flag,model,value=self.field)
        r = self.PBE.laplacian(mesh,model,X,flag,value=self.field) - self.PBE.kappa**2*tf.math.sinh(u)     
        return r
    