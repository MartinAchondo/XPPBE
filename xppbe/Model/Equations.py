import tensorflow as tf
import bempp.api
import numpy as np

from xppbe.Model.PDE_Model import PBE

# PBE Equations and schemes 

class PBE_Std(PBE):

    def __init__(self,*args,**kwargs):
        self.scheme = 'standard'
        super().__init__(*args,**kwargs)

        self.PDE_in = Poisson(self,self.domain_properties,field='phi')
        if self.equation=='linear':
            self.PDE_out = Helmholtz(self,self.domain_properties,field='phi')
        elif self.equation=='nonlinear':
            self.PDE_out = Non_Linear(self,self.domain_properties,field='phi')

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
            G_val = tf.stop_gradient(self.G(X))
            if flag != 'interface':
                phi_r = phi - G_val
            elif flag =='interface':
                phi_r = tf.stack([tf.reshape(phi[:,0],(-1,1))-G_val,tf.reshape(phi[:,1],(-1,1))-G_val], axis=1)
        return phi_r

    def get_dphi(self,X,Nv,flag,model,value='phi'):
        x = self.mesh.get_X(X)
        nv = self.mesh.get_X(Nv)
        du_1 = self.directional_gradient(self.mesh,model,x,nv,'molecule',value='phi')
        du_2 = self.directional_gradient(self.mesh,model,x,nv,'solvent',value='phi')
        if value=='react':
            dG_dn = tf.stop_gradient(self.dG_n(X,Nv))
            du_1 -= dG_dn
            du_2 -= dG_dn
        return du_1,du_2

    def get_solvation_energy(self,model):

        u_interface,_,_ = self.get_phi_interface(model)
        u_interface = u_interface.numpy().flatten()
        _,du_1,du_2 = self.get_dphi_interface(model)
        du_1 = du_1.numpy().flatten()
        du_2 = du_2.numpy().flatten()
        du_1_interface = (du_1+du_2*self.PDE_out.epsilon/self.PDE_in.epsilon)/2
        du_1_interface = du_1_interface.numpy()

        phi = bempp.api.GridFunction(self.space, coefficients=u_interface)
        dphi = bempp.api.GridFunction(self.space, coefficients=du_1_interface)

        phi_q = self.slp_q * dphi - self.dlp_q * phi
        
        G_solv = self.solvation_energy_phi_qs(phi_q.real)
        
        return G_solv


class PBE_Reg_1(PBE):

    def __init__(self,*args,**kwargs):
        self.scheme = 'regularized_1'
        super().__init__(*args,**kwargs)

        self.PDE_in = Laplace(self,self.domain_properties,field='react')
        if self.equation=='linear':
            self.PDE_out = Helmholtz(self,self.domain_properties,field='react')
        elif self.equation=='nonlinear':
            self.PDE_out = Non_Linear(self,self.domain_properties,field='react')

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
            G_val = tf.stop_gradient(self.G(X))
            if flag != 'interface':
                phi = phi_r + G_val
            elif flag =='interface':
                phi = tf.stack([tf.reshape(phi_r[:,0],(-1,1))+G_val,tf.reshape(phi_r[:,1],(-1,1))+G_val], axis=1)
        return phi 

    def get_dphi(self,X,Nv,flag,model,value='phi'):
        x = self.mesh.get_X(X)
        nv = self.mesh.get_X(Nv)
        du_1 = self.directional_gradient(self.mesh,model,x,nv,'molecule',value='react')
        du_2 = self.directional_gradient(self.mesh,model,x,nv,'solvent',value='react')
        if value=='phi':
            dG_dn = tf.stop_gradient(self.dG_n(X,Nv))
            du_1 += dG_dn
            du_2 += dG_dn
        return du_1,du_2

    def get_solvation_energy(self,model):
        X = self.x_qs
        phi_q = self.get_phi(X,'molecule',model,'react')
        phi_q = phi_q.numpy().reshape(-1)
        G_solv = self.solvation_energy_phi_qs(phi_q)  
        return G_solv


class PBE_Reg_2(PBE):

    def __init__(self,*args,**kwargs):
        self.scheme = 'regularized_2'
        super().__init__(*args,**kwargs)

        self.PDE_in = Laplace(self,self.domain_properties,field='react')
        if self.equation=='linear':
            self.PDE_out = Helmholtz(self,self.domain_properties,field='phi')
        elif self.equation=='nonlinear':
            self.PDE_out = Non_Linear(self,self.domain_properties,field='phi')

    def get_phi(self,X,flag,model,value='phi'):
        if flag=='molecule':
            phi_r = tf.reshape(model(X,flag)[:,0], (-1,1)) 
        elif flag=='solvent':
            phi_r = tf.reshape(model(X,flag)[:,1], (-1,1)) - tf.stop_gradient(self.G(X))
        elif flag=='interface':
            phi_t = model(X,flag)
            G_val = tf.stop_gradient(self.G(X))
            phi_r = tf.stack([tf.reshape(phi_t[:,0],(-1,1)),tf.reshape(phi_t[:,1],(-1,1))-G_val], axis=1)

        if value =='react':
            return phi_r
        
        if value == 'phi':
            G_val = tf.stop_gradient(self.G(X))
            if flag != 'interface':
                phi = phi_r + G_val
            elif flag =='interface':
                phi = tf.stack([tf.reshape(phi_r[:,0],(-1,1))+G_val,tf.reshape(phi_r[:,1],(-1,1))+G_val], axis=1)
        return phi 

    def get_dphi(self,X,Nv,flag,model,value='phi'):
        x = self.mesh.get_X(X)
        nv = self.mesh.get_X(Nv)
        du_1 = self.directional_gradient(self.mesh,model,x,nv,'molecule',value='react')
        du_2 = self.directional_gradient(self.mesh,model,x,nv,'solvent',value='phi')
        dG_dn = tf.stop_gradient(self.dG_n(X,Nv))
        if value=='phi':
            du_1 += dG_dn
        elif value =='react':
            du_2 -= dG_dn
        return du_1,du_2
    
    def get_solvation_energy(self,model):
        X = self.x_qs
        phi_q = self.get_phi(X,'molecule',model,'react')
        phi_q = phi_q.numpy().reshape(-1)
        G_solv = self.solvation_energy_phi_qs(phi_q)  
        return G_solv
    

# Residuals
class Equations_utils():

    DTYPE = 'float32'

    def __init__(self, PBE, domain_properties, field):
        
        self.PBE = PBE
        self.field = field
        for key, value in domain_properties.items():
            if key != 'molecule':
                setattr(self, key, tf.constant(value, dtype=self.DTYPE))
            else:
                setattr(self, key, value)

    def residual_loss(self,mesh,model,X,SU,flag):
        r = self.get_r(mesh,model,X,SU,flag)     
        Loss_r = tf.reduce_mean(tf.square(r))
        return Loss_r


class Laplace(Equations_utils):

    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.epsilon = self.epsilon_1

    def get_r(self,mesh,model,X,SU,flag):
        r = self.PBE.laplacian(mesh,model,X,flag,value=self.field)     
        return r


class Poisson(Equations_utils):

    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.epsilon = self.epsilon_1

    def get_r(self,mesh,model,X,SU,flag):
        x,y,z = X
        source = self.PBE.source(mesh.stack_X(x,y,z)) if SU==None else SU
        r = self.PBE.laplacian(mesh,model,X,flag, value=self.field) - source   
        return r


class Helmholtz(Equations_utils):

    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.epsilon = self.epsilon_2

    def get_r(self,mesh,model,X,SU,flag):
        x,y,z = X
        R = mesh.stack_X(x,y,z)
        u = self.PBE.get_phi(R,flag,model,value=self.field)
        r = self.PBE.laplacian(mesh,model,X,flag,value=self.field) - self.kappa**2*u  
        return r  


class Non_Linear(Equations_utils):

    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.epsilon = self.epsilon_2

    def get_r(self,mesh,model,X,SU,flag):
        x,y,z = X
        R = mesh.stack_X(x,y,z)
        u = self.PBE.get_phi(R,flag,model,value=self.field)
        r = self.PBE.laplacian(mesh,model,X,flag,value=self.field) - self.kappa**2*tf.math.sinh(u)     
        return r
    