import numpy as np
import tensorflow as tf

from .PDE_Model import PBE

# PBE Equations and schemes 

class PBE_Direct(PBE):

    def __init__(self,*args,**kwargs):
        self.scheme = 'direct'
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
            phi = (tf.reshape(model(X,flag)[:,0], (-1,1))+tf.reshape(model(X,flag)[:,1], (-1,1)))/2

        if value=='phi':
            return phi 

        if value == 'react':
            G_val = tf.stop_gradient(self.G(X))
            phi_r = phi - G_val

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
        phi,dphi = self.get_grid_coefficients_faces(model)
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
            phi_r = (tf.reshape(model(X,flag)[:,0], (-1,1))+tf.reshape(model(X,flag)[:,1], (-1,1)))/2
        
        if value =='react':
            return phi_r
        
        if value == 'phi':
            G_val = tf.stop_gradient(self.G(X))
            phi = phi_r + G_val

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
            G_val = tf.stop_gradient(self.G(X))
            phi_r = (tf.reshape(model(X,flag)[:,0], (-1,1))+tf.reshape(model(X,flag)[:,1], (-1,1))-G_val)/2

        if value =='react':
            return phi_r
        
        if value == 'phi':
            G_val = tf.stop_gradient(self.G(X))
            phi = phi_r + G_val

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
    

class PBE_Var_Direct(PBE_Direct):

    def __init__(self,*args,**kwargs):
        self.scheme = 'direct'
        super().__init__(*args,**kwargs)

        self.PDE_in = Variational_Poisson(self,self.domain_properties,field='phi')
        if self.equation=='linear':
            self.PDE_out = Variational_Helmholtz(self,self.domain_properties,field='phi')
        elif self.equation=='nonlinear':
            self.PDE_out = Variational_Non_Linear(self,self.domain_properties,field='phi')

class PBE_Var_Reg_1(PBE_Reg_1):

    def __init__(self,*args,**kwargs):
        self.scheme = 'regularized_1'
        super().__init__(*args,**kwargs)

        self.PDE_in = Variational_Laplace(self,self.domain_properties,field='react')
        if self.equation=='linear':
            self.PDE_out = Variational_Helmholtz(self,self.domain_properties,field='react')
        elif self.equation=='nonlinear':
            self.PDE_out = Variational_Non_Linear(self,self.domain_properties,field='react')



class PBE_Bound(PBE):

    def __init__(self,*args,**kwargs):
        self.scheme = 'direct'
        super().__init__(*args,**kwargs)

        self.PDE_in = Boundary_Poisson(self,self.domain_properties,field='phi')
        self.PDE_out = Boundary_Helmholtz(self,self.domain_properties,field='phi')

    def get_phi(self,X,flag,model,value='phi'):
 
        if flag=='molecule':
            slp = self.bempp.operators.potential.laplace.single_layer(self.neumann_space, X.numpy().transpose())
            dlp = self.bempp.operators.potential.laplace.double_layer(self.dirichl_space, X.numpy().transpose())
            phi_c,dphi_c = self.get_grid_coefficients_faces(model)
            phi = slp * dphi_c - dlp * phi_c
            phi = phi.reshape(-1,1) + self.G(X)
            
        elif flag=='solvent':
            slp = self.bempp.operators.potential.modified_helmholtz.single_layer(self.neumann_space, X.numpy().transpose(),self.kappa)
            dlp = self.bempp.operators.potential.modified_helmholtz.double_layer(self.dirichl_space, X.numpy().transpose(),self.kappa)
            phi_c,dphi_c = self.get_grid_coefficients_faces(model)
            phi = slp * dphi_c - dlp * phi_c
            phi = phi.reshape(-1,1)
            
        elif flag=='interface':
            phi = tf.reshape(model(X,flag)[:,0]+model(X,flag)[:,1],(-1,1))/2

        if value=='phi':
            return phi 

        if value == 'react':
            G_val = tf.stop_gradient(self.G(X))
            phi_r = phi - G_val
            
        return phi_r

    def get_dphi(self,X,Nv,flag,model,value='phi'):
        x = self.mesh.get_X(X)
        nv = self.mesh.get_X(Nv)
        du_1 = self.directional_gradient(self.mesh,model,x,nv,'interface',value='phi')
        if value=='react':
            dG_dn = tf.stop_gradient(self.dG_n(X,Nv))
            du_1 -= dG_dn
        return du_1,du_1*self.epsilon_1/self.epsilon_2

    def get_solvation_energy(self,model):
        phi,dphi = self.get_grid_coefficients_faces(model)
        phi_q = self.slp_q * dphi - self.dlp_q * phi
        G_solv = self.solvation_energy_phi_qs(phi_q.real)
        return G_solv
    
    def get_phi_interface(self,X,model,**kwargs):      
        u = self.get_phi(X,'interface',model,**kwargs)
        u = u[:,0]
        return u,u,u
    
    def get_dphi_interface(self,X,N_v,model, value='phi'):
        du_1,du_2 = self.get_dphi(X,N_v,'',model,value)
        du_prom = du_1*self.epsilon_1
        return du_prom,du_1,du_2



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

        if self.field == 'phi':
            self.get_r = self.get_r_total
        elif self.field == 'react':
            self.get_r = self.get_r_reg

    def get_r_total(self,mesh,model,X,SU,flag):
        x,y,z = X
        R = mesh.stack_X(x,y,z)
        phi = self.PBE.get_phi(R,flag,model,value=self.field)
        r = self.PBE.laplacian(mesh,model,X,flag,value=self.field) - self.kappa**2*phi  
        return r  
    
    def get_r_reg(self,mesh,model,X,SU,flag):
        x,y,z = X
        R = mesh.stack_X(x,y,z)
        phi = self.PBE.get_phi(R,flag,model,value=self.field)
        r = self.PBE.laplacian(mesh,model,X,flag,value=self.field) - self.kappa**2*(phi+self.PBE.G(X))  
        return r 


class Non_Linear(Equations_utils):

    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.epsilon = self.epsilon_2

        if self.field == 'phi':
            self.get_r = self.get_r_total
        elif self.field == 'react':
            self.get_r = self.get_r_reg

    def get_r_total(self,mesh,model,X,SU,flag):
        x,y,z = X
        R = mesh.stack_X(x,y,z)
        phi = self.PBE.get_phi(R,flag,model,value=self.field)
        r = self.PBE.laplacian(mesh,model,X,flag,value=self.field) - self.kappa**2*self.T_adim*self.PBE.aprox_sinh(phi/self.T_adim)     
        return r
    
    def get_r_reg(self,mesh,model,X,SU,flag):
        x,y,z = X
        R = mesh.stack_X(x,y,z)
        phi = self.PBE.get_phi(R,flag,model,value=self.field)
        r = self.PBE.laplacian(mesh,model,X,flag,value=self.field) - self.kappa**2*self.T_adim*self.PBE.aprox_sinh((phi+self.PBE.G(X))/self.T_adim)     
        return r

class Variational_Laplace(Equations_utils):

    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.epsilon = self.epsilon_1

    def get_r(self,mesh,model,X,SU,flag):
        x,y,z = X
        R = mesh.stack_X(x,y,z)
        gx,gy,gz = self.PBE.gradient(mesh,model,X,flag,value=self.field)
        r = self.epsilon*(gx**2+gy**2+gz**2)/2   
        return r
    
class Variational_Poisson(Equations_utils):

    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.epsilon = self.epsilon_1

    def get_r(self,mesh,model,X,SU,flag):
        x,y,z = X
        R = mesh.stack_X(x,y,z)
        source = self.PBE.source(mesh.stack_X(x,y,z)) if SU==None else SU
        phi = self.PBE.get_phi(R,flag,model,value=self.field)
        gx,gy,gz = self.PBE.gradient(mesh,model,X,flag,value=self.field)
        r = self.epsilon*(gx**2+gy**2+gz**2)/2 - source*phi  
        return r
    
class Variational_Helmholtz(Equations_utils):

    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.epsilon = self.epsilon_2

        if self.field == 'phi':
            self.get_r = self.get_r_total
        elif self.field == 'react':
            self.get_r = self.get_r_reg

    def get_r_total(self,mesh,model,X,SU,flag):
        x,y,z = X
        R = mesh.stack_X(x,y,z)
        phi = self.PBE.get_phi(R,flag,model,value=self.field)
        gx,gy,gz = self.PBE.gradient(mesh,model,X,flag,value=self.field)
        r = self.epsilon*(gx**2+gy**2+gz**2)/2 - self.kappa*2 * phi**2
        return r
    
    def get_r_reg(self,mesh,model,X,SU,flag):
        x,y,z = X
        R = mesh.stack_X(x,y,z)
        phi = self.PBE.get_phi(R,flag,model,value=self.field)
        gx,gy,gz = self.PBE.gradient(mesh,model,X,flag,value=self.field)
        r = self.epsilon*(gx**2+gy**2+gz**2)/2 - self.kappa*2 * phi**2 - self.kappa*2*phi*self.PBE.G(X)
        return r

    
class Variational_Non_Linear(Equations_utils):

    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.epsilon = self.epsilon_2

        if self.field == 'phi':
            self.get_r = self.get_r_total
        elif self.field == 'react':
            self.get_r = self.get_r_reg

    def get_r_total(self,mesh,model,X,SU,flag):
        x,y,z = X
        R = mesh.stack_X(x,y,z)
        phi = self.PBE.get_phi(R,flag,model,value=self.field)
        gx,gy,gz = self.PBE.gradient(mesh,model,X,flag,value=self.field)
        r = self.epsilon*(gx**2+gy**2+gz**2)/2 - self.kappa**2*self.T_adim*self.PBE.aprox_sinh(phi/self.T_adim) *phi
        return r
    
    def get_r_reg(self,mesh,model,X,SU,flag):
        x,y,z = X
        R = mesh.stack_X(x,y,z)
        phi = self.PBE.get_phi(R,flag,model,value=self.field)
        gx,gy,gz = self.PBE.gradient(mesh,model,X,flag,value=self.field)
        r = self.epsilon*(gx**2+gy**2+gz**2)/2 - self.kappa**2*self.T_adim*self.PBE.aprox_sinh(phi+self.PBE.G(X)/self.T_adim) *phi
        return r


class Boundary_Poisson(Equations_utils):

    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.areas = tf.transpose(tf.constant(self.PBE.mesh.mol_faces_areas, dtype=self.DTYPE))
        self.normals = tf.constant(self.PBE.mesh.mol_faces_normal, dtype=self.DTYPE)
        self.centroids = tf.constant(self.PBE.mesh.mol_faces_centroid, dtype=self.DTYPE)

    def get_r(self,mesh,model,X,SU,flag):
        x,y,z = X
        R = mesh.stack_X(x,y,z)
        X_c = self.centroids
        N_v = self.normals
        phi_i = tf.transpose(self.PBE.get_phi(X_c,flag,model,value=self.field))
        dphi_i = tf.transpose(self.PBE.get_dphi(X_c,N_v,flag,model,value='phi')[0])
        integrand = (self.PBE.G_L(R,X_c)*dphi_i - self.PBE.dG_L(R,X_c)*phi_i)*self.areas
        integral = tf.reduce_sum(integrand, axis=1, keepdims=True) 
        phi = self.PBE.get_phi(R,flag,model,value=self.field)
        r = 0.5*phi - self.PBE.G(R) - integral
        return r


class Boundary_Helmholtz(Equations_utils):

    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.areas = tf.transpose(tf.constant(self.PBE.mesh.mol_faces_areas, dtype=self.DTYPE))
        self.normals = tf.constant(self.PBE.mesh.mol_faces_normal, dtype=self.DTYPE)
        self.centroids = tf.constant(self.PBE.mesh.mol_faces_centroid, dtype=self.DTYPE)

    def get_r(self,mesh,model,X,SU,flag):
        x,y,z = X
        R = mesh.stack_X(x,y,z)
        X_c = self.centroids
        N_v = self.normals
        phi_i = tf.transpose(self.PBE.get_phi(X_c,flag,model,value=self.field))
        dphi_i = tf.transpose(self.PBE.get_dphi(X_c,N_v,flag,model,value='phi')[1])
        integrand = (- self.PBE.G_Y(R,X_c)*dphi_i + self.PBE.dG_Y(R,X_c)*phi_i)*self.areas
        integral =  tf.reduce_sum(integrand, axis=1, keepdims=True) 
        phi = self.PBE.get_phi(R,flag,model,value=self.field)
        r = 0.5*phi - integral
        return r
        
        