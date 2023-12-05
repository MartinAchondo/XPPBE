import os
import tensorflow as tf
import numpy as np
import bempp.api

from Model.PDE_utils import PDE_utils
from Model.Molecules.Charges import get_charges_list


class PBE(PDE_utils):

    qe = 1.60217663e-19
    eps0 = 8.8541878128e-12     
    kb = 1.380649e-23              
    Na = 6.02214076e23

    def __init__(self, inputs, mesh, model,path):      

        self.mesh = mesh
        self.main_path = path

        self.sigma = self.mesh.G_sigma

        self.inputs = inputs
        for key, value in inputs.items():
            setattr(self, key, value)
        
        self.PDE_in = Poisson(inputs)
        self.PDE_in.mesh = self.mesh.interior_obj

        if model=='linear':
            self.PDE_out = Helmholtz(inputs)
        elif model=='nonlinear':
            self.PDE_out = Non_Linear(inputs)
        self.PDE_out.mesh = self.mesh.exterior_obj

        self.get_charges()

        super().__init__()

    @property
    def get_PDEs(self):
        PDEs = [self.PDE_in,self.PDE_out]
        return PDEs


    def get_charges(self):
        path_files = os.path.join(self.main_path,'Model','Molecules')
        self.q_list = get_charges_list(os.path.join(path_files,self.molecule,self.molecule+'.pqr'))
        for charge in self.q_list:
            for i in range(3):
                charge.x_q[i] -= self.mesh.centroid[i]

    def source(self,x,y,z):
        sum = 0
        for q_obj in self.q_list:
            qk = q_obj.q
            xk,yk,zk = q_obj.x_q
            
            deltak = tf.exp((-1/(2*self.sigma**2))*((x-xk)**2+(y-yk)**2+(z-zk)**2))
            sum += qk*deltak

        normalizer = (1/((2*self.pi)**(3.0/2)*self.sigma**3))
        sum *= normalizer
        return (-1/self.epsilon_1)*sum


    def border_value(self,x,y,z):
        sum = 0
        for q_obj in self.q_list:
            qk = q_obj.q
            xk,yk,zk = q_obj.x_q
            r = tf.sqrt((x-xk)**2+(y-yk)**2+(z-zk)**2)
            sum += qk*tf.exp(-self.kappa*r)/r
        return (1/(4*self.pi*self.epsilon_2))*sum


    def get_loss_I(self,solver,solver_ex,XI_data):
        loss = 0
        
        XI,N_v = XI_data
        X = solver.mesh.get_X(XI)

        u_1 = solver.model(XI)
        u_2 = solver_ex.model(XI)

        n_v = solver.mesh.get_X(N_v)
        du_1 = self.directional_gradient(solver.mesh,solver.model,X,n_v)
        du_2 = self.directional_gradient(solver_ex.mesh,solver_ex.model,X,n_v)

        u_prom = (u_1+u_2)/2
        du_prom = (du_1*solver.PDE.epsilon + du_2*solver_ex.PDE.epsilon)/2
        
        loss += tf.reduce_mean(tf.square(u_1 - u_prom)) 
        loss += tf.reduce_mean(tf.square(du_1*solver.PDE.epsilon - du_prom))
            
        return loss
    

    def get_loss_experimental(self,solvers,X_exp):

        ang_to_m = 1e-10
        to_V = self.qe/(self.eps0 * ang_to_m)   
        kT = self.kb*self.T
        C = self.qe/kT                

        loss = tf.constant(0.0, dtype=self.DTYPE)
        n = len(X_exp)

        s1,s2 = solvers

        for X_in,X_out,x_q,phi_ens_exp in X_exp:

            if X_in != None:
                C_phi1 = s1.model(X_in) * to_V * C 
                r1 = tf.sqrt(tf.reduce_sum(tf.square(x_q - X_in), axis=1, keepdims=True))
                G2_p_1 =  tf.math.reduce_sum(self.aprox_exp(-C_phi1)/r1**6)
                G2_m_1 = tf.math.reduce_sum(self.aprox_exp(C_phi1)/r1**6)
            else:
                G2_p_1 = 0.0
                G2_m_1 = 0.0

            C_phi2 = s2.model(X_out) * to_V * C

            r2 = tf.math.sqrt(tf.reduce_sum(tf.square(x_q - X_out), axis=1, keepdims=True))

            # G2_p = G2_p_1 + tf.math.reduce_sum(self.aprox_exp(-C_phi2)/r2**6)
            # G2_m = G2_m_1 + tf.math.reduce_sum(self.aprox_exp(C_phi2)/r2**6)

            G2_p = G2_p_1 + tf.math.reduce_sum(self.aprox_exp(-C_phi2-6*tf.math.log(r2)))
            G2_m = G2_m_1 + tf.math.reduce_sum(self.aprox_exp(C_phi2-6*tf.math.log(r2)))

            phi_ens_pred = -kT/(2*self.qe) * tf.math.log(G2_p/G2_m) * 1000 

            loss += tf.square(phi_ens_pred - phi_ens_exp)

        loss *= (1/n)

        return loss

    def get_phi_interface(self,solver,solver_ex):      
        verts = tf.constant(self.mesh.verts)
        u1 = solver.model(verts)
        u2 = solver_ex.model(verts)
        u_mean = (u1+u2)/2
        return u_mean.numpy()
    
    def get_dphi_interface(self,solver,solver_ex): 
        verts = tf.constant(self.mesh.verts)     
        X = solver.mesh.get_X(verts)
        n_v = solver.mesh.get_X(self.mesh.normal)
        du_1 = self.directional_gradient(solver.mesh,solver.model,X,n_v)
        du_2 = self.directional_gradient(solver_ex.mesh,solver_ex.model,X,n_v)
        du_prom = (du_1*solver.PDE.epsilon + du_2*solver_ex.PDE.epsilon)/2
        return du_prom.numpy()
    
    def get_solvation_energy(self,solver,solver_ex):

        n = len(self.q_list)
        qs = np.zeros(n)
        x_qs = np.zeros((n,3))
        for i,q in enumerate(self.q_list):
            qs[i] = q.q
            x_qs[i,:] = q.x_q

        elements = self.mesh.faces
        vertices = self.mesh.verts
        grid = bempp.api.Grid(vertices.transpose(), elements.transpose())

        space = bempp.api.function_space(grid, "P", 1)
        dirichl_space = space
        neumann_space = space

        slp_q = bempp.api.operators.potential.laplace.single_layer(neumann_space, x_qs.transpose())
        dlp_q = bempp.api.operators.potential.laplace.double_layer(dirichl_space, x_qs.transpose())

        u_values = self.get_phi_interface(solver,solver_ex).flatten()
        du_values = self.get_dphi_interface(solver,solver_ex).flatten()
        
        phi = bempp.api.GridFunction(space, coefficients=u_values)
        dphi = bempp.api.GridFunction(space, coefficients=du_values)

        phi_q = slp_q * dphi - dlp_q * phi
        G_solv = 2 * np.pi * 332.064 * np.sum(qs * phi_q).real  # kcal/kmol

        return G_solv   

    def analytic_Born_Ion(self,r):
        rI = 1
        epsilon_1 = self.epsilon_1
        epsilon_2 = self.epsilon_2
        kappa = self.kappa
        q = 1.0

        f_IN = lambda r: (q/(4*self.pi)) * ( 1/(epsilon_1*r) - 1/(epsilon_1*rI) + 1/(epsilon_2*(1+kappa*rI)*rI) )
        f_OUT = lambda r: (q/(4*self.pi)) * (np.exp(-kappa*(r-rI))/(epsilon_2*(1+kappa*rI)*r))

        y = np.piecewise(r, [r<=rI, r>rI], [f_IN, f_OUT])

        return y


class Poisson(PDE_utils):

    def __init__(self, inputs):

        self.sigma = 0.04

        for key, value in inputs.items():
            setattr(self, key, value)

        self.epsilon = self.epsilon_1

        super().__init__()

    def residual_loss(self,mesh,model,X,SU):
        x,y,z = X
        r = self.laplacian(mesh,model,X) - SU       
        Loss_r = tf.reduce_mean(tf.square(r))
        return Loss_r




class Helmholtz(PDE_utils):

    def __init__(self, inputs):

        self.sigma = 0.04

        for key, value in inputs.items():
            setattr(self, key, value)

        self.epsilon = self.epsilon_2

        super().__init__()


    def residual_loss(self,mesh,model,X,SU):
        x,y,z = X
        R = mesh.stack_X(x,y,z)
        u = model(R)
        r = self.laplacian(mesh,model,X) - self.kappa**2*u      
        Loss_r = tf.reduce_mean(tf.square(r))
        return Loss_r  



class Non_Linear(PDE_utils):

    def __init__(self, inputs):

        self.sigma = 0.04

        for key, value in inputs.items():
            setattr(self, key, value)

        self.epsilon = self.epsilon_2

        super().__init__()


    def residual_loss(self,mesh,model,X,SU):
        x,y,z = X
        R = mesh.stack_X(x,y,z)
        u = model(R)
        r = self.laplacian(mesh,model,X) - self.kappa**2*tf.math.sinh(u)    # revisar unidades  
        Loss_r = tf.reduce_mean(tf.square(r))
        return Loss_r
    