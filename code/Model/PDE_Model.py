import os
import tensorflow as tf
import numpy as np
import bempp.api

from Model.PDE_utils import PDE_utils
from Mesh.Charges_utils import get_charges_list


class PBE(PDE_utils):

    qe = 1.60217663e-19
    eps0 = 8.8541878128e-12     
    kb = 1.380649e-23              
    Na = 6.02214076e23
    ang_to_m = 1e-10
    to_V = qe/(eps0 * ang_to_m)   

    def __init__(self, inputs, mesh, model, path):      

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
        self.get_integral_operators()

        super().__init__()

    @property
    def get_PDEs(self):
        PDEs = [self.PDE_in,self.PDE_out]
        return PDEs

    def get_charges(self):
        path_files = os.path.join(self.main_path,'Molecules')
        self.q_list = get_charges_list(os.path.join(path_files,self.molecule,self.molecule+'.pqr'))
        n = len(self.q_list)
        self.qs = np.zeros(n)
        self.x_qs = np.zeros((n,3))
        for i,q in enumerate(self.q_list):
            self.qs[i] = q.q
            self.x_qs[i,:] = q.x_q
        self.total_charge = np.sum(self.qs)

    def get_integral_operators(self):

        elements = self.mesh.mol_faces
        vertices = self.mesh.mol_verts
        self.grid = bempp.api.Grid(vertices.transpose(), elements.transpose())

        self.space = bempp.api.function_space(self.grid, "P", 1)
        self.dirichl_space = self.space
        self.neumann_space = self.space

        self.slp_q = bempp.api.operators.potential.laplace.single_layer(self.neumann_space, self.x_qs.transpose())
        self.dlp_q = bempp.api.operators.potential.laplace.double_layer(self.dirichl_space, self.x_qs.transpose())

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


    def get_loss_I(self,solver,solver_ex,XI_data,components=[True,True]):
        
        loss = 0
        XI,N_v = XI_data
        X = solver.mesh.get_X(XI)

        if components[0]:
            u_1 = solver.model(XI)
            u_2 = solver_ex.model(XI)
            u_prom = (u_1+u_2)/2
            loss += tf.reduce_mean(tf.square(u_1 - u_prom)) 

        if components[1]:
            n_v = solver.mesh.get_X(N_v)
            du_1 = self.directional_gradient(solver.mesh,solver.model,X,n_v)
            du_2 = self.directional_gradient(solver_ex.mesh,solver_ex.model,X,n_v)
            du_prom = (du_1*solver.PDE.epsilon + du_2*solver_ex.PDE.epsilon)/2
            loss += tf.reduce_mean(tf.square(du_1*solver.PDE.epsilon - du_prom))
            
        return loss
    
    def get_phi_ens(self,solvers,X_mesh,X_q):
        
        kT = self.kb*self.T
        C = self.qe/kT                

        s1,s2 = solvers
        X_in,X_out = X_mesh
        phi_ens_L = list()

        for x_q in X_q:

            if X_in != None:
                C_phi1 = s1.model(X_in) * self.to_V * C 
                r1 = tf.sqrt(tf.reduce_sum(tf.square(x_q - X_in), axis=1, keepdims=True))
                G2_p_1 =  tf.math.reduce_sum(self.aprox_exp(-C_phi1)/r1**6)
                G2_m_1 = tf.math.reduce_sum(self.aprox_exp(C_phi1)/r1**6)
            else:
                G2_p_1 = 0.0
                G2_m_1 = 0.0

            C_phi2 = s2.model(X_out) * self.to_V * C
            r2 = tf.math.sqrt(tf.reduce_sum(tf.square(x_q - X_out), axis=1, keepdims=True))
            G2_p = G2_p_1 + tf.math.reduce_sum(self.aprox_exp(-C_phi2)/r2**6)
            G2_m = G2_m_1 + tf.math.reduce_sum(self.aprox_exp(C_phi2)/r2**6)

            phi_ens_pred = -kT/(2*self.qe) * tf.math.log(G2_p/G2_m) * 1000  # to_mV
            phi_ens_L.append(phi_ens_pred)

        return phi_ens_L    

    def get_loss_experimental(self,solvers,X_exp):             

        loss = tf.constant(0.0, dtype=self.DTYPE)
        n = len(X_exp)
        X,X_values = X_exp
        x_q_L,phi_ens_exp_L = zip(*X_values)
        phi_ens_pred_L = self.get_phi_ens(solvers,X,x_q_L)

        for phi_pred,phi_exp in zip(phi_ens_pred_L,phi_ens_exp_L):
            loss += tf.square(phi_pred - phi_exp)

        loss *= (1/n)

        return loss
    

    def get_loss_Gauss(self,solvers,XI_data):
        loss = 0
        s1,s2 = solvers
        XI,N_v = XI_data
        X = s1.mesh.get_X(XI)
        n_v = s1.mesh.get_X(N_v)
        du_1 = self.directional_gradient(s1.mesh,s1.model,X,n_v)
        du_2 = self.directional_gradient(s2.mesh,s2.model,X,n_v)
        du_prom = (du_1*s1.PDE.epsilon + du_2*s2.PDE.epsilon)/2

        faces = self.mesh.mol_faces
        areas = self.mesh.areas
        du_faces = tf.reduce_mean(tf.gather(du_prom, faces), axis=1)

        integral = tf.reduce_sum(du_faces * areas)
        loss += tf.reduce_mean(tf.square(integral - self.total_charge))

        return loss


    def get_phi_interface(self,solver,solver_ex):      
        verts = tf.constant(self.mesh.mol_verts)
        u1 = solver.model(verts)
        u2 = solver_ex.model(verts)
        u_mean = (u1+u2)/2
        return u_mean.numpy()
    
    def get_dphi_interface(self,solver,solver_ex): 
        verts = tf.constant(self.mesh.mol_verts)     
        X = solver.mesh.get_X(verts)
        n_v = solver.mesh.get_X(self.mesh.mol_normal)
        du_1 = self.directional_gradient(solver.mesh,solver.model,X,n_v)
        du_2 = self.directional_gradient(solver_ex.mesh,solver_ex.model,X,n_v)
        du_prom = (du_1*solver.PDE.epsilon + du_2*solver_ex.PDE.epsilon)/2
        return du_prom.numpy(),du_1.numpy(),du_2.numpy()
    
    def get_solvation_energy(self,solver,solver_ex):

        u_interface = self.get_phi_interface(solver,solver_ex).flatten()
        _,du_1,du_2 = self.get_dphi_interface(solver,solver_ex)
        du_1 = du_1.flatten()
        du_2 = du_2.flatten()
        du_1_interface = (du_1+du_2*solver_ex.PDE.epsilon/solver.PDE.epsilon)/2

        phi = bempp.api.GridFunction(self.space, coefficients=u_interface)
        dphi = bempp.api.GridFunction(self.space, coefficients=du_1_interface)

        phi_q = self.slp_q * dphi - self.dlp_q * phi
        
        G_solv = 0.5*np.sum(self.qs * phi_q).real
        G_solv *= self.to_V*self.qe*self.Na*(10**-3/4.184)   # kcal/mol
        
        return G_solv

    def analytic_Born_Ion(self,r):
        rI = self.mesh.R_mol
        epsilon_1 = self.epsilon_1
        epsilon_2 = self.epsilon_2
        kappa = self.kappa
        q = self.q_list[0].q

        f_IN = lambda r: (q/(4*self.pi)) * ( 1/(epsilon_1*r) - 1/(epsilon_1*rI) + 1/(epsilon_2*(1+kappa*rI)*rI) )
        f_OUT = lambda r: (q/(4*self.pi)) * (np.exp(-kappa*(r-rI))/(epsilon_2*(1+kappa*rI)*r))

        y = np.piecewise(r, [r<=rI, r>rI], [f_IN, f_OUT])

        return y


class Poisson(PDE_utils):

    def __init__(self, inputs):

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

        for key, value in inputs.items():
            setattr(self, key, value)
        self.epsilon = self.epsilon_2
        super().__init__()

    def residual_loss(self,mesh,model,X,SU):
        x,y,z = X
        R = mesh.stack_X(x,y,z)
        u = model(R)
        r = self.laplacian(mesh,model,X) - self.kappa**2*tf.math.sinh(u)     
        Loss_r = tf.reduce_mean(tf.square(r))
        return Loss_r
    