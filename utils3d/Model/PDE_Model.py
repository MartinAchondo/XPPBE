import os
import tensorflow as tf
import numpy as np

from Model.PDE_utils import PDE_utils
from Model.Molecules.Charges import get_charges_list


class PBE(PDE_utils):

    def __init__(self, inputs, mesh, model):
        
        self.sigma = 0.04

        self.mesh = mesh

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
        path_files = os.path.join(os.getcwd(),'utils3d','Model','Molecules')
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
    

    def get_loss_experimental(self,s1,s2,X_exp):
        # [(X1,X2,phi_ens),...]
        # X contains Xi and ri (a point and the radius to the charge)

        qe = 1.60217663e-19
        eps0 = 8.8541878128e-12
        ang_to_m = 1e-10
        to_V = qe/(eps0 * ang_to_m)
        kT = 4.11e-21 
        Na = 6.02214076e23
        C = qe/kT

        loss = 0.0
        n = len(X_exp)

        for X_in,X_out,phi_ens_exp in X_exp:

            if X_in != None:
                X1,r1 = X_in
                phi1 = s1.model(X1) * to_V
                G2_p_1 =  tf.math.reduce_sum(tf.math.exp(-C*phi1)/r1**6)
                G2_m_1 = tf.math.reduce_sum(tf.math.exp(C*phi1)/r1**6)
            else:
                G2_p_1 = 0.0
                G2_m_1 = 0.0

            X2,r2 = X_out

            phi2 = s2.model(X2) * to_V

            G2_p = G2_p_1 + tf.math.reduce_sum(tf.math.exp(-C*phi2)/r2**6)
            G2_m = G2_m_1 + tf.math.reduce_sum(tf.math.exp(C*phi2)/r2**6)

            phi_ens_pred = -kT/(2*qe) * tf.math.log(G2_p/G2_m) * 1000

            loss += tf.square(phi_ens_pred - phi_ens_exp)

        loss *= (1/n)

        return loss


    def analytic(self,r):
        rI = 1
        epsilon_1 = 1
        epsilon_2 = 80
        kappa = 0.125
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
        r = self.laplacian(mesh,model,X) - self.kappa**2*tf.math.sinh(u)      
        Loss_r = tf.reduce_mean(tf.square(r))
        return Loss_r
    