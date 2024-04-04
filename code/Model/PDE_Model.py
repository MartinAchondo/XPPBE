import os
import numpy as np
import bempp.api
import tensorflow as tf

from Mesh.Charges_utils import get_charges_list
from Model.Solutions_utils import Solution_utils

class PBE(Solution_utils):

    qe = 1.60217663e-19
    eps0 = 8.8541878128e-12     
    kb = 1.380649e-23              
    Na = 6.02214076e23
    ang_to_m = 1e-10
    to_V = qe/(eps0 * ang_to_m)  
    cal2j = 4.184

    DTYPE = 'float32'
    pi = tf.constant(np.pi, dtype=DTYPE)

    def __init__(self, inputs, mesh, model, path):      

        self.mesh = mesh
        self.main_path = path
        self.eq = model

        self.sigma = self.mesh.G_sigma
        self.inputs = inputs
        for key, value in inputs.items():
            setattr(self, key, value)

        self.get_charges()
        self.get_integral_operators()

        super().__init__()

    @property
    def get_PDEs(self):
        PDEs = [self.PDE_in,self.PDE_out]
        return PDEs
            
    def get_phi_interface(self,model,**kwargs):      
        verts = tf.constant(self.mesh.mol_verts, dtype=self.DTYPE)
        u = self.get_phi(verts,'interface',model,**kwargs)
        u_mean = (u[:,0]+u[:,1])/2
        return u_mean.numpy(),u[:,0].numpy(),u[:,1].numpy()
    
    def get_dphi_interface(self,model, value='phi'): 
        verts = tf.constant(self.mesh.mol_verts, dtype=self.DTYPE)     
        N_v = self.mesh.mol_normal
        du_1,du_2 = self.get_dphi(verts,N_v,'',model,value)
        du_prom = (du_1*self.PDE_in.epsilon + du_2*self.PDE_out.epsilon)/2
        return du_prom.numpy(),du_1.numpy(),du_2.numpy()
    
    
    def get_phi_ens(self,model,X_mesh,X_q):
        
        kT = self.kb*self.T
        C = self.qe/kT                

        (X_out,flag) = X_mesh
        phi_ens_L = list()

        for x_q in X_q:

            C_phi2 = self.get_phi(X_out,flag, model) * self.to_V * C
            r2 = tf.math.sqrt(tf.reduce_sum(tf.square(x_q - X_out), axis=1, keepdims=True))
            G2_p = tf.math.reduce_sum(self.aprox_exp(-C_phi2)/r2**6)
            G2_m = tf.math.reduce_sum(self.aprox_exp(C_phi2)/r2**6)

            phi_ens_pred = -kT/(2*self.qe) * tf.math.log(G2_p/G2_m) * 1000  # to_mV
            phi_ens_L.append(phi_ens_pred)

        return phi_ens_L    
    

    # Losses

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
            loss_k = self.dirichlet_loss(self.mesh,model,X,U,flag)
            L['K1'] += loss_k   

        if 'K2' in X_batches and not validation:
            ((X,U),flag) = X_batches['K2']
            loss_k = self.dirichlet_loss(self.mesh,model,X,U,flag)
            L['K2'] += loss_k 

        if 'I' in X_batches:
            if 'Iu' in self.mesh.domain_mesh_names:
                L['Iu'] += self.get_loss_I(model,X_batches['I'], 'Iu')
            if 'Id' in self.mesh.domain_mesh_names:
                L['Id'] += self.get_loss_I(model,X_batches['I'], 'Id')
            if 'Ir' in self.mesh.domain_mesh_names:
                L['Ir'] += self.get_loss_I(model,X_batches['I'], 'Ir')    

        if 'E2' in X_batches and not validation:
            L['E2'] += self.get_loss_experimental(model,X_batches['E2'])

        if 'G' in X_batches and not validation:
            L['G'] += self.get_loss_Gauss(model,X_batches['G'])

        return L


    def dirichlet_loss(self,mesh,model,XD,UD,flag):
        Loss_d = 0
        u_pred = self.get_phi(XD,flag,model)
        loss = tf.reduce_mean(tf.square(UD - u_pred)) 
        Loss_d += loss
        return Loss_d

    def get_loss_I(self,model,XI_data,loss_type='Iu'):
        
        loss = 0
        ((XI,N_v),flag) = XI_data
        X = self.mesh.get_X(XI)

        if loss_type=='Iu':
            u = self.get_phi(XI,flag,model)
            loss += tf.reduce_mean(tf.square(u[:,0]-u[:,1])) 

        elif loss_type=='Id':
            du_1,du_2 = self.get_dphi(XI,N_v,flag,model,value='phi')
            loss += tf.reduce_mean(tf.square(du_1*self.PDE_in.epsilon - du_2*self.PDE_out.epsilon))
        
        elif loss_type=='Ir':
            r1 = self.PDE_in.get_r(self.mesh,model,X,None,'molecule')
            r2 = self.PDE_out.get_r(self.mesh,model,X,None,'solvent')
            loss += tf.reduce_mean(tf.square(r1-r2)) 
            
        return loss

    def get_loss_experimental(self,model,X_exp):             

        loss = tf.constant(0.0, dtype=self.DTYPE)
        n = len(X_exp)
        ((X,X_values),flag) = X_exp
        x_q_L,phi_ens_exp_L = zip(*X_values)
        phi_ens_pred_L = self.get_phi_ens(model,(X,flag),x_q_L)

        for phi_pred,phi_exp in zip(phi_ens_pred_L,phi_ens_exp_L):
            loss += tf.square(phi_pred - phi_exp)

        loss *= (1/n)

        return loss
    
    def get_loss_Gauss(self,model,XI_data):
        loss = 0
        ((XI,N_v,areas),flag) = XI_data
        du_1,du_2 = self.get_dphi(XI,N_v,flag,model,value='phi')
        du_prom = (du_1*self.PDE_in.epsilon + du_2*self.PDE_out.epsilon)/2

        integral = tf.reduce_sum(du_prom * areas)
        loss += tf.reduce_mean(tf.square(integral - self.total_charge))

        return loss
    
    def get_loss_preconditioner(self, X_batches, model):
        L = self.create_L()

        #residual
        if 'P1' in X_batches:
            ((X,U),flag) = X_batches['P1']
            loss_p = self.dirichlet_loss(self.mesh,model,X,U,flag)
            L['P1'] += loss_p  

        if 'P2' in X_batches:
            ((X,U),flag) = X_batches['P2']
            loss_p = self.dirichlet_loss(self.mesh,model,X,U,flag)
            L['P2'] += loss_p  
            
        return L

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

    def aprox_exp(self,x):
        aprox = 1.0 + x + x**2/2.0 + x**3/6.0 + x**4/24.0
        return aprox

    # Differential operators

    def laplacian(self,mesh,model,X,flag,value='phi'):
        x,y,z = X
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(y)
            tape.watch(z)
            R = mesh.stack_X(x,y,z)
            u = self.get_phi(R,flag,model,value)
            u_x = tape.gradient(u,x)
            u_y = tape.gradient(u,y)
            u_z = tape.gradient(u,z)
        u_xx = tape.gradient(u_x,x)
        u_yy = tape.gradient(u_y,y)
        u_zz = tape.gradient(u_z,z)
        del tape
        return u_xx + u_yy + u_zz

    def gradient(self,mesh,model,X,flag,value='phi'):
        x,y,z = X
        with tf.GradientTape(persistent=True,watch_accessed_variables=False) as tape:
            tape.watch(x)
            tape.watch(y)
            tape.watch(z)
            R = mesh.stack_X(x,y,z)
            u = self.get_phi(R,flag,model,value)
        u_x = tape.gradient(u,x)
        u_y = tape.gradient(u,y)
        u_z = tape.gradient(u,z)
        del tape
        return (u_x,u_y,u_z)
    
    def directional_gradient(self,mesh,model,X,n_v,flag,value='phi'):
        gradient = self.gradient(mesh,model,X,flag,value)
        dir_deriv = 0
        for j in range(3):
            dir_deriv += n_v[j]*gradient[j]
        return dir_deriv
    
    # utils

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
    
    @classmethod
    def create_L(cls):
        cls.names = ['R1','D1','N1','K1','Q1','R2','D2','N2','K2','G','Iu','Id','Ir','E2','P1','P2']
        L = dict()
        for t in cls.names:
            L[t] = tf.constant(0.0, dtype=cls.DTYPE)
        return L



