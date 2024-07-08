import os
import numpy as np
import tensorflow as tf

from xppbe.Mesh.Charges_utils import get_charges_list
from .Solutions_utils import Solution_utils

class PBE(Solution_utils):

    DTYPE = 'float32'

    qe = tf.constant(1.60217663e-19, dtype=DTYPE)
    eps0 = tf.constant(8.8541878128e-12, dtype=DTYPE)     
    kb = tf.constant(1.380649e-23, dtype=DTYPE)              
    Na = tf.constant(6.02214076e23, dtype=DTYPE)
    ang_to_m = tf.constant(1e-10, dtype=DTYPE)
    cal2j = tf.constant(4.184, dtype=DTYPE)
    # to_V = qe/(eps0 * ang_to_m)  
    # T_fact = kb/(to_V*qe)

    pi = tf.constant(np.pi, dtype=DTYPE)

    def __init__(self, domain_properties, mesh, equation, pinns_method, adim, main_path, molecule_dir, results_path):      

        self.mesh = mesh
        self.main_path = main_path
        self.equation = equation
        self.pinns_method = pinns_method
        self.adim = adim
        self.molecule_path = molecule_dir
        self.results_path = results_path

        self.calculate_properties(domain_properties)

        self.get_charges()
        if self.scheme == 'direct':
            self.bempp = None
            self.get_integral_operators()

        super().__init__()

    @property
    def get_PDEs(self):
        PDEs = [self.PDE_in,self.PDE_out]
        return PDEs
    
    def calculate_properties(self,domain_properties):

        self.domain_properties = {
                'molecule': 'born_ion',
                'epsilon_1':  1,
                'epsilon_2': 80,
                'kappa': 0.125,
                'T' : 300 
                }
        
        T = domain_properties['T'] if 'T' in domain_properties else self.domain_properties['T']
        kappa = domain_properties['kappa'] if 'kappa' in domain_properties else self.domain_properties['kappa']
        epsilon_2 = domain_properties['epsilon_2'] if 'epsilon_2' in domain_properties else self.domain_properties['epsilon_2']

        domain_properties['concentration'] = (kappa/self.ang_to_m)**2*(self.eps0*epsilon_2*self.kb*T)/(2*self.qe**2*self.Na)/1000

        if self.adim == 'qe_eps0_angs':
            self.to_V = self.qe/(self.eps0 * self.ang_to_m)  
            domain_properties['beta'] = T*self.kb*self.eps0*self.ang_to_m/self.qe**2
            domain_properties['gamma'] = 1
        elif self.adim == 'kbT_qe':
            self.to_V = self.kb*self.T/self.qe
            domain_properties['beta'] = 1
            domain_properties['gamma'] = T*self.kb*self.eps0*self.ang_to_m/self.qe**2
        
        for key in ['molecule','epsilon_1','epsilon_2','kappa','T','concentration','beta','gamma']:
            if key in domain_properties:
                self.domain_properties[key] = domain_properties[key]
            if key != 'molecule':
                setattr(self, key, tf.constant(self.domain_properties[key], dtype=self.DTYPE))
            else:
                setattr(self, key, self.domain_properties[key])



        self.sigma = self.mesh.G_sigma
            
    def get_phi_interface(self,X,model,**kwargs):      
        u_mean = self.get_phi(X,'interface',model,**kwargs)
        u_1 = self.get_phi(X,'molecule',model,**kwargs)
        u_2 = self.get_phi(X,'solvent',model,**kwargs)
        return u_mean[:,0],u_1[:,0],u_2[:,0]
    
    def get_dphi_interface(self,X,N_v,model,value='phi'): 
        du_1,du_2 = self.get_dphi(X,N_v,'',model,value)
        du_prom = (du_1*self.PDE_in.epsilon + du_2*self.PDE_out.epsilon)/2
        return du_prom,du_1,du_2

    def get_phi_interface_verts(self,model,**kwargs):      
        verts = tf.constant(self.mesh.mol_verts, dtype=self.DTYPE)
        return self.get_phi_interface(verts,model)
    
    def get_dphi_interface_verts(self,model,value='phi'): 
        verts = tf.constant(self.mesh.mol_verts, dtype=self.DTYPE)     
        N_v = self.mesh.mol_verts_normal
        return self.get_dphi_interface(verts,N_v,model)
    
    
    def get_phi_ens(self,model,X_mesh,q_L, method='mean', pinn=True, known_method=False):        

        (X_solv,flag) = X_mesh
        phi_ens_L = list()

        for x_q,r_q in q_L:
            
            if method=='exponential':
                if pinn: 
                    phi = self.get_phi(X_solv,flag, model)
                else: 
                    phi = self.phi_known(known_method,'phi',tf.constant(X_solv),'solvent').reshape(-1,1)
                r_H = tf.math.sqrt(tf.reduce_sum(tf.square(x_q - X_solv), axis=1, keepdims=True))
                G2_p = tf.math.reduce_sum(self.aprox_exp(-phi/self.T_adim)/r_H**6)
                G2_m = tf.math.reduce_sum(self.aprox_exp(phi/self.T_adim)/r_H**6)
                phi_ens_pred = - self.T_adim/2 * tf.math.log(G2_p/G2_m)
            
            elif method=='mean':
                r_H = tf.math.sqrt(tf.reduce_sum(tf.square(x_q - X_solv), axis=1))
                X_ens = tf.boolean_mask(X_solv, r_H < (r_q + self.mesh.dR_exterior))
                if pinn: 
                    phi = self.get_phi(X_ens,flag, model)
                else: 
                    phi = self.phi_known(known_method,'phi',tf.constant(X_ens),'solvent').reshape(-1,1)
                phi_ens_pred = tf.reduce_mean(phi)

            phi_ens_L.append(phi_ens_pred)

        return phi_ens_L    
    
    def solvation_energy_phi_qs(self,phi_q):
        G_solv = 0.5*tf.reduce_sum(self.qs * phi_q)
        G_solv *= self.to_V*self.qe*self.Na*(10**-3/self.cal2j)   
        return G_solv

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

            if 'IB1' in self.mesh.domain_mesh_names: 
                ((X,N),flag) = X_batches['I']
                loss_r = self.PDE_in.residual_loss(self.mesh,model,self.mesh.get_X(X),N,flag)
                L['IB1'] += loss_r   

            if 'IB2' in self.mesh.domain_mesh_names: 
                ((X,N),flag) = X_batches['I']
                loss_r = self.PDE_out.residual_loss(self.mesh,model,self.mesh.get_X(X),N,flag)
                L['IB2'] += loss_r   

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
            u1 = self.get_phi(XI,'molecule',model)
            u2 = self.get_phi(XI,'solvent',model)
            loss += tf.reduce_mean(tf.square(u1-u2)) 

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
        ((X,X_values),flag,method) = X_exp
        q_L,phi_ens_exp_L = zip(*X_values)
        phi_ens_pred_L = self.get_phi_ens(model,(X,flag),q_L,method)

        for phi_pred,phi_exp in zip(phi_ens_pred_L,phi_ens_exp_L):
            loss += tf.square(phi_pred - phi_exp/self.to_V)

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


    @staticmethod
    def aprox_exp(x):
        aprox = 1.0 + x + x**2/2.0 + x**3/6.0 + x**4/24.0
        return aprox
    
    @staticmethod
    def aprox_sinh(x):
        aprox = x + x**3/6.0 
        return aprox

    # Differential operators

    def laplacian(self,mesh,model,X,flag,value='phi'):
        x,y,z = X
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
            tape.watch(x)
            tape.watch(y)
            tape.watch(z)
            with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape2:
                tape2.watch(x)
                tape2.watch(y)
                tape2.watch(z)
                R = mesh.stack_X(x,y,z)
                u = self.get_phi(R,flag,model,value)
            u_x = tape2.gradient(u,x)
            u_y = tape2.gradient(u,y)
            u_z = tape2.gradient(u,z)
        u_xx = tape.gradient(u_x,x)
        u_yy = tape.gradient(u_y,y)
        u_zz = tape.gradient(u_z,z)
        del tape
        del tape2
        return u_xx + u_yy + u_zz

    def gradient(self,mesh,model,X,flag,value='phi'):
        x,y,z = X
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
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
        self.pqr_path = os.path.join(self.molecule_path,self.molecule+'.pqr')
        self.q_list = get_charges_list(self.pqr_path)
        #self.scale_q_factor = min(self.q_list, key=lambda q_obj: np.abs(q_obj.q)).q

        n = len(self.q_list)
        self.qs = np.zeros(n)
        self.x_qs = np.zeros((n,3))
        for i,q in enumerate(self.q_list):
            self.qs[i] = q.q
            self.x_qs[i,:] = q.x_q        
        self.total_charge = tf.constant(np.sum(self.qs), dtype=self.DTYPE)
        self.qs = tf.constant(self.qs, dtype=self.DTYPE)
        self.x_qs = tf.constant(self.x_qs, dtype=self.DTYPE)

        scale_min_value_1, scale_max_value_1 = 0.,0.
        scale_min_value_2, scale_max_value_2 = 0.,0.
        for i,q in enumerate(self.q_list):
            phi = self.analytic_Born_Ion(0.0, R=q.r_q, index_q=i)
            phi_save = phi.copy()
            for j,q2 in enumerate(self.q_list):
                if i==j: 
                    continue
                rx = np.linalg.norm(q.x_q-q2.x_q)
                phi += self.analytic_Born_Ion(rx, R=rx, index_q=j)

            if self.fields[0] == 'phi':
                phi_1 = phi + self.G(self.x_qs[i,:] + tf.constant([[q.r_q,0,0]],dtype=self.DTYPE))
                phi_1_save = phi_save + self.G(self.x_qs[i,:] + tf.constant([[q.r_q,0,0]],dtype=self.DTYPE))
            else:
                phi_1 = phi 
                phi_1_save = phi_save
            
            phi_1_max = phi_1_save if phi_1_save>phi_1 else phi_1
            phi_1_min = phi_1_save if phi_1_save<phi_1 else phi_1
            if phi_1_max > scale_max_value_1:
                scale_max_value_1 = phi_1_max
            if phi_1_min < scale_min_value_1:
                scale_min_value_1 = phi_1_min

            if self.fields[1] == 'phi':
                phi_2 = phi + self.G(self.x_qs[i,:] + tf.constant([[q.r_q,0,0]],dtype=self.DTYPE))
                phi_2_save = phi_save + self.G(self.x_qs[i,:] + tf.constant([[q.r_q,0,0]],dtype=self.DTYPE))
            else:
                phi_2 = phi 
                phi_2_save = phi_save
           
            phi_2_max = phi_2_save if phi_2_save>phi_2 else phi_2
            phi_2_min = phi_2_save if phi_2_save<phi_2 else phi_2
            if phi_2_max > scale_max_value_2:
                scale_max_value_2 = phi_2_max
            if phi_2_min < scale_min_value_2:
                scale_min_value_2 = phi_2_min

        self.scale_phi_1 = [float(scale_min_value_1),float(scale_max_value_1)]
        self.scale_phi_2 = [float(scale_min_value_2),float(scale_max_value_2)]
    

    def get_integral_operators(self):
        if self.bempp == None:
            import bempp.api
            self.bempp = bempp.api
        elements = self.mesh.mol_faces
        vertices = self.mesh.mol_verts
        self.grid = self.bempp.Grid(vertices.transpose(), elements.transpose())
        self.space = self.bempp.function_space(self.grid, "DP", 0)
        self.dirichl_space = self.space
        self.neumann_space = self.space

        self.slp_q = bempp.api.operators.potential.laplace.single_layer(self.neumann_space, self.x_qs.numpy().transpose())
        self.dlp_q = bempp.api.operators.potential.laplace.double_layer(self.dirichl_space, self.x_qs.numpy().transpose())

        vertices = self.grid.vertices
        faces_normals = self.grid.normals
        elements = self.grid.elements
        centroids = np.zeros((3, elements.shape[1]))
        for i, element in enumerate(elements.T):
            centroids[:, i] = np.mean(vertices[:, element], axis=1)

        self.mesh.grid_centroids = tf.reshape(tf.constant(centroids.transpose(),dtype=self.DTYPE), (-1,3))
        self.mesh.grid_faces_normals = tf.reshape(tf.constant(faces_normals.transpose(),dtype=self.DTYPE), (-1,3))
    
    def get_grid_coefficients_faces(self,model):

        X = self.mesh.grid_centroids
        Nv = self.mesh.grid_faces_normals
        phi_mean,_,_ = self.get_phi_interface(X,model)
        u_interface = phi_mean.numpy().flatten()
        _,du_1,_ = self.get_dphi_interface(X,Nv,model)
        du_1_interface = du_1.numpy().flatten()

        phi = self.bempp.GridFunction(self.space, coefficients=u_interface)
        dphi = self.bempp.GridFunction(self.space, coefficients=du_1_interface)

        return phi,dphi


    @classmethod
    def create_L(cls):
        cls.names = ['R1','D1','N1','K1','Q1','R2','D2','N2','K2','G','Iu','Id','Ir','E2','P1','P2','IB1','IB2']
        L = dict()
        for t in cls.names:
            L[t] = tf.constant(0.0, dtype=cls.DTYPE)
        return L



