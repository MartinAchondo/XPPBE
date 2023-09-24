
import os
import numpy as np
import tensorflow as tf
import trimesh
import matplotlib.pyplot as plt
import plotly.express as px

from Model.Molecules.Charges import import_charges_from_pqr

class Mesh():

    DTYPE='float32'

    def __init__(self,name,molecule):

        self.name = name
        self.molecule = molecule

        self.X_R = None
        self.XU_D = None
        self.XU_N = None
        self.XU_K = None
        self.X_I = None

        self.solver_mesh_data = {
            'R': self.X_R,
            'D': self.XU_D,
            'N': self.XU_N,
            'K': self.XU_K, 
            'I': self.X_I,
            'P': None,
        }

        self.prior_data = dict()
        self.solver_mesh_names = set()
        self.solver_mesh_N = dict()

    def adapt_meshes(self,meshes):

        self.meshes = meshes

        for bl in self.meshes.values():
            type_b = bl['type']
            value = bl['value']
            fun = bl['fun']
            if 'dr' in bl:
                deriv = bl['dr']
            if not 'noise' in bl:
                bl['noise'] = False

            if type_b in self.prior_data:
                X = self.prior_data[type_b]

            if type_b in ('R','D','K','N','P'):    
                x,y,z = self.get_X(X)
                if value != None:
                    U = self.value_u_b(x, y, z, value=value)
                elif fun != None:
                    U = fun(x, y, z)
                else:
                    file = bl['file']
                    X,U = self.read_file_data(file)
                self.solver_mesh_data[type_b] = self.create_Datasets(X,U)

            self.solver_mesh_names.add(type_b)
            if type_b in self.solver_mesh_N:
                self.solver_mesh_N[type_b] += len(X)
            else:
                self.solver_mesh_N[type_b] = len(X)
            
        del self.prior_data

    def read_file_data(self,file):
        x_b, y_b, z_b, phi_b = list(), list(), list(), list()

        path_files = os.path.join(os.getcwd(),'utils3d','Model','Molecules')

        with open(os.path.join(path_files,self.molecule,file),'r') as f:
            for line in f:
                condition, x, y, z, phi = line.strip().split()

                if int(condition) == self.name:
                    x_b.append(float(x))
                    y_b.append(float(y))
                    z_b.append(float(z))
                    phi_b.append(float(phi))

        x_b = tf.constant(np.array(x_b, dtype=self.DTYPE))
        y_b = tf.constant(np.array(y_b, dtype=self.DTYPE))
        z_b = tf.constant(np.array(z_b, dtype=self.DTYPE))
        phi_b = tf.constant(np.array(phi_b, dtype=self.DTYPE))

        x_b = tf.reshape(x_b,[x_b.shape[0],1])
        y_b = tf.reshape(y_b,[y_b.shape[0],1])
        z_b = tf.reshape(z_b,[z_b.shape[0],1])
        phi_b = tf.reshape(phi_b,[phi_b.shape[0],1])
    
        X = tf.concat([x_b, y_b, z_b], axis=1)

        return X,phi_b

    @classmethod
    def get_X(cls,X):
        R = list()
        for i in range(X.shape[1]):
            R.append(X[:,i:i+1])
        return R

    @classmethod
    def stack_X(cls,x,y,z):
        R = tf.stack([x[:,0], y[:,0], z[:,0]], axis=1)
        return R
    
    def create_Dataset(cls,X):
        dataset_X = tf.data.Dataset.from_tensor_slices(X)
        dataset_X = dataset_X.shuffle(buffer_size=len(X))
        X_batches = dataset_X.batch(int(len(X)/cls.N_batches))
        return X_batches

    def create_Datasets(cls, X, Y):
        dataset_XY = tf.data.Dataset.from_tensor_slices((X, Y))
        dataset_XY = dataset_XY.shuffle(buffer_size=len(X))
        XY_batches = dataset_XY.batch(int(len(X)/cls.N_batches))
        return XY_batches
    
    def value_u_b(self,x, y, z, value):
        n = x.shape[0]
        return tf.ones((n,1), dtype=self.DTYPE)*value

    


class Molecule_Mesh():

    DTYPE = 'float32'
    pi = np.pi

    def __init__(self, molecule, N_points, refinement=True, N_batches=1):

        for key, value in N_points.items():
            setattr(self, key, value)
        self.N_batches = N_batches
        self.molecule = molecule
        self.refinement = refinement

        self.read_create_meshes(Mesh)

        self.domain_mesh_names = set()
        self.domain_mesh_data = dict()
        self.domain_mesh_N = dict()

    
    def read_create_meshes(self, Mesh_class):

        path_files = os.path.join(os.getcwd(),'utils3d','Model','Mesh')

        with open(os.path.join(path_files,self.molecule,self.molecule+'.face'),'r') as face_f:
            face = face_f.read()
        with open(os.path.join(path_files,self.molecule,self.molecule+'.vert'),'r') as vert_f:
            vert = vert_f.read()

        self.faces = np.vstack(np.char.split(face.split("\n")[0:-1]))[:, :3].astype(int) - 1
        verts = np.vstack(np.char.split(vert.split("\n")[0:-1]))[:, :3].astype(float)
        self.centroid = np.mean(verts, axis=0)
        self.verts = verts - self.centroid

        self.normal = np.vstack(np.char.split(vert.split("\n")[0:-1]))[:, 3:6].astype(np.float32)

        R_max_mol = np.max(self.verts)
        R_min_mol = np.min(self.verts)
        self.R_mol = np.max([R_max_mol,R_min_mol])

        self.mesh = trimesh.Trimesh(vertices=self.verts, faces=self.faces)

        self.interior_obj, self.exterior_obj = self.create_mesh_objs(Mesh_class)

        self.interior_obj.lb = [-self.R_mol,-self.R_mol,-self.R_mol]
        self.interior_obj.ub = [self.R_mol,self.R_mol,self.R_mol]
        self.interior_obj.N_batches = self.N_batches

        self.exterior_obj.lb = [-self.R_exterior,-self.R_exterior,-self.R_exterior]
        self.exterior_obj.ub = [self.R_exterior,self.R_exterior,self.R_exterior]    
        self.exterior_obj.N_batches = self.N_batches


    def create_mesh_objs(self, Mesh_class):
        
        mesh_interior = Mesh_class(name=1, molecule=self.molecule)
        mesh_exterior = Mesh_class(name=2, molecule=self.molecule)

        #########################################################################

        xspace = np.linspace(-self.R_mol, self.R_mol, self.N_interior, dtype=self.DTYPE)
        yspace = np.linspace(-self.R_mol, self.R_mol, self.N_interior, dtype=self.DTYPE)
        zspace = np.linspace(-self.R_mol, self.R_mol, self.N_interior, dtype=self.DTYPE)
        X, Y, Z = np.meshgrid(xspace, yspace, zspace)

        points = np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=1)
        interior_points_bool = self.mesh.contains(points)
        interior_points = points[interior_points_bool]

        if self.refinement:
            path_files = os.path.join(os.getcwd(),'utils3d','Model','Molecules')
            _,Lx_q,_,_,_,_ = import_charges_from_pqr(os.path.join(path_files,self.molecule,self.molecule+'.pqr'))

            delta = 0.04
            dx = np.array([delta,0.0,0.0])
            dy = np.array([0.0,delta,0.0])
            dz = np.array([0.0,0.0,delta])

            n_q = Lx_q.shape[0]
            X_temp = np.zeros((n_q*7,3))

            for i,x_q in enumerate(Lx_q):
                c = 1
                x_q = x_q - self.centroid
                X_temp[i*7,:] = x_q
                for k in [-1,1]:
                    X_temp[i*7+c,:] = x_q + k*dx
                    X_temp[i*7+c+1,:] = x_q + k*dy
                    X_temp[i*7+c+2,:] = x_q + k*dz
                    c = 4
            X_in = tf.constant(X_temp, dtype=self.DTYPE)
            interior_points = tf.concat([interior_points,X_in], axis=0)

        mesh_interior.prior_data['R'] = tf.constant(interior_points)

        #########################################################################

        self.R_exterior =  self.R_mol+self.dR_exterior
        xspace = np.linspace(-self.R_exterior, self.R_exterior, self.N_exterior, dtype=self.DTYPE)
        yspace = np.linspace(-self.R_exterior, self.R_exterior, self.N_exterior, dtype=self.DTYPE)
        zspace = np.linspace(-self.R_exterior, self.R_exterior, self.N_exterior, dtype=self.DTYPE)
        X, Y, Z = np.meshgrid(xspace, yspace, zspace)

        points = np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=1)
        interior_points_bool = self.mesh.contains(points)

        exterior_points_bool = ~interior_points_bool  
        exterior_points = points[exterior_points_bool]

        exterior_distances = np.linalg.norm(exterior_points, axis=1)
        exterior_points = exterior_points[exterior_distances <= self.R_exterior]


        r_bl = np.linspace(self.R_exterior, self.R_exterior, self.N_border, dtype=self.DTYPE)
        theta_bl = np.linspace(0, self.pi, self.N_border, dtype=self.DTYPE)
        phi_bl = np.linspace(0, 2*self.pi, self.N_border, dtype=self.DTYPE)
        
        R_bl, Theta_bl, Phi_bl = np.meshgrid(r_bl, theta_bl, phi_bl)
        X_bl = R_bl*np.sin(Theta_bl)*np.cos(Phi_bl)
        Y_bl = R_bl*np.sin(Theta_bl)*np.sin(Phi_bl)
        Z_bl = R_bl*np.cos(Theta_bl)
        
        x_bl = tf.constant(X_bl.flatten())
        x_bl = tf.reshape(x_bl,[x_bl.shape[0],1])
        y_bl = tf.constant(Y_bl.flatten())
        y_bl = tf.reshape(y_bl,[y_bl.shape[0],1])
        z_bl = tf.constant(Z_bl.flatten())
        z_bl = tf.reshape(z_bl,[z_bl.shape[0],1])
    
        mesh_exterior.prior_data['R'] = tf.constant(exterior_points)
        mesh_exterior.prior_data['D'] = tf.concat([x_bl, y_bl, z_bl], axis=1)

        #############################################################################

        # N = int(mesh_length/mesh_dx)

        return mesh_interior, mesh_exterior
    

    def adapt_meshes_domain(self,data,q_list):
        
        for bl in data.values():
            type_b = bl['type']

            if type_b in ('I'):
                N = self.normal
                X = tf.constant(self.verts, dtype=self.DTYPE)
                X_I = self.interior_obj.create_Datasets(X, N)
                self.domain_mesh_names.add(type_b)
                self.domain_mesh_data[type_b] = X_I
                self.domain_mesh_N[type_b] = len(X)

            elif type_b in ('E'):
                file = bl['file']
                path_files = os.path.join(os.getcwd(),'utils3d','Model','Molecules')

                with open(os.path.join(path_files,self.molecule,file),'r') as f:
                    L_phi = dict()
                    for line in f:
                        res_n, phi = line.strip().split()
                        L_phi[str(res_n)] = float(phi)

                X_exp = list()

                for q in q_list:
                    if q.atom_name == 'H' and str(q.res_num) in L_phi:
                        mesh_length = 10
                        mesh_dx = 1
                        N = int(mesh_length / mesh_dx)
                        x = np.linspace(q.x_q[0] - mesh_length / 2, q.x_q[0] + mesh_length / 2, num=N)
                        y = np.linspace(q.x_q[1] - mesh_length / 2, q.x_q[1] + mesh_length / 2, num=N)
                        z = np.linspace(q.x_q[2] - mesh_length / 2, q.x_q[2] + mesh_length / 2, num=N)

                        X, Y, Z = np.meshgrid(x, y, z)
                        pos_mesh = np.zeros((3, N * N * N))
                        pos_mesh[0, :] = X.flatten()
                        pos_mesh[1, :] = Y.flatten()
                        pos_mesh[2, :] = Z.flatten()

                        for q_check in q_list:

                            x_diff = q_check.x_q[0] - pos_mesh[0, :]
                            y_diff = q_check.x_q[1] - pos_mesh[1, :]
                            z_diff = q_check.x_q[2] - pos_mesh[2, :]
                    
                            r_q_mesh = np.sqrt(x_diff**2 + y_diff**2 + z_diff**2)

                            explode_local_index = np.nonzero(r_q_mesh >= q_check.r_explode)[0]

                            pos_mesh = pos_mesh[:, explode_local_index]

                        explode_points = pos_mesh.transpose()
                        interior_points_bool = self.mesh.contains(explode_points)
                        interior_points = explode_points[interior_points_bool]
                        
                        exterior_points_bool = ~interior_points_bool  
                        exterior_points = explode_points[exterior_points_bool]

                        exterior_distances = np.linalg.norm(exterior_points, axis=1)
                        exterior_points = exterior_points[exterior_distances <= self.R_exterior]

                        X1 = tf.constant(interior_points, dtype=self.DTYPE)
                        X2 = tf.constant(exterior_points, dtype=self.DTYPE)

                        x_diff = q.x_q[0] - interior_points[:,0]
                        y_diff = q.x_q[1] - interior_points[:,1]
                        z_diff = q.x_q[2] - interior_points[:,2]
                        r1 = np.sqrt(x_diff**2 + y_diff**2 + z_diff**2)

                        r1 = tf.constant(r1, dtype=self.DTYPE)
                        r1 = tf.reshape(r1,[r1.shape[0],1])

                        x_diff = q.x_q[0] - exterior_points[:,0]
                        y_diff = q.x_q[1] - exterior_points[:,1]
                        z_diff = q.x_q[2] - exterior_points[:,2]
                        r2 = np.sqrt(x_diff**2 + y_diff**2 + z_diff**2)

                        r2 = tf.constant(r2, dtype=self.DTYPE)
                        r2 = tf.reshape(r2,[r2.shape[0],1])

                        X_in = self.create_Datasets(X1,r1)
                        X_out = self.create_Datasets(X2,r2)
                        phi_ens = tf.constant(L_phi[str(q.res_num)] , dtype=self.DTYPE)

                        X_exp.append((X_in,X_out,phi_ens))

                self.domain_mesh_names.add(type_b)
                self.domain_mesh_data[type_b] = X_exp



    def create_Dataset(cls,X):
        dataset_X = tf.data.Dataset.from_tensor_slices(X)
        if len(X) == 0:
            return None
        dataset_X = dataset_X.shuffle(buffer_size=len(X))
        X_batches = dataset_X.batch(int(len(X)))
        return next(iter(X_batches))

    def create_Datasets(cls, X, Y):
        dataset_XY = tf.data.Dataset.from_tensor_slices((X, Y))
        if len(X) == 0:
            return None
        dataset_XY = dataset_XY.shuffle(buffer_size=len(X))
        XY_batches = dataset_XY.batch(int(len(X)))
        return next(iter(XY_batches))


    def plot_molecule(self):
        mesh = self.mesh
        mesh.visual.vertex_colors = [0.0, 0.0, 0.0, 1.0] 
        mesh.visual.face_colors = [1.0, 0.0, 0.0, 0.9]  
        mesh.show()


    def plot_mesh_2d(self):

        fig, ax = plt.subplots()

        xm,ym,zm = self.interior_obj.get_X(self.interior_obj.solver_mesh_data['R'])
        plane = np.abs(zm) < 0.5
        ax.scatter(xm[plane], ym[plane], c='r', marker='.', alpha=0.1)

        xm,ym,zm = self.exterior_obj.get_X(self.exterior_obj.solver_mesh_data['R'])
        plane = np.abs(zm) < 2
        ax.scatter(xm[plane], ym[plane], c='g', marker='.', alpha=0.1)

        xm,ym,zm = self.interior_obj.get_X(self.interior_obj.solver_mesh_data['I'])
        plane = np.abs(zm) < 0.1
        ax.scatter(xm[plane], ym[plane], c='b', marker='.', alpha=0.3)

        xm,ym,zm = self.exterior_obj.get_X(self.exterior_obj.solver_mesh_data['D'])
        plane = np.abs(zm) < 2
        ax.scatter(xm[plane], ym[plane], c='m', marker='.', alpha=0.1)

        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_title('Positions of collocation points and boundary data')

        ax.set_xlim([-self.R_exterior,self.R_exterior])
        ax.set_ylim([-self.R_exterior,self.R_exterior])

        plt.show()


    def plot_mesh_3d(self):

        fig = px.scatter_3d()

        xm,ym,zm = self.interior_obj.get_X(self.interior_obj.prior_data['R'])
        fig.add_scatter3d(x=xm, y=ym, z=zm, mode="markers", marker=dict(size=5, color="red"), name="Inner Points")

        xm,ym,zm = self.exterior_obj.get_X(self.exterior_obj.prior_data['R'])
        fig.add_scatter3d(x=xm, y=ym, z=zm, mode="markers", marker=dict(size=5, color="blue"), name="Outer Points")

        xm,ym,zm = self.interior_obj.get_X(self.interior_obj.prior_data['I'])
        fig.add_scatter3d(x=xm, y=ym, z=zm, mode="markers", marker=dict(size=5, color="green"), name="Interface Points")

        xm,ym,zm = self.exterior_obj.get_X(self.exterior_obj.prior_data['D'])
        fig.add_scatter3d(x=xm, y=ym, z=zm, mode="markers", marker=dict(size=5, color="yellow"), name="Border Points")


        fig.update_layout(scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"))

        fig.show()



if __name__=='__main__':

    N_points = {'N_interior': 19,
              'N_exterior': 20,
              'N_border': 20,
              'dR_exterior': 20
             }

    molecule_mesh = Molecule_Mesh('1ubq', N_points)
    mesh_domain = {'type': 'E', 'file': 'data_experimental.dat'}

    # path_files = os.path.join(os.getcwd(),'utils3d','Model','Molecules')
    # q_list = get_charges_list(os.path.join(path_files,'1ubq','1ubq'+'.pqr'))
    # molecule_mesh.adapt_meshes_domain(mesh_domain,q_list)
    # #molecule_mesh.plot_molecule()


