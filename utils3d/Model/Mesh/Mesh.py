
import os
import numpy as np
import tensorflow as tf
import trimesh
import matplotlib.pyplot as plt
import plotly.express as px


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

        self.data_mesh = {
            'R': self.X_R,
            'D': self.XU_D,
            'N': self.XU_N,
            'K': self.XU_K, 
            'I': self.X_I,
            'P': None,
        }

        self.prior_data = dict()
        self.meshes_names = set()
        self.meshes_N = dict()

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
                self.data_mesh[type_b] = self.create_Datasets(X,U)
            
            elif type_b in ('I'):
                N = self.normal
                self.data_mesh[type_b] = self.create_Datasets(X,N)

            self.meshes_names.add(type_b)
            if type_b in self.meshes_N:
                self.meshes_N[type_b] += len(X)
            else:
                self.meshes_N[type_b] = len(X)
            
        del self.prior_data

    def read_file_data(self,file):
        x_b, y_b, z_b, phi_b = list(), list(), list(), list()

        path_files = os.path.join(os.getcwd(),'utils3d','Model','Molecules')

        with open(os.path.join(path_files,self.molecule,file),'r') as f:
            for line in f:
                condition, x, y, z, phi = line.strip().split(' ')

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

    def __init__(self, molecule, N_points, N_batches=1):

        for key, value in N_points.items():
            setattr(self, key, value)
        self.N_batches = N_batches

        self.molecule = molecule

        self.read_create_meshes(Mesh)

    
    def read_create_meshes(self, Mesh_class):

        path_files = os.path.join(os.getcwd(),'utils3d','Model','Molecules')

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

        self.interior_obj.normal = self.normal
        self.interior_obj.lb = [-self.R_mol,-self.R_mol,-self.R_mol]
        self.interior_obj.ub = [self.R_mol,self.R_mol,self.R_mol]
        self.interior_obj.N_batches = self.N_batches

        self.exterior_obj.normal = self.normal
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

        mesh_interior.prior_data['R'] = tf.constant(interior_points)
        mesh_interior.prior_data['I'] = tf.constant(self.verts, dtype=self.DTYPE)

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
        mesh_exterior.prior_data['I'] = tf.constant(self.verts, dtype=self.DTYPE)
        mesh_exterior.prior_data['D'] = tf.concat([x_bl, y_bl, z_bl], axis=1)

        #############################################################################

        return mesh_interior, mesh_exterior
    

    def plot_molecule(self):
        mesh = self.mesh
        mesh.visual.vertex_colors = [0.0, 0.0, 0.0, 1.0] 
        mesh.visual.face_colors = [1.0, 0.0, 0.0, 0.9]  
        mesh.show()


    def plot_mesh_2d(self):

        fig, ax = plt.subplots()

        xm,ym,zm = self.interior_obj.get_X(self.interior_obj.prior_data['R'])
        plane = np.abs(zm) < 0.5
        ax.scatter(xm[plane], ym[plane], c='r', marker='.', alpha=0.1)

        xm,ym,zm = self.exterior_obj.get_X(self.exterior_obj.prior_data['R'])
        plane = np.abs(zm) < 2
        ax.scatter(xm[plane], ym[plane], c='g', marker='.', alpha=0.1)

        xm,ym,zm = self.interior_obj.get_X(self.interior_obj.prior_data['I'])
        plane = np.abs(zm) < 0.1
        ax.scatter(xm[plane], ym[plane], c='b', marker='.', alpha=0.3)

        xm,ym,zm = self.exterior_obj.get_X(self.exterior_obj.prior_data['D'])
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
              'dR_exterior': 8
             }

    molecule_mesh = Molecule_Mesh('1ubq', N_points)

    #molecule_mesh.plot_molecule()


