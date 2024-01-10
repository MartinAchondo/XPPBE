
import os
import numpy as np
import tensorflow as tf
import trimesh
import matplotlib.pyplot as plt

from Mesh.Solver_Mesh import Solver_Mesh
from Mesh.Charges_utils import import_charges_from_pqr


class Molecule_Mesh():

    DTYPE = 'float32'
    pi = np.pi

    def __init__(self, molecule, N_points, simulation='Main', path='', plot=False, result_path=''):
        
        self.mesh_density = 3
        for key, value in N_points.items():
            setattr(self, key, value)
        self.molecule = molecule
        self.plot = plot
        self.main_path = path
        self.simulation_name = simulation
        self.result_path = self.main_path if result_path=='' else result_path

        self.read_create_meshes(Solver_Mesh)

        self.domain_mesh_names = set()
        self.domain_mesh_data = dict()
        self.domain_mesh_N = dict()
    
    def read_create_meshes(self, Mesh_class):

        path_files = os.path.join(self.main_path,'Molecules','Saved_meshes')

        with open(os.path.join(path_files,self.molecule,self.molecule+f'_d{self.mesh_density}'+'.face'),'r') as face_f:
            face = face_f.read()
        with open(os.path.join(path_files,self.molecule,self.molecule+f'_d{self.mesh_density}'+'.vert'),'r') as vert_f:
            vert = vert_f.read()

        self.faces = np.vstack(np.char.split(face.split("\n")[0:-1]))[:, :3].astype(int) - 1
        self.verts = np.vstack(np.char.split(vert.split("\n")[0:-1]))[:, :3].astype(float)
        self.normal = np.vstack(np.char.split(vert.split("\n")[0:-1]))[:, 3:6].astype(np.float32)
    
        self.centroid = np.mean(self.verts, axis=0)

        vertx = self.verts[self.faces]
        self.areas = np.linalg.norm(np.cross(vertx[:, 1, :] - vertx[:, 0, :], vertx[:, 2, :] - vertx[:, 0, :]), axis=1) / 2

        r = np.sqrt((self.verts[:,0]-self.centroid[0])**2 + (self.verts[:,1]-self.centroid[1])**2 + (self.verts[:,2]-self.centroid[2])**2)
        self.R_mol = np.max(r)
    
        self.mesh = trimesh.Trimesh(vertices=self.verts, faces=self.faces)

        self.interior_obj, self.exterior_obj = self.create_mesh_objs(Mesh_class)

        print("Mesh initialization ready")

    def create_mesh_objs(self, Mesh_class):
        
        mesh_interior = Mesh_class(name=1, molecule=self.molecule, path=self.main_path)
        mesh_exterior = Mesh_class(name=2, molecule=self.molecule, path=self.main_path)

        #########################################################################

        xmax, ymax, zmax = np.max(self.verts, axis=0)
        xmin, ymin, zmin = np.min(self.verts, axis=0)
        mesh_interior.lb = [xmin,ymin,zmin]
        mesh_interior.ub = [xmax,ymax,zmax]

        self.N_interior = int(2*self.R_mol/self.dx_interior)

        xspace = np.linspace(-self.R_mol, self.R_mol, self.N_interior, dtype=self.DTYPE) + self.centroid[0]
        yspace = np.linspace(-self.R_mol, self.R_mol, self.N_interior, dtype=self.DTYPE) + self.centroid[1]
        zspace = np.linspace(-self.R_mol, self.R_mol, self.N_interior, dtype=self.DTYPE) + self.centroid[2]
        X, Y, Z = np.meshgrid(xspace, yspace, zspace)

        points = np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=1)
        interior_points_bool = self.mesh.contains(points)
        interior_points = points[interior_points_bool]
        mesh_interior.prior_data['R'] = tf.constant(interior_points)

        #########################################################################

        path_files = os.path.join(self.main_path,'Molecules')
        _,Lx_q,R_q,_,_,_ = import_charges_from_pqr(os.path.join(path_files,self.molecule,self.molecule+'.pqr'))

        for x_q,r_q in zip(Lx_q,R_q):
            X_in = self.generate_charge_dataset(x_q,r_q)

            if not 'Q' in mesh_interior.prior_data:
                mesh_interior.prior_data['Q'] = X_in
            else:
                mesh_interior.prior_data['Q'] = tf.concat([mesh_interior.prior_data['Q'],X_in], axis=0)

        #########################################################################

        self.R_exterior =  self.R_mol+self.dR_exterior

        self.N_exterior = int(2*self.R_exterior/self.dx_exterior)
        xspace = np.linspace(-self.R_exterior, self.R_exterior, self.N_exterior, dtype=self.DTYPE) + self.centroid[0]
        yspace = np.linspace(-self.R_exterior, self.R_exterior, self.N_exterior, dtype=self.DTYPE) + self.centroid[1]
        zspace = np.linspace(-self.R_exterior, self.R_exterior, self.N_exterior, dtype=self.DTYPE) + self.centroid[2]
        X, Y, Z = np.meshgrid(xspace, yspace, zspace)

        points = np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=1)
        interior_points_bool = self.mesh.contains(points)

        exterior_points_bool = ~interior_points_bool  
        exterior_points = points[exterior_points_bool]

        exterior_distances = np.linalg.norm(exterior_points-self.centroid, axis=1)
        exterior_points = exterior_points[exterior_distances <= self.R_exterior]

        r_bl = np.linspace(self.R_exterior, self.R_exterior, self.N_border, dtype=self.DTYPE)
        theta_bl = np.linspace(0, 2*self.pi, self.N_border, dtype=self.DTYPE)
        phi_bl = np.linspace(0, self.pi, self.N_border, dtype=self.DTYPE)
        
        R_bl, Theta_bl, Phi_bl = np.meshgrid(r_bl, theta_bl, phi_bl)
        X_bl = self.centroid[0] + R_bl*np.sin(Phi_bl)*np.cos(Theta_bl)
        Y_bl = self.centroid[1] + R_bl*np.sin(Phi_bl)*np.sin(Theta_bl)
        Z_bl = self.centroid[2] + R_bl*np.cos(Phi_bl)
        
        x_bl = tf.constant(X_bl.flatten())
        x_bl = tf.reshape(x_bl,[x_bl.shape[0],1])
        y_bl = tf.constant(Y_bl.flatten())
        y_bl = tf.reshape(y_bl,[y_bl.shape[0],1])
        z_bl = tf.constant(Z_bl.flatten())
        z_bl = tf.reshape(z_bl,[z_bl.shape[0],1])
    
        mesh_exterior.prior_data['R'] = tf.constant(exterior_points)
        mesh_exterior.prior_data['D'] = tf.concat([x_bl, y_bl, z_bl], axis=1)

        xmax, ymax, zmax = self.R_exterior + self.centroid
        xmin, ymin, zmin = -self.R_exterior + self.centroid
        mesh_exterior.lb = [xmin,ymin,zmin]
        mesh_exterior.ub = [xmax,ymax,zmax] 

        #############################################################################

        if self.plot:
            X_plot = dict()
            X_plot['Inner Domain'] = interior_points
            X_plot['Charges'] = mesh_interior.prior_data['Q'].numpy()
            X_plot['Interface'] = self.verts
            X_plot['Outer Domain'] = exterior_points
            X_plot['Outer Border'] = np.column_stack((X_bl.ravel(), Y_bl.ravel(), Z_bl.ravel()))
            self.save_data_plot(X_plot)

        return mesh_interior, mesh_exterior
    

    def generate_charge_dataset(self,x_q,r_q):            
    
        x_q_tensor = tf.constant(x_q, dtype=self.DTYPE)
        sigma = self.G_sigma*3 if self.G_sigma<0.8*r_q else 0.8*r_q
        r = sigma * tf.sqrt(tf.random.uniform(shape=(self.N_pq,), minval=0, maxval=1))
        theta = tf.random.uniform(shape=(self.N_pq,), minval=0, maxval=2*np.pi)
        phi = tf.random.uniform(shape=(self.N_pq,), minval=0, maxval=np.pi)

        x_random = x_q[0] + r * tf.sin(phi) * tf.cos(theta)
        y_random = x_q[1] + r * tf.sin(phi) * tf.sin(theta)
        z_random = x_q[2] + r * tf.cos(phi)
        X_in = tf.concat([tf.reshape(x_q_tensor,[1,3]), tf.stack([x_random, y_random, z_random], axis=-1)], axis=0)

        return X_in
    

    def adapt_meshes_domain(self,data,q_list):
        
        for bl in data.values():
            type_b = bl['type']

            if type_b in ('I'):
                N = self.normal
                X = tf.constant(self.verts, dtype=self.DTYPE)
                X_I = self.interior_obj.create_Datasets(X, N)
                self.domain_mesh_names.add('Iu')
                self.domain_mesh_names.add('Id')
                self.domain_mesh_data[type_b] = X_I
                self.domain_mesh_N[type_b] = len(X)
            
            elif type_b in ('G'):
                self.domain_mesh_names.add(type_b)
                self.domain_mesh_data[type_b] = None

            elif type_b in ('E'):
                file = bl['file']
                path_files = os.path.join(self.main_path,'Molecules')

                with open(os.path.join(path_files,self.molecule,file),'r') as f:
                    L_phi = dict()
                    for line in f:
                        res_n, phi = line.strip().split()
                        L_phi[str(res_n)] = float(phi)

                X_exp = list()
                X_exp_values = list()

                mesh_length = self.R_exterior*2
                mesh_dx = self.dx_experimental
                N = int(mesh_length / mesh_dx)
                x = np.linspace(-mesh_length / 2, mesh_length / 2, num=N) + self.centroid[0]
                y = np.linspace(-mesh_length / 2, mesh_length / 2, num=N) + self.centroid[1]
                z = np.linspace(-mesh_length / 2, mesh_length / 2, num=N) + self.centroid[2]

                X, Y, Z = np.meshgrid(x, y, z)
                pos_mesh = np.stack((X.flatten(), Y.flatten(), Z.flatten()), axis=0)

                for q_check in q_list:

                    x_diff, y_diff, z_diff = q_check.x_q[:, np.newaxis] - pos_mesh
                    r_q_mesh = np.sqrt(x_diff**2 + y_diff**2 + z_diff**2)
                    explode_local_index = np.nonzero(r_q_mesh >= q_check.r_explode)[0]
                    pos_mesh = pos_mesh[:, explode_local_index]

                explode_points = pos_mesh.transpose()
                interior_points_bool = self.mesh.contains(explode_points)
                interior_points = explode_points[interior_points_bool]
                
                exterior_points_bool = ~interior_points_bool  
                exterior_points = explode_points[exterior_points_bool]

                exterior_distances = np.linalg.norm(exterior_points-self.centroid, axis=1)
                exterior_points = exterior_points[exterior_distances <= self.R_exterior]

                X1 = tf.constant(interior_points, dtype=self.DTYPE)
                X2 = tf.constant(exterior_points, dtype=self.DTYPE)

                X_in = self.create_Dataset_and_Tensor(X1)
                X_out = self.create_Dataset_and_Tensor(X2)

                X_exp.append((X_in,X_out))
                
                for q in q_list:
                    if q.atom_name == 'H' and str(q.res_num) in L_phi:

                        phi_ens = tf.constant(L_phi[str(q.res_num)] , dtype=self.DTYPE)
                        xq = tf.reshape(tf.constant(q.x_q, dtype=self.DTYPE), (1,3))
                        X_exp_values.append((xq,phi_ens))

                X_exp.append(X_exp_values)
                self.domain_mesh_names.add(type_b)
                self.domain_mesh_data[type_b] = X_exp

                if self.plot:
                    X_plot = dict()
                    X_plot['Experimental'] = explode_points[np.linalg.norm(explode_points-self.centroid, axis=1) <= self.R_exterior]
                    self.save_data_plot(X_plot)


    def create_Dataset_and_Tensor(cls,X):
        if len(X) == 0:
            return None
        dataset_X = tf.data.Dataset.from_tensor_slices(X)
        X_batches = dataset_X.batch(int(len(X)))
        return next(iter(X_batches))

    def save_data_plot(self,X_plot):
        path_files = os.path.join(self.result_path,'results',self.simulation_name,'mesh')
        os.makedirs(path_files, exist_ok=True)

        for subset_name, subset_data in X_plot.items():
            file_name = os.path.join(path_files,f'{subset_name}.csv')
            np.savetxt(file_name, subset_data, delimiter=',', header='X,Y,Z', comments='')


    def plot_molecule(self):
        mesh = self.mesh
        mesh.visual.vertex_colors = [0.0, 0.0, 0.0, 1.0] 
        mesh.visual.face_colors = [1.0, 0.0, 0.0, 0.9]  
        mesh.show()


if __name__=='__main__':

    N_points = {'N_interior': 19,
              'N_exterior': 20,
              'N_border': 20,
              'dR_exterior': 20
             }

    molecule_mesh = Molecule_Mesh('1ubq', N_points)
    mesh_domain = {'type': 'E', 'file': 'data_experimental.dat'}

    # path_files = os.path.join(os.getcwd(),'code','Model','Molecules')
    # q_list = get_charges_list(os.path.join(path_files,'1ubq','1ubq'+'.pqr'))
    # molecule_mesh.adapt_meshes_domain(mesh_domain,q_list)
    # #molecule_mesh.plot_molecule()

