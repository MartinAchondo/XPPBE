
import os
import numpy as np
import tensorflow as tf
import trimesh
import matplotlib.pyplot as plt

from Model.Mesh.Solver_Mesh import Solver_Mesh
from Model.Molecules.Charges import import_charges_from_pqr

    
class Molecule_Mesh():

    DTYPE = 'float32'
    pi = np.pi

    def __init__(self, molecule, N_points, refinement=False, plot=False, path=''):

        for key, value in N_points.items():
            setattr(self, key, value)
        self.molecule = molecule
        self.refinement = refinement
        self.plot = plot
        self.main_path = path

        self.read_create_meshes(Solver_Mesh)

        self.domain_mesh_names = set()
        self.domain_mesh_data = dict()
        self.domain_mesh_N = dict()
    
    def read_create_meshes(self, Mesh_class):

        path_files = os.path.join(self.main_path,'Model','Mesh')

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

        self.exterior_obj.lb = [-self.R_exterior,-self.R_exterior,-self.R_exterior]
        self.exterior_obj.ub = [self.R_exterior,self.R_exterior,self.R_exterior]    


    def create_mesh_objs(self, Mesh_class):
        
        mesh_interior = Mesh_class(name=1, molecule=self.molecule, path=self.main_path)
        mesh_exterior = Mesh_class(name=2, molecule=self.molecule, path=self.main_path)

        # N = int(mesh_length/mesh_dx)
        #########################################################################
        self.N_interior = int(2*self.R_mol/self.dx_interior)
        xspace = np.linspace(-self.R_mol, self.R_mol, self.N_interior, dtype=self.DTYPE)
        yspace = np.linspace(-self.R_mol, self.R_mol, self.N_interior, dtype=self.DTYPE)
        zspace = np.linspace(-self.R_mol, self.R_mol, self.N_interior, dtype=self.DTYPE)
        X, Y, Z = np.meshgrid(xspace, yspace, zspace)

        points = np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=1)
        interior_points_bool = self.mesh.contains(points)
        interior_points = points[interior_points_bool]

        if self.refinement:
            path_files = os.path.join(self.main_path,'Model','Molecules')
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
            interior_points_2 = tf.concat([interior_points,X_in], axis=0)

        else:
            interior_points_2 = interior_points

        mesh_interior.prior_data['R'] = tf.constant(interior_points_2)

        #########################################################################

        self.R_exterior =  self.R_mol+self.dR_exterior
        self.N_exterior = int(2*self.R_exterior/self.dx_exterior)
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

        if self.plot:
            X_plot = dict()
            X_plot['Inner Domain'] = interior_points
            X_plot['Charges'] = X_temp
            X_plot['Interface'] = self.verts
            X_plot['Outer Domain'] = exterior_points
            X_plot['Outer Border'] = np.column_stack((X_bl.ravel(), Y_bl.ravel(), Z_bl.ravel()))
            self.save_data_plot(X_plot)


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
                path_files = os.path.join(self.main_path,'Model','Molecules')

                with open(os.path.join(path_files,self.molecule,file),'r') as f:
                    L_phi = dict()
                    for line in f:
                        res_n, phi = line.strip().split()
                        L_phi[str(res_n)] = float(phi)

                X_exp = list()
                all_explode_points = list()

                for q in q_list:
                    if q.atom_name == 'H' and str(q.res_num) in L_phi:
                        mesh_length = self.R_exterior*2
                        mesh_dx = self.dx_experimental
                        N = int(mesh_length / mesh_dx)
                        x = np.linspace(q.x_q[0] - mesh_length / 2, q.x_q[0] + mesh_length / 2, num=N)
                        y = np.linspace(q.x_q[1] - mesh_length / 2, q.x_q[1] + mesh_length / 2, num=N)
                        z = np.linspace(q.x_q[2] - mesh_length / 2, q.x_q[2] + mesh_length / 2, num=N)

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

                        exterior_distances = np.linalg.norm(exterior_points, axis=1)
                        exterior_points = exterior_points[exterior_distances <= self.R_exterior]

                        X1 = tf.constant(interior_points, dtype=self.DTYPE)
                        X2 = tf.constant(exterior_points, dtype=self.DTYPE)

                        X_in = self.create_Dataset_and_Tensor(X1)
                        X_out = self.create_Dataset_and_Tensor(X2)
                        phi_ens = tf.constant(L_phi[str(q.res_num)] , dtype=self.DTYPE)
                        xq = tf.reshape(tf.constant(q.x_q, dtype=self.DTYPE), (1,3))

                        X_exp.append((X_in,X_out,xq,phi_ens))

                        if self.plot:
                            all_explode_points.append(explode_points[np.linalg.norm(explode_points, axis=1) <= self.R_exterior])

                self.domain_mesh_names.add(type_b)
                self.domain_mesh_data[type_b] = X_exp

                if self.plot:
                    X_plot = dict()
                    X_plot['Experimental'] = np.concatenate(all_explode_points)
                    self.save_data_plot(X_plot)


    def create_Dataset_and_Tensor(cls,X):
        if len(X) == 0:
            return None
        dataset_X = tf.data.Dataset.from_tensor_slices(X)
        dataset_X = dataset_X.shuffle(buffer_size=len(X))
        X_batches = dataset_X.batch(int(len(X)))
        return next(iter(X_batches))

    def save_data_plot(self,X_plot):
        path_files = os.path.join(self.main_path,'Post','Plot3d','data')
        os.makedirs(path_files, exist_ok=True)

        for subset_name, subset_data in X_plot.items():
            file_name = os.path.join(path_files,f'{subset_name}.csv')
            np.savetxt(file_name, subset_data, delimiter=',', header='X,Y,Z', comments='')


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


