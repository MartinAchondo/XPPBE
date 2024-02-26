import os
import numpy as np
import tensorflow as tf
import trimesh
import pygamer

from Mesh.Solver_Mesh import Solver_Mesh
from Mesh.Charges_utils import import_charges_from_pqr, convert_pqr2xyzr
from Mesh.Mesh_utils  import generate_msms_mesh,generate_nanoshaper_mesh

class Region_Mesh():

    DTYPE = 'float32'
    pi = np.pi

    def __init__(self,type_m,obj,vertices=None,elements=None,*args,**kwargs):
        self.type_m = type_m
        self.obj = obj
        if self.type_m=='trimesh':
            self.vertices = obj.vertices if vertices is None else vertices
            self.elements = obj.faces if elements is None else elements
        if self.type_m=='tetmesh':
            self.vertices = np.array([list(vID.data().position) for vID in obj.vertexIDs]) if vertices is None else vertices
            self.elements = np.array([list(i.indices()) for i in obj.cellIDs]) if elements is None else elements
        if self.type_m=='points':
            self.vertices = None
            self.elements = None
            for key, value in kwargs.items():
                setattr(self, key, value)

    def get_dataset(self):
        if self.type_m=='trimesh':
            dataset = self.random_points_in_elements(self.vertices,self.elements,3)
        if self.type_m=='tetmesh':
            dataset = self.random_points_in_elements(self.vertices,self.elements,4)
        if self.type_m=='points':
            dataset = self.random_points_near_charges(self.charges[0],self.charges[1])
        return dataset
    
    def random_points_in_elements(vertices, elements,num_vert_per_elem):
        num_elements = len(elements)
        random_coordinates = np.random.uniform(0, 1, size=(num_elements, num_vert_per_elem))
        normalization_factors = np.sum(random_coordinates, axis=1, keepdims=True)
        random_coordinates /= normalization_factors
        random_points = np.sum(vertices[elements] * random_coordinates[:, :, np.newaxis], axis=1)
        return random_points

    def random_points_near_charges(self,Lx_q,R_q):
        random_points = None
        for x_q,r_q in zip(Lx_q,R_q):
            X_in = self.generate_one_charge_dataset(x_q,r_q)
            if random_points is None:
                random_points = X_in
            else:
                random_points = tf.concat([random_points,X_in], axis=0)
        return random_points

    def generate_one_charge_dataset(self,x_q,r_q):            
    
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


class Molecule_Mesh():

    DTYPE = 'float32'
    pi = np.pi

    def __init__(self, molecule, N_points, simulation='Main', path='', plot=False, result_path=''):


        for key, value in N_points.items():
            setattr(self, key, value)

        self.molecule = molecule
        self.plot = plot
        self.main_path = path
        self.simulation_name = simulation
        self.result_path = self.main_path if result_path=='' else result_path

        self.path_files = os.path.join(self.main_path,'Molecules','Saved_meshes','Temp')
        self.path_pqr = os.path.join(self.main_path,'Molecules',self.molecule,self.molecule+'.pqr')
        self.path_xyzr = os.path.join(self.main_path,'Molecules',self.molecule,self.molecule+'.xyzr')
        
        self.region_meshes = dict()

        self.read_create_meshes(Solver_Mesh)

        self.domain_mesh_names = set()
        self.domain_mesh_data = dict()
        self.domain_mesh_N = dict()
    
    def read_create_meshes(self, Mesh_class):

        self.create_molecule_mesh()
        self.create_sphere_mesh()
        self.create_interior_mesh()
        self.create_exterior_mesh()

        self.interior_obj, self.exterior_obj = self.create_mesh_objs(Mesh_class)

        print("Mesh initialization ready")



    def create_molecule_mesh(self):

        convert_pqr2xyzr(self.path_pqr,self.path_xyzr,for_mesh=True)

        if self.mesh_generator == 'msms':
            generate_msms_mesh(self.path_xyzr,self.path_files,self.molecule,self.density_mol)
        elif self.mesh_generator == 'nanoshaper':
            generate_nanoshaper_mesh(self.path_xyzr,self.path_files,self.molecule,self.density_mol)
            
        with open(os.path.join(self.path_files,self.molecule+f'_d{self.density_mol}'+'.face'),'r') as face_f:
            face = face_f.read()
        with open(os.path.join(self.path_files,self.molecule+f'_d{self.density_mol}'+'.vert'),'r') as vert_f:
            vert = vert_f.read()

        self.mol_faces = np.vstack(np.char.split(face.split("\n")[0:-1]))[:, :3].astype(int) - 1
        self.mol_verts = np.vstack(np.char.split(vert.split("\n")[0:-1]))[:, :3].astype(float)
        self.mol_normal = np.vstack(np.char.split(vert.split("\n")[0:-1]))[:, 3:6].astype(np.float32)
        self.centroid = np.mean(self.mol_verts, axis=0)

        vertx = self.mol_verts[self.mol_faces]

        mol_faces_normal = np.cross(vertx[:, 1, :] - vertx[:, 0, :], vertx[:, 2, :] - vertx[:, 0, :])
        self.mol_faces_normal = mol_faces_normal/np.linalg.norm(mol_faces_normal)

        self.mol_areas = np.linalg.norm(np.cross(vertx[:, 1, :] - vertx[:, 0, :], vertx[:, 2, :] - vertx[:, 0, :]), axis=1) / 2

        r = np.sqrt((self.mol_verts[:,0]-self.centroid[0])**2 + (self.mol_verts[:,1]-self.centroid[1])**2 + (self.mol_verts[:,2]-self.centroid[2])**2)
        self.R_mol = np.max(r)
    
        self.mol_mesh = trimesh.Trimesh(vertices=self.mol_verts, faces=self.mol_faces)

        self.mol_mesh.export(os.path.join(self.path_files,self.molecule+f'_d{self.density_mol}'+'.off'), file_type='off')

        self.region_meshes['I'] = Region_Mesh('trimesh',self.mol_mesh,self.mol_verts,self.mol_faces)
        self.region_meshes['I'].normals = self.mol_normal   # normals at verts, add at faces?
        self.region_meshes['I'].areas = self.mol_areas

    def create_sphere_mesh(self):
        r = self.R_mol + self.dR_exterior
        self.sphere_mesh = trimesh.creation.icosphere(radius=r, subdivisions=self.density_border)
        self.sphere_mesh.export(os.path.join(self.path_files,'mesh_sphere'+f'_d{self.density_border}'+'.off'),file_type='off')
        self.region_meshes['D'] = Region_Mesh('trimesh',self.sphere_mesh)

    def create_interior_mesh(self):

        mesh_molecule = pygamer.readOFF(os.path.join(self.path_files,self.molecule+f'_d{self.density_mol}'+'.off'))
        mesh_split = mesh_molecule.splitSurfaces() 

        print("Found %i meshes in 1"%len(mesh_split))
        
        mesh1 = mesh_split[0]  
        mesh1.correctNormals() 
        gInfo = mesh1.getRoot() 
        
        gInfo.ishole = False         
        meshes = [mesh1] 

        self.int_tetmesh = pygamer.makeTetMesh(meshes, '-pq'+str(self.hmin_interior)+'aYAO2/3')  

        self.region_meshes['R1'] = Region_Mesh('tetmesh',self.int_tetmesh)

    def create_exterior_mesh(self):
        mesh_molecule = pygamer.readOFF(os.path.join(self.path_files,self.molecule+f'_d{self.density_mol}'+'.off'))
        mesh_split = mesh_molecule.splitSurfaces() 
        print("Found %i meshes in 1"%len(mesh_split))
        
        mesh1 = mesh_split[0]  
        mesh1.correctNormals() 
        gInfo = mesh1.getRoot() 
        gInfo.ishole = True   

        mesh_sphere = pygamer.readOFF(os.path.join(self.path_files,'mesh_sphere'+f'_d{3}'+'.off'))
        mesh_sphere.correctNormals()   
        print("Found %i meshes in 2"%len(mesh_sphere.splitSurfaces()))
        gInfo = mesh_sphere.getRoot() 
        gInfo.ishole = False

        meshes = [mesh1,mesh_sphere] 

        self.ext_tetmesh = pygamer.makeTetMesh(meshes, '-pq'+str(self.hmin_exterior)+'aYAO2/3') 

        self.region_meshes['R2'] = Region_Mesh('tetmesh',self.ext_tetmesh)


    def create_mesh_objs(self, Mesh_class):
        
        mesh_interior = Mesh_class(name=1, molecule=self.molecule, path=self.main_path)
        mesh_exterior = Mesh_class(name=2, molecule=self.molecule, path=self.main_path)

        #########################################################################

        self.R_exterior =  self.R_mol+self.dR_exterior

        xmax, ymax, zmax = np.max(self.mol_verts, axis=0)
        xmin, ymin, zmin = np.min(self.mol_verts, axis=0)
        mesh_interior.lb = [xmin,ymin,zmin]
        mesh_interior.ub = [xmax,ymax,zmax]

        xmax, ymax, zmax = self.R_exterior + self.centroid
        xmin, ymin, zmin = -self.R_exterior + self.centroid
        mesh_exterior.lb = [xmin,ymin,zmin]
        mesh_exterior.ub = [xmax,ymax,zmax] 

        mesh_interior.prior_data['R'] = tf.constant(self.region_meshes['R1'].vertices, dtype=self.DTYPE)

        #########################################################################

        path_files = os.path.join(self.main_path,'Molecules')
        _,Lx_q,R_q,_,_,_ = import_charges_from_pqr(os.path.join(path_files,self.molecule,self.molecule+'.pqr'))

        self.region_meshes['Q'] = Region_Mesh(type_m='points', obj=None,charges=(Lx_q,R_q), G_sigma=self.G_sigma, N_pq=self.N_pq)
        mesh_interior.prior_data['Q'] = self.region_meshes['Q'].get_dataset()

        #########################################################################
    

        mesh_exterior.prior_data['R'] = tf.constant(self.region_meshes['R2'].vertices, dtype=self.DTYPE)
        mesh_exterior.prior_data['D'] = tf.constant(self.region_meshes['D'].vertices, dtype=self.DTYPE)


        #############################################################################

        if self.plot:
            X_plot = dict()
            X_plot['Inner Domain'] = self.region_meshes['R1'].vertices
            X_plot['Charges'] = mesh_interior.prior_data['Q'].numpy()
            X_plot['Interface'] = self.region_meshes['I'].vertices
            X_plot['Outer Domain'] = self.region_meshes['R2'].vertices
            X_plot['Outer Border'] = self.region_meshes['D'].vertices
            self.save_data_plot(X_plot)

        return mesh_interior, mesh_exterior
    


    def adapt_meshes_domain(self,data,q_list):
        
        for bl in data.values():
            type_b = bl['type']

            if type_b in ('I'):
                N = self.mol_normal
                X = tf.constant(self.mol_verts, dtype=self.DTYPE)
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
                interior_points_bool = self.mol_mesh.contains(explode_points)
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

    N_points = {'dx_interior': 1.2,
                'dx_exterior': 2.5,
                'N_border': 15,
                'dR_exterior': 4,
                'dx_experimental': 4,
                'N_pq': 10,
                'G_sigma': 0.04,
                'density_mol': 2,
                'density_2': 2,
                'mesh_generator': 'msms'
                }
    Mol_mesh = Molecule_Mesh('methanol', 
                                N_points=N_points, 
                                plot=True,
                                path=os.path.join(os.getcwd(),'code'),
                                simulation='test'
                                )
        
