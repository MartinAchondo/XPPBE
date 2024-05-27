import os
import shutil
import numpy as np
import tensorflow as tf
import trimesh
import pygamer
import logging

from .Charges_utils import import_charges_from_pqr, convert_pqr2xyzr, convert_pdb2pqr
from .Mesh_utils  import generate_msms_mesh,generate_nanoshaper_mesh

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
            if vertices is None:
                n_verts = obj.nVertices
                n_cells = obj.nCells
                self.vertices = np.zeros((n_verts,3), dtype='float32')
                self.elements = np.zeros((n_cells,4), dtype='int')
                for num,i in enumerate(obj.cellIDs):
                    indices = i.indices()
                    verts = [list(obj.getVertex([idx]).data().position) for idx in list(indices)]
                    self.elements[num,:] = indices
                    for j in range(4):
                        self.vertices[indices[j],:] = verts[j]
            else:
                self.vertices = vertices
                self.elements = elements
        if self.type_m=='points':
            for key, value in kwargs.items():
                setattr(self, key, value)
            self.vertices = self.get_dataset()
            self.elements = None

    def get_dataset(self):
        if self.type_m=='trimesh':
            dataset = self.random_points_in_elements(self.vertices,self.elements,3)
        if self.type_m=='tetmesh':
            dataset = self.random_points_in_elements(self.vertices,self.elements,4)
        if self.type_m=='points':
            dataset = self.random_points_near_charges(self.charges[0],self.charges[1])
        return tf.constant(dataset, dtype=self.DTYPE)
    
    @staticmethod
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


class Domain_Mesh():

    DTYPE = 'float32'
    pi = np.pi

    def __init__(self, molecule, mesh_properties, simulation='Main', path='', save_points=False, result_path='', molecule_path=''):

        self.mesh_properties = {
            'vol_max_interior': 0.04,
            'vol_max_exterior': 0.1,
            'density_mol': 10,
            'density_border': 4,
            'dx_experimental': 1.0,
            'N_pq': 100,
            'G_sigma': 0.04,
            'mesh_generator': 'msms',
            'probe_radius': 1.4,
            'dR_exterior': 6,
            'force_field': 'AMBER'
            }
        
        for key in self.mesh_properties:
            if key in mesh_properties:
                self.mesh_properties[key] = mesh_properties[key]
            setattr(self, key, self.mesh_properties[key])

        self.molecule = molecule
        self.save_points = save_points
        self.main_path = path
        self.simulation_name = simulation
        self.result_path = self.main_path if result_path=='' else result_path
        self.molecule_path = os.path.join(self.main_path,'Molecules',molecule) if molecule_path=='' else molecule_path

        self.path_files = os.path.join(self.main_path,'Mesh','Temp')
        if os.path.exists(self.path_files):
            shutil.rmtree(self.path_files)
        os.makedirs(self.path_files)
        self.path_pqr = os.path.join(self.molecule_path,self.molecule+'.pqr')
        self.path_xyzr = os.path.join(self.molecule_path,self.molecule+'.xyzr')
        self.path_pdb = os.path.join(self.molecule_path,self.molecule+'.pdb')

        self.region_meshes = dict()
        self.domain_mesh_names = set()
        self.domain_mesh_data = dict()

        self.read_create_meshes()
        
    
    def read_create_meshes(self):
        self.create_molecule_mesh()
        self.create_sphere_mesh()
        self.create_interior_mesh()
        self.create_exterior_mesh()
        self.create_charges_mesh()
        self.create_mesh_obj()
        print("Mesh initialization ready")


    def create_molecule_mesh(self):

        if not os.path.exists(self.path_pqr):
            convert_pdb2pqr(self.path_pdb,self.path_pqr,self.force_field)

        convert_pqr2xyzr(self.path_pqr,self.path_xyzr,for_mesh=True)

        if self.mesh_generator == 'msms':
            generate_msms_mesh(self.path_xyzr,self.path_files,self.molecule,self.density_mol,self.probe_radius)
        elif self.mesh_generator == 'nanoshaper':
            generate_nanoshaper_mesh(self.path_xyzr,self.path_files,self.molecule,self.density_mol,self.probe_radius)
            
        with open(os.path.join(self.path_files,self.molecule+f'_d{self.density_mol}'+'.face'),'r') as face_f:
            face = face_f.read()
            mol_faces = np.vstack(np.char.split(face.split("\n")[0:-1]))[:, :3].astype(int) - 1
        with open(os.path.join(self.path_files,self.molecule+f'_d{self.density_mol}'+'.vert'),'r') as vert_f:
            vert = vert_f.read()
            mol_verts = np.vstack(np.char.split(vert.split("\n")[0:-1]))[:, :3].astype(np.float32)

        self.mol_mesh = trimesh.Trimesh(vertices=mol_verts, faces=mol_faces)
        self.mol_mesh.export(os.path.join(self.path_files,self.molecule+f'_d{self.density_mol}'+'.off'), file_type='off')

        self.create_calculate_mesh_data()

        self.region_meshes['I'] = Region_Mesh('trimesh',self.mol_mesh,self.mol_verts,self.mol_faces)
        self.region_meshes['I'].normals = self.mol_faces_normal 
        self.region_meshes['I'].areas = self.mol_faces_areas

    def create_calculate_mesh_data(self):

        self.mol_faces = self.mol_mesh.faces
        self.mol_verts = self.mol_mesh.vertices

        self.centroid = np.mean(self.mol_verts, axis=0)
        
        self.mol_faces_normal = self.mol_mesh.face_normals.reshape(-1,3).astype(np.float32)
        self.mol_verts_normal = self.mol_mesh.vertex_normals.reshape(-1,3).astype(np.float32)

        vertx = self.mol_verts[self.mol_faces]
        mol_areas = 0.5*np.linalg.norm(np.cross(vertx[:, 1, :] - vertx[:, 0, :], vertx[:, 2, :] - vertx[:, 0, :]), axis=1).astype(np.float32)
        self.mol_faces_areas = mol_areas.reshape(-1,1)
        
        r = np.sqrt((self.mol_verts[:,0]-self.centroid[0])**2 + (self.mol_verts[:,1]-self.centroid[1])**2 + (self.mol_verts[:,2]-self.centroid[2])**2)
        self.R_mol = np.max(r)
        self.R_max_dist = np.max(np.sqrt((self.mol_verts[:,0])**2 + (self.mol_verts[:,1])**2 + (self.mol_verts[:,2])**2))
        
        self.mol_faces_centroid = np.mean(self.mol_verts[self.mol_faces], axis=1).reshape(-1,3).astype(np.float32)

    def create_sphere_mesh(self):
        r = self.R_max_dist + self.dR_exterior
        self.sphere_mesh = trimesh.creation.icosphere(radius=r, subdivisions=self.density_border)
        self.sphere_mesh.export(os.path.join(self.path_files,'mesh_sphere'+f'_d{self.density_border}'+'.off'),file_type='off')
        self.region_meshes['D2'] = Region_Mesh('trimesh',self.sphere_mesh)

    def create_interior_mesh(self):
        mesh_molecule = pygamer.readOFF(os.path.join(self.path_files,self.molecule+f'_d{self.density_mol}'+'.off'))
        mesh_split = mesh_molecule.splitSurfaces() 
        
        self.mesh_molecule_pyg = mesh_split[0]  
        self.mesh_molecule_pyg.correctNormals() 
        gInfo = self.mesh_molecule_pyg.getRoot() 
        gInfo.ishole = False   

        meshes = [self.mesh_molecule_pyg] 
        self.int_tetmesh = pygamer.makeTetMesh(meshes, f'pq1.2a{self.vol_max_interior}YAO2/3Q')  
        self.region_meshes['R1'] = Region_Mesh('tetmesh',self.int_tetmesh)

    def create_exterior_mesh(self):
        mesh_molecule = pygamer.readOFF(os.path.join(self.path_files,self.molecule+f'_d{self.density_mol}'+'.off'))
        mesh_split = mesh_molecule.splitSurfaces() 
        
        self.mesh_molecule_pyg_2 = mesh_split[0]  
        self.mesh_molecule_pyg_2.correctNormals() 
        gInfo = self.mesh_molecule_pyg_2.getRoot() 
        gInfo.ishole = True   

        self.mesh_sphere_pyg = pygamer.readOFF(os.path.join(self.path_files,'mesh_sphere'+f'_d{self.density_border}'+'.off'))
        self.mesh_sphere_pyg.correctNormals()   
        gInfo = self.mesh_sphere_pyg.getRoot() 
        gInfo.ishole = False

        meshes = [self.mesh_molecule_pyg_2,self.mesh_sphere_pyg] 
        self.ext_tetmesh = pygamer.makeTetMesh(meshes, f'pq1.2a{self.vol_max_exterior}YAO2/3Q') 
        self.region_meshes['R2'] = Region_Mesh('tetmesh',self.ext_tetmesh)

    def create_charges_mesh(self):

        _,Lx_q,R_q,_,_,_ = import_charges_from_pqr(self.path_pqr)
        self.region_meshes['Q1'] = Region_Mesh(type_m='points', obj=None,charges=(Lx_q,R_q), G_sigma=self.G_sigma, N_pq=self.N_pq)

    def create_mesh_obj(self):

        self.R_exterior =  self.R_max_dist+self.dR_exterior

        mol_min, mol_max = np.min(self.mol_verts, axis=0), np.max(self.mol_verts, axis=0)
        self.scale_1 = [mol_min.tolist(), mol_max.tolist()]

        sphere_min, sphere_max = np.min(self.sphere_mesh.vertices, axis=0), np.max(self.sphere_mesh.vertices, axis=0)
        self.scale_2 = [sphere_min.tolist(), sphere_max.tolist()]

        if self.save_points:
            X_plot = dict()
            for type_b,mesh in self.region_meshes.items():
                X_plot[f'{type_b}_verts'] = mesh.vertices
                X_plot[f'{type_b}_sample'] = mesh.get_dataset().numpy()
            self.save_data_plot(X_plot)

    def adapt_meshes_domain(self,data,q_list):

        self.meshes_info = data
        
        for bl in self.meshes_info.values():

            type_b = bl['type']
            flag = bl['domain'] 

            if type_b[0] in ('R','D','K','N','P','Q'): 
                if type_b in self.region_meshes:
                    X = tf.constant(self.region_meshes[type_b].vertices, dtype=self.DTYPE)
                else:
                    X = None
                X,U = self.get_XU(X,bl)
                self.domain_mesh_data[type_b] = ((X,U),flag)
                self.domain_mesh_names.add(type_b)

                if type_b[0]=='K':
                    if self.save_points: 
                        X_plot = dict()
                        X_plot[f'{type_b}_verts'] = X.numpy()
                        self.save_data_plot(X_plot)

            elif type_b in ('Iu','Id','Ir'):
                N = self.mol_verts_normal
                X = tf.constant(self.mol_verts, dtype=self.DTYPE)
                X_I = (X, N)
                self.domain_mesh_names.add(type_b)
                self.domain_mesh_data['I'] = (X_I,flag)
            
            elif type_b in ('G'):
                self.domain_mesh_names.add(type_b)
                self.domain_mesh_data[type_b] = ((tf.constant(self.mol_faces_centroid, dtype=self.DTYPE),
                                                  tf.constant(self.mol_faces_normal, dtype=self.DTYPE),
                                                  tf.constant(self.mol_faces_areas, dtype=self.DTYPE)
                                                  ),flag)

            elif type_b[0] in ('E'):
                file = bl['file']
                ens_method = bl['method']

                with open(os.path.join(self.molecule_path,file),'r') as f:
                    L_phi = dict()
                    for line in f:
                        num, phi = line.strip().split()
                        L_phi[str(num)] = float(phi)

                L_names = ['H']
                if 'sphere' in self.molecule  or self.molecule == 'born_ion':
                    L_names = ['I']

                X_exp = list()
                X_exp_values = list()

                mesh_length = self.R_exterior*2
                mesh_dx = self.dx_experimental
                N = int(mesh_length / mesh_dx)
                ctr = self.centroid
                x = np.linspace(ctr[0]-mesh_length / 2, ctr[0]+mesh_length / 2, num=N) 
                y = np.linspace(ctr[1]-mesh_length / 2, ctr[1]+mesh_length / 2, num=N) 
                z = np.linspace(ctr[2]-mesh_length / 2, ctr[2]+mesh_length / 2, num=N) 

                X, Y, Z = np.meshgrid(x, y, z)
                pos_mesh = np.stack((X.flatten(), Y.flatten(), Z.flatten()), axis=0)

                for q_check in q_list:
                    r_q_mesh = np.linalg.norm(q_check.x_q[:, np.newaxis] - pos_mesh, axis=0)
                    explode_local_index = np.nonzero(r_q_mesh >= q_check.r_explode)[0]
                    pos_mesh = pos_mesh[:, explode_local_index]

                explode_points = pos_mesh.transpose()
                exterior_points_bool = ~self.mol_mesh.contains(explode_points)
                exterior_points = explode_points[exterior_points_bool]

                exterior_distances = np.linalg.norm(exterior_points, axis=1)
                exterior_points = exterior_points[exterior_distances <= self.R_exterior]

                X_out = tf.constant(exterior_points, dtype=self.DTYPE)
                X_exp.append(X_out)
                
                j = 1
                for q in q_list:
                    if q.atom_name in L_names:
                        if str(j) in L_phi:
                            phi_ens = tf.constant(L_phi[str(j)] , dtype=self.DTYPE)
                            xq = tf.reshape(tf.constant(q.x_q, dtype=self.DTYPE), (1,3))
                            r_q = tf.constant(q.r_q, dtype=self.DTYPE)
                            X_exp_values.append(((xq,r_q),phi_ens))
                        j += 1

                X_exp.append(X_exp_values)
                self.domain_mesh_names.add(type_b)
                self.domain_mesh_data[type_b] = (X_exp,flag,ens_method)

                if self.save_points:
                    X_plot = dict()
                    X_plot['E2_verts'] = exterior_points
                    self.save_data_plot(X_plot)


    def get_XU(self,X,bl):  

        value = bl['value'] if 'value' in bl else None
        fun = bl['fun'] if 'fun' in bl else None
        file = bl['file'] if 'file' in bl else None

        if value != None:
            x,y,z = self.get_X(X)
            U = self.value_u_b(x, y, z, value=value)
        elif fun != None:
            U = fun(X)
        elif file != None:
            X,U = self.read_file_data(file,bl['domain'])
            noise = bl['noise'] if 'noise' in bl else False
            if noise:
                U = U*self.add_noise(U)
        return X,U

    @staticmethod
    def add_noise(x):    
        n = x.shape[0]
        mu, sigma = 1, 0.1
        s = np.array(np.random.default_rng().normal(mu, sigma, n), dtype='float32')
        s = tf.reshape(s,[n,1])
        return s

    def read_file_data(self,file,domain):
        if domain=='molecule':
            name = 1
        elif domain=='solvent':
            name = 2
        x_b, y_b, z_b, phi_b = list(), list(), list(), list()
        with open(os.path.join(self.molecule_path,file),'r') as f:
            for line in f:
                condition, x, y, z, phi = line.strip().split()

                if int(condition) == name:
                    x_b.append(float(x))
                    y_b.append(float(y))
                    z_b.append(float(z))
                    phi_b.append(float(phi))

        x_b = tf.constant(np.array(x_b, dtype=self.DTYPE)[:, None])
        y_b = tf.constant(np.array(y_b, dtype=self.DTYPE)[:, None])
        z_b = tf.constant(np.array(z_b, dtype=self.DTYPE)[:, None])
        phi_b = tf.constant(np.array(phi_b, dtype=self.DTYPE)[:, None])

        X = tf.concat([x_b, y_b, z_b], axis=1)

        return X,phi_b


    @staticmethod
    def get_X(X):
        return tf.split(X, num_or_size_splits=X.shape[1], axis=1)

    @staticmethod
    def stack_X(x,y,z):
        return tf.stack([tf.squeeze(x, axis=1), tf.squeeze(y, axis=1), tf.squeeze(z, axis=1)], axis=1)
    
    @classmethod
    def value_u_b(cls,x, y, z, value):
        n = x.shape[0]
        return tf.ones((n,1), dtype=cls.DTYPE)*value

    def save_data_plot(self,X_plot):
        path_files = os.path.join(self.result_path,'mesh')
        os.makedirs(path_files, exist_ok=True)
        logger = logging.getLogger(__name__)

        for subset_name, subset_data in X_plot.items():
            file_name = os.path.join(path_files,f'{subset_name}.csv')
            np.savetxt(file_name, subset_data, delimiter=',', header='X,Y,Z', comments='')
            logger.info(f'Subset {subset_name}: {len(subset_data)}')
     
