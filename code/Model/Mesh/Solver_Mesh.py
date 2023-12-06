import os
import numpy as np
import tensorflow as tf


class Solver_Mesh():

    DTYPE='float32'

    def __init__(self,name,molecule,path):

        self.name = name
        self.molecule = molecule
        self.main_path = path

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

            if type_b in ('R','D','K','N','P','Q'):    
                x,y,z = self.get_X(X)
                if value != None:
                    U = self.value_u_b(x, y, z, value=value)
                elif fun != None:
                    U = fun(x, y, z)
                else:
                    file = bl['file']
                    X,U = self.read_file_data(file)
                    #if type_b=='K' or type_b=='P':    # Sacar despues (solo pruebas)
                    #    U = U*self.add_noise(U)
                    #    print(type_b)
                self.solver_mesh_data[type_b] = self.create_Datasets(X,U)

            self.solver_mesh_names.add(type_b)
            if type_b in self.solver_mesh_N:
                self.solver_mesh_N[type_b] += len(X)
            else:
                self.solver_mesh_N[type_b] = len(X)
            
        del self.prior_data

    def add_noise(self,x):      # Sacar despues (solo pruebas)
        n = x.shape[0]
        mu, sigma = 1, 0.08 
        s = np.array(np.random.default_rng().normal(mu, sigma, n), dtype='float32')
        s = tf.reshape(s,[n,1])
        return s

    def read_file_data(self,file):
        x_b, y_b, z_b, phi_b = list(), list(), list(), list()

        path_files = os.path.join(self.main_path,'Model','Molecules')

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
        return dataset_X

    def create_Datasets(cls, X, Y):
        dataset_XY = tf.data.Dataset.from_tensor_slices((X, Y))
        return dataset_XY
    
    def value_u_b(self,x, y, z, value):
        n = x.shape[0]
        return tf.ones((n,1), dtype=self.DTYPE)*value
