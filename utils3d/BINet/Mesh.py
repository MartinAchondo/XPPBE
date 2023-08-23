import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import bempp.api

class Mesh():
    
    def __init__(self, domain,
        N_b=50,
        N_r=200
        ):
        self.DTYPE='float64'
        self.pi = tf.constant(np.pi, dtype=self.DTYPE)
        self.N_b = N_b
        self.N_r = int(np.sqrt(N_r))
        self.lb = domain[0]
        self.ub = domain[1]


    def create_sphere(self):
        self.grid = bempp.api.shapes.sphere(h=0.1)
        self.X_r = tf.constant(np.transpose(np.array(self.grid.vertices)),  dtype=tf.float64)


    def create_mesh(self,borders,ins_domain):

        self.borders = borders
        self.ins_domain = ins_domain
        self.XD_data = list()
        self.UD_data = list()
        self.XN_data = list()
        self.UN_data = list()
        self.derN = list()
        self.XI_data = list()
        self.derI = list()

        self.BP = list()


        self.create_sphere()

        self.D_r = (self.X_r[:,0],self.X_r[:,1],self.X_r[:,2])

        self.data_mesh = {
            'residual': self.X_r,
            'dirichlet': (self.XD_data,self.UD_data),
            'neumann': (self.XN_data,self.UN_data,self.derN),
            'interface': (self.XI_data,self.derI)
        }

    def to_bempp(self,X):  #points en bempp
        R = X.numpy()
        return np.transpose(R)

    # u evaluated has shape (1,n) algo asi

    def get_X(self,X):
        R = list()
        for i in range(X.shape[1]):
            R.append(X[:,i:i+1])
        return R

    def stack_X(self,x,y,z):
        R = tf.stack([x[:,0], y[:,0], z[:,0]], axis=1)
        return R



########################################################################################################


def set_domain(X):
    x,y,z = X
    xmin = x[0]
    xmax = x[1]
    ymin = y[0]
    ymax = y[1]
    zmin = z[0]
    zmax = z[1]

    lb = tf.constant([xmin, ymin,zmin], dtype='float64')
    ub = tf.constant([xmax, ymax,zmax], dtype='float64')

    return (lb,ub)


if __name__=='__main__':
    domain = ([-1,1],[-1,1],[-1,1])
    #PDE = PDE_Model()
    domain = set_domain(domain)

    lb = {'type':'D', 'value':0, 'fun':None, 'dr':None, 'r':1}

    borders = {'1':lb}
    ins_domain = {'rmax': 1}

    mesh = Mesh(domain, N_b=20, N_r=1500)
    mesh.create_mesh(borders, ins_domain)

    mesh.plot_points_2d()


