import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class Mesh():
    
    def __init__(self, domain,
        N_b=50,
        N_r=200
        ):
        self.DTYPE='float32'
        self.pi = tf.constant(np.pi, dtype=self.DTYPE)
        self.N_b = N_b
        self.N_r = int(np.sqrt(N_r))
        self.lb = domain[0]
        self.ub = domain[1]

    def fun_u_b(self,x, y, z, value):
        n = x.shape[0]
        return tf.ones((n,1), dtype=self.DTYPE)*value

    def fun_ux_b(self,x, y, z, value):
        n = x.shape[0]
        return tf.ones((n,1), dtype=self.DTYPE)*value


    def add_data(self,border,x1,x2,x3,X):
        type_b = border['type']
        value = border['value']
        fun = border['fun']
        deriv = border['dr']
        if type_b == 'D':
            if fun == None:
                u_b = self.fun_u_b(x1, x2, x3, value=value)
            else:
                u_b = fun(x1, x2, x3)
            self.XD_data.append(X)
            self.UD_data.append(u_b)
        elif type_b == 'N':
            if fun == None:
                ux_b = self.fun_ux_b(x1, x2, x3, value=value)
            else:
                ux_b = fun(x1, x2, x3)
            self.XN_data.append(X)
            self.UN_data.append(ux_b)
            self.derN.append(deriv)
        elif type_b == 'I':
            self.XI_data.append(X)


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
 
        x_bl = 0
        y_bl = 0
        z_bl = 0

        self.BP = list()

        #estan a bases de radios (fijar un radio)
        for bl in self.borders.values():
            rr = bl['r']

            r_bl = np.linspace(rr, rr, self.N_r + 1, dtype=self.DTYPE)
            theta_bl = np.linspace(0, self.pi, self.N_r + 1, dtype=self.DTYPE)
            phi_bl = np.linspace(0, 2*self.pi, self.N_r + 1, dtype=self.DTYPE)
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
            
            XX_bl = tf.concat([x_bl, y_bl, z_bl], axis=1)
            self.add_data(bl,x_bl,y_bl,z_bl,XX_bl)
            self.BP.append((x_bl,y_bl,z_bl))


        #crear dominio circular (cascaron para generalizar)
        xspace = np.linspace(self.lb[0], self.ub[0], self.N_r + 1, dtype=self.DTYPE)
        yspace = np.linspace(self.lb[1], self.ub[1], self.N_r + 1, dtype=self.DTYPE)
        zspace = np.linspace(self.lb[2], self.ub[2], self.N_r + 1, dtype=self.DTYPE)
        X, Y, Z = np.meshgrid(xspace, yspace, zspace)
        
        if 'rmin' not in ins_domain:
            ins_domain['rmin'] = -0.1

        r = np.sqrt(X**2 + Y**2 + Z**2)
        inside1 = r < ins_domain['rmax']
        X1 = X[inside1]
        Y1 = Y[inside1]
        Z1 = Z[inside1]
        r = np.sqrt(X1**2 + Y1**2 + Z1**2)
        inside = r > ins_domain['rmin']

        self.X_r = tf.constant(np.vstack([X1[inside].flatten(),Y1[inside].flatten(), Z1[inside].flatten()]).T)


        self.D_r = (self.X_r[:,0],self.X_r[:,1],self.X_r[:,2])

        self.data_mesh = {
            'residual': self.X_r,
            'dirichlet': (self.XD_data,self.UD_data),
            'neumann': (self.XN_data,self.UN_data,self.derN),
            'interface': (self.XI_data,self.derI)
        }

    def get_X(self,X):
        R = list()
        for i in range(X.shape[1]):
            R.append(X[:,i:i+1])
        return R

    def stack_X(self,x,y,z):
        R = tf.stack([x[:,0], y[:,0], z[:,0]], axis=1)
        return R

    def plot_points_2d(self):
        xm,ym,zm = self.D_r
        fig = plt.figure(figsize=(9,6))
        for x,y,z in self.BP:
            plane = np.abs(z)<10**-4
            plt.scatter(x[plane], y[plane], marker='X')
        plane = np.abs(zm) <10**-4
        plt.scatter(xm[plane], ym[plane], c='r', marker='.', alpha=0.1)
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.title('Positions of collocation points and boundary data')
        plt.show();

    def plot_points_3d(self, alpha1=25, alpha2=25):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        xm,ym,zm = self.D_r
        fig = plt.figure(figsize=(9,6))
        for x,y,z in self.BP:
            ax.scatter(x, y, z, marker='X')
        ax.scatter(xm, ym, zm, c='r', marker='.', alpha=0.1)
        ax.view_init(alpha1,alpha2)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        plt.show()
        #ax.title('Positions of collocation points and boundary data');

# modificar para dominio circular en cartesianas

########################################################################################################


def set_domain(X):
    x,y,z = X
    xmin = x[0]
    xmax = x[1]
    ymin = y[0]
    ymax = y[1]
    zmin = z[0]
    zmax = z[1]

    lb = tf.constant([xmin, ymin,zmin], dtype='float32')
    ub = tf.constant([xmax, ymax,zmax], dtype='float32')

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


