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

    def fun_u_b(self,x, y, value):
        n = x.shape[0]
        return tf.ones((n,1), dtype=self.DTYPE)*value

    def fun_ux_b(self,x, y, value):
        n = x.shape[0]
        return tf.ones((n,1), dtype=self.DTYPE)*value


    def add_data(self,border,x1,x2,X):
        type_b = border['type']
        value = border['value']
        fun = border['fun']
        deriv = border['dr']
        if type_b == 'D':
            if fun == None:
                u_b = self.fun_u_b(x1, x2, value=value)
            else:
                u_b = fun(x1, x2)
            self.XD_data.append(X)
            self.UD_data.append(u_b)
        elif type_b == 'N':
            if fun == None:
                ux_b = self.fun_ux_b(x1, x2, value=value)
            else:
                ux_b = fun(x1, x2)
            self.XN_data.append(X)
            self.UN_data.append(ux_b)
            self.derN.append(deriv)
        elif type_b == 'I':
            self.XI_data.append(X)

    def create_mesh(self,borders):

        self.borders = borders

        self.XD_data = list()
        self.UD_data = list()
        self.XN_data = list()
        self.UN_data = list()
        self.derN = list()
        self.XI_data = list()
        self.derI = list()
 
        #estan a bases de radios (fijar un radio)
        for bl in self.borders.values():
            r_bl = tf.ones((self.N_b,1), dtype=self.DTYPE)*bl['r']
            theta_bl = tf.constant(np.linspace(0, 2*self.pi, self.N_b, dtype=self.DTYPE))
            theta_bl = tf.reshape(theta_bl,[theta_bl.shape[0],1])
            x_bl = r_bl*tf.cos(theta_bl)
            y_bl = r_bl*tf.sin(theta_bl)
            X_bl = tf.concat([x_bl, y_bl], axis=1)
            self.add_data(bl,x_bl,y_bl,X_bl)


        #crear dominio circular (cascaron para generalizar)
        xspace = np.linspace(self.lb[0], self.ub[0], self.N_r + 1, dtype=self.DTYPE)
        yspace = np.linspace(self.lb[1], self.ub[1], self.N_r + 1, dtype=self.DTYPE)
        ############################33#####
        Lx = list()
        Ly = list()
        r = xspace**2+yspace**2
        for i in range(len(xspace)):
            if r[i]<1:
                Lx.append(xspace)
                Ly.append(yspace)
        #################################33
        X, Y = np.meshgrid(np.array(Lx), np.array(Ly))
        self.X_r = tf.constant(np.vstack([X.flatten(),Y.flatten()]).T)
 
        self.D_bl = (x_bl,y_bl)
        self.D_r = (self.X_r[:,0],self.X_r[:,1])

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

    def stack_X(self,x,y):
        L = list()
        R = tf.stack([self.x[:,0], self.y[:,0]], axis=1)
        return R

    def plot_points(self):
        x1,y1 = self.D_bl
        rm,om = self.D_r
        fig = plt.figure(figsize=(9,6))
        xm = rm*np.cos(om)
        ym = rm*np.sin(om)
        plt.scatter(x1, y1, marker='X', vmin=-1, vmax=1)
        plt.scatter(xm, ym, c='r', marker='.', alpha=0.1)
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.title('Positions of collocation points and boundary data');

# modificar para dominio circular en cartesianas

########################################################################################################


def set_domain(X):
    x,y = X
    xmin = x[0]
    xmax = x[1]
    ymin = y[0]
    ymax = y[1]

    lb = tf.constant([xmin, ymin], dtype='float32')
    ub = tf.constant([xmax, ymax], dtype='float32')

    return (lb,ub)


if __name__=='__main__':
    domain = ([0.01,1],[0,2*np.pi])
    #PDE = PDE_Model()
    domain = set_domain(domain)

    lb = {'type':'D', 'value':0, 'fun':None, 'dr':None, 'r':1}

    borders = {'1':lb}

    mesh = Mesh(domain, N_b=20, N_r=1500)
    mesh.create_mesh(borders)

    print(mesh.get_X(mesh.data_mesh['residual']))