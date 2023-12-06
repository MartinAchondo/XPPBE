import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
import os
import pandas as pd

logger = logging.getLogger(__name__)


class Postprocessing():

    def __init__(self,XPINN, save=False, directory='', last=False):

        self.DTYPE='float32'
        self.pi = tf.constant(np.pi, dtype=self.DTYPE)
        self.save = save
        self.directory = directory

        self.XPINN = XPINN
        self.NN = [XPINN.solver1,XPINN.solver2]
        self.models = [XPINN.solver1.model,XPINN.solver2.model]
        self.meshes = [XPINN.solver1.mesh,XPINN.solver2.mesh]
        self.mesh = XPINN.mesh
        self.PDE = XPINN.PDE

        self.loss_last = np.format_float_scientific(self.XPINN.loss_hist[-1], unique=False, precision=3)


    def plot_loss_history(self, flag=True, plot_w=False, ax=None):
        if not ax:
            fig = plt.figure(figsize=(7,5))
            ax = fig.add_subplot(111)
        ax.semilogy(range(len(self.XPINN.loss_hist)), self.XPINN.loss_hist,'k-',label='Loss')
        if flag: 
            iter = 1
            c = [['r','b','g','c','m','lime','darkslategrey'],['salmon','royalblue','springgreen','aqua', 'pink','yellowgreen','teal']]
            for NN in self.NN:
                if not plot_w:
                    w = {'R': 1.0, 'D': 1.0, 'N': 1.0, 'K': 1.0, 'I': 1.0, 'E': 1.0, 'Q': 1.0, 'G': 1.0}
                elif plot_w:
                    w = NN.w_hist
                meshes_names = NN.Mesh_names
                if 'R' in meshes_names:
                    ax.semilogy(range(len(NN.loss_r)), w['R']*np.array(NN.loss_r),c[iter-1][0],label=f'Loss_R_{iter}')
                if 'D' in meshes_names:
                    ax.semilogy(range(len(NN.loss_bD)), w['D']*np.array(NN.loss_bD),c[iter-1][1],label=f'Loss_D_{iter}')
                if 'N' in meshes_names:
                    ax.semilogy(range(len(NN.loss_bN)), w['N']*np.array(NN.loss_bN),c[iter-1][2],label=f'Loss_N_{iter}')
                if 'K' in meshes_names:
                    ax.semilogy(range(len(NN.loss_bK)), w['K']*np.array(NN.loss_bK),c[iter-1][3],label=f'Loss_K_{iter}')
                if 'Q' in meshes_names:
                    ax.semilogy(range(len(NN.loss_bQ)), w['Q']*np.array(NN.loss_bQ),'gold',label=f'Loss_Q_{iter}')
                if 'I' in meshes_names:
                    ax.semilogy(range(len(NN.loss_bI)), w['I']*np.array(NN.loss_bI),c[iter-1][4],label=f'Loss_I_{iter}')
                if 'E' in meshes_names:
                    ax.semilogy(range(len(self.XPINN.loss_exp)), w['E']*np.array(self.XPINN.loss_exp),c[iter-1][5],label=f'Loss_E_{iter}')
                if 'G' in meshes_names:
                    ax.semilogy(range(len(NN.loss_G)), w['G']*np.array(NN.loss_G),c[iter-1][6],label=f'Loss_G_{iter}')
                iter += 1      
        ax.legend()
        ax.set_xlabel('$n: iterations$')
        ax.set_ylabel(r'$\mathcal{L}: Losses$')
        text_l = r'$\phi_{\theta}$'
        ax.set_title(f'Solution {text_l} of PDE, Iterations: {self.XPINN.N_iters}, Loss: {self.loss_last}')
        ax.grid()

        if self.save:
            path = 'loss_history.png' if not plot_w  else 'loss_history_w.png' 
            path_save = os.path.join(self.directory,path)
            fig.savefig(path_save)
            logger.info(f'Loss history Plot saved: {path}')


    def plot_weights_history(self, ax=None):
        if not ax:
            fig = plt.figure(figsize=(7,5))
            ax = fig.add_subplot(111)

        iter = 1
        c = [['r','b','g','c','m','lime','darkslategrey'],['salmon','royalblue','springgreen','aqua', 'pink','yellowgreen','teal']]
        for NN in self.NN:
            if True:
                w = NN.w_hist
                meshes_names = NN.Mesh_names
                if 'R' in meshes_names:
                    ax.semilogy(range(len(w['R'])), w['R'], c[iter-1][0],label=f'w_R_{iter}')
                if 'D' in meshes_names:
                    ax.semilogy(range(len(w['D'])), w['D'], c[iter-1][1],label=f'w_D_{iter}')
                if 'N' in meshes_names:
                    ax.semilogy(range(len(w['N'])), w['N'], c[iter-1][2],label=f'w_N_{iter}')
                if 'K' in meshes_names:
                    ax.semilogy(range(len(w['K'])), w['K'], c[iter-1][3],label=f'w_K_{iter}')
                if 'Q' in meshes_names:
                    ax.semilogy(range(len(w['Q'])), w['Q'], 'gold',label=f'w_Q_{iter}')
                if 'I' in meshes_names:
                    ax.semilogy(range(len(w['I'])), w['I'], c[iter-1][4],label=f'w_I_{iter}')
                if 'E' in meshes_names:
                    ax.semilogy(range(len(w['E'])), w['E'],c[iter-1][5],label=f'w_E_{iter}')
                if 'G' in meshes_names:
                    ax.semilogy(range(len(w['G'])), w['G'],c[iter-1][6],label=f'w_G_{iter}')
            iter += 1
        ax.legend()
        ax.set_xlabel('$n: iterations$')
        ax.set_ylabel('w: weights')
        text_l = r'$\phi_{\theta}$'
        ax.set_title(f'Solution {text_l} of PDE, Iterations: {self.XPINN.N_iters}, Loss: {self.loss_last}')
        ax.grid()

        if self.save:
            path = 'weights_history.png'
            path_save = os.path.join(self.directory,path)
            fig.savefig(path_save)
            logger.info(f'Loss history Plot saved: {path}')


    def plot_G_solv_history(self, ax=None):
        if not ax:
            fig = plt.figure(figsize=(7,5))
            ax = fig.add_subplot(111)
        ax.plot(self.XPINN.G_solv_hist.keys(), self.XPINN.G_solv_hist.values(),'k-',label='G_solv')
        ax.legend()
        ax.set_xlabel('$n: iterations$')
        text_l = r'$G_{solv}$'
        ax.set_ylabel(text_l)
        max_iter = max(map(int,list(self.XPINN.G_solv_hist.keys())))
        Gsolv_value = np.format_float_positional(self.XPINN.G_solv_hist[str(max_iter)], unique=False, precision=2)
        ax.set_title(f'Solution {text_l} of PDE, Iterations: {max_iter}, G_solv: {Gsolv_value} kcal/kmol')
        ax.grid()

        if self.save:
            path = 'Gsolv_history.png'
            path_save = os.path.join(self.directory,path)
            fig.savefig(path_save)


    def plot_meshes_3D(self):

        color_dict = {
            'Charges': 'red',
            'Inner Domain': 'lightgreen',
            'Interface': 'purple',
            'Outer Domain': 'lightblue',
            'Outer Border': 'orange',
            'Experimental': 'cyan',
            'test': 'red'
        }

        subsets_directory = os.path.join(self.directory,'mesh')
        csv_files = [file for file in os.listdir(subsets_directory) if file.endswith('.csv')]
        fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])
        for i, csv_file in enumerate(csv_files):

            name = csv_file.replace('.csv','')
            data = pd.read_csv(os.path.join(subsets_directory, csv_file))
            trace = go.Scatter3d(
                x=data['X'],
                y=data['Y'],
                z=data['Z'],
                mode='markers',
                marker=dict(size=4, opacity=0.7, color=color_dict[name]),
                name=name
            )
            fig.add_trace(trace)

        fig.update_layout(title='Dominio 3D', scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
        fig.write_html(os.path.join(self.directory, 'mesh_plot_3d.html'))


    def plot_interface_3D(self,variable='phi', values=None):
        
        vertices = self.mesh.verts
        elements = self.mesh.faces
         
        if variable == 'phi':
            values = self.PDE.get_phi_interface(*self.NN).flatten()
        elif variable == 'dphi':
            values = self.PDE.get_dphi_interface(*self.NN).flatten()

        fig = go.Figure()
        fig.add_trace(go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                            i=elements[:, 0], j=elements[:, 1], k=elements[:, 2],
                            intensity=values, colorscale='RdBu_r'))

        fig.update_layout(scene=dict(aspectmode='data'))

        fig.write_html(os.path.join(self.directory, f'Interface_{variable}.html'))


    def plot_phi_line(self, N=100, x0=np.array([0,0,0]), theta=0, phi=0):

        r = np.linspace(-self.mesh.R_exterior,self.R_exterior,N)
        x = x0[0] + r * tf.sin(phi) * tf.cos(theta)
        y = x0[1] + r * tf.sin(phi) * tf.sin(theta)
        z = x0[2] + r * tf.cos(phi)
        X = tf.stack([x, y, z], axis=-1)


    def plot_phi_line(self, N=100, x0=np.array([0,0,0]), theta=0, phi=np.pi/2):
        fig, ax = plt.subplots()
        
        r = np.linspace(-self.mesh.R_exterior,self.mesh.R_exterior,N)
        x = x0[0] + r * np.sin(phi) * np.cos(theta)
        y = x0[1] + r * np.sin(phi) * np.sin(theta)
        z = x0[2] + r * np.cos(phi)
        points = np.stack((x.ravel(), y.ravel(), z.ravel()), axis=1)
        X_in,X_out,_ = self.get_interior_exterior(points)
        
        u_in = self.XPINN.solver1.model(tf.constant(X_in))
        u_out = self.XPINN.solver2.model(tf.constant(X_out))

        x_diff, y_diff, z_diff = x0[:, np.newaxis] - X_in.transpose()
        r_in = np.sqrt(x_diff**2 + y_diff**2 + z_diff**2)
        n = np.argmin(r_in)
        r_in[:n] = -r_in[:n]

        x_diff, y_diff, z_diff = x0[:, np.newaxis] - X_out.transpose()
        r_out = np.sqrt(x_diff**2 + y_diff**2 + z_diff**2)
        n = np.argmin(r_out)
        r_out_1 = -r_out[:n]
        r_out_2 = r_out[n:]

        ax.plot(r_in,u_in[:,0], label='Solute', c='r')
        ax.plot(r_out_1,u_out[:n,0], label='Solvent', c='b')
        ax.plot(r_out_2,u_out[n:,0], c='b')
        
        ax.set_xlabel('r')
        ax.set_ylabel(r'$\phi_{\theta}$')
        text_l = r'$\phi_{\theta}$'
        text_theta = r'$\theta$'
        text_phi = r'$\phi$'
        theta = np.format_float_positional(theta, unique=False, precision=2)
        phi = np.format_float_positional(phi, unique=False, precision=2)
        text_x0 = r'$x_0$'
        ax.set_title(f'Solution {text_l} of PDE, Iterations: {self.XPINN.N_iters};  ({text_x0}=[{x0[0]},{x0[1]},{x0[2]}] {text_theta}={theta}, {text_phi}={phi})')
        ax.grid()
        ax.legend()

        if self.save:
            path = 'solution.png'
            path_save = os.path.join(self.directory,path)
            fig.savefig(path_save)


    def plot_phi_contour(self, N=100, x0=np.array([0,0,0]), n=np.array([1,0,0])):
        
        fig,ax = plt.subplots()
        n = n/np.linalg.norm(n)
        if np.all(n == np.array([0,0,1])):
            u = np.array([1,0,0])
        else:
            u = np.array([np.copysign(n[2],n[0]),
                        np.copysign(n[2],n[1]),
                        -np.copysign(np.abs(n[0])+np.abs(n[1]),n[2])])
        u = u/np.linalg.norm(u)
        v = np.cross(n,u)
        v = v/np.linalg.norm(v)
    
        R_exterior = self.mesh.R_mol+self.mesh.dR_exterior/2
        t = np.linspace(-R_exterior,R_exterior,N)
        s = np.linspace(-R_exterior,R_exterior,N)
        T,S = np.meshgrid(t,s)
        x = x0[0] + T*u[0] + S*v[0]
        y = x0[1] + T*u[1] + S*v[1]
        z = x0[2] + T*u[2] + S*v[2]
        points = np.stack((x.ravel(), y.ravel(), z.ravel()), axis=1)
        X_in,X_out,bools = self.get_interior_exterior(points,R_exterior)

        u_in = self.XPINN.solver1.model(tf.constant(X_in))
        u_out = self.XPINN.solver2.model(tf.constant(X_out))

        vmax,vmin = self.get_max_min(u_in,u_out)
        s = ax.scatter(T.ravel()[bools[0]], S.ravel()[bools[0]], c=u_in[:,0],vmin=vmin,vmax=vmax)
        s = ax.scatter(T.ravel()[bools[1]], S.ravel()[bools[1]], c=u_out[:,0],vmin=vmin,vmax=vmax)

        fig.colorbar(s, ax=ax)
        ax.set_xlabel(r'$u$')
        ax.set_ylabel(r'$v$')
        text_l = r'$\phi_{\theta}$'
        text_n = r'$n$'
        text_x0 = r'$x_0$'
        ax.set_title(f'Solution {text_l} of PDE, Iterations: {self.XPINN.N_iters};  ({text_x0}=[{x0[0]},{x0[1]},{x0[2]}] {text_n}=[{np.format_float_positional(n[0], unique=False, precision=2)},{np.format_float_positional(n[1], unique=False, precision=2)},{np.format_float_positional(n[2], unique=False, precision=2)}])')
        ax.grid()
        ax.legend()

        if self.save:
            path = 'contour.png'
            path_save = os.path.join(self.directory,path)
            fig.savefig(path_save)


    def get_max_min(self,u1,u2):
        U = tf.concat([u1,u2], axis=0)
        vmax = tf.reduce_max(U)
        vmin = tf.reduce_min(U)
        return vmax,vmin

    def get_interior_exterior(self,points,R_exterior=None):
        if R_exterior==None:
            R_exterior = self.mesh.R_exterior
        interior_points_bool = self.mesh.mesh.contains(points)
        interior_points = points[interior_points_bool]
        exterior_points_bool = ~interior_points_bool  
        exterior_points = points[exterior_points_bool]
        exterior_distances = np.linalg.norm(exterior_points, axis=1)
        exterior_points = exterior_points[exterior_distances <= R_exterior]
        bool_2 = np.linalg.norm(points, axis=1) <= R_exterior*exterior_points_bool
        bools = (interior_points_bool,bool_2)
        return interior_points,exterior_points, bools
    

    def plot_architecture(self,domain=1):
        
        input_layer = tf.keras.layers.Input(shape=self.XPINN.solvers[domain].model.input_shape_N[1:], name='input')
        visual_model = tf.keras.models.Model(inputs=input_layer, outputs=self.XPINN.solvers[domain].model.call(input_layer))

        if self.save:
            path = f'model_architecture_{domain}.png'
            path_save = os.path.join(self.directory,path)

            tf.keras.utils.plot_model(visual_model, to_file=path_save,
                                        show_shapes=True,
                                        show_dtype=False,
                                        show_layer_names=True,
                                        expand_nested=True,
                                        show_layer_activations=True,
                                        dpi = 150)
        
        self.XPINN.solvers[domain].model.build_Net()


class Born_Ion_Post(Postprocessing):

    def __init_subclass__(self,*kargs,**kwargs):
        super().__init_(*kargs,**kwargs)


    


if __name__=='__main__':
    pass