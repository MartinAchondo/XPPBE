import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
import os
import pandas as pd
import json

logger = logging.getLogger(__name__)

class Postprocessing():

    def __init__(self,XPINN, save=False, directory=''):

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

        self.loss_last = [np.format_float_scientific(self.XPINN.losses['TL'][-1], unique=False, precision=3),
                          np.format_float_scientific(self.NN[0].losses['TL'][-1], unique=False, precision=3),
                          np.format_float_scientific(self.NN[1].losses['TL'][-1], unique=False, precision=3)]

        save_folders = ['plots_solution', 'plots_losses', 'plots_weights', 'plots_meshes', 'plots_model']
        
        for folder in save_folders:
            setattr(self, f'path_{folder}', folder)
            path = os.path.join(self.directory, folder)
            os.makedirs(path, exist_ok=True)

    def plot_loss_history(self, domain=1, plot_w=False, loss='all'):
        fig,ax = plt.subplots()
        domain -= 1
        c = {'TL': 'k','R':'r','D':'b','N':'g', 'K': 'gold','Q': 'c','Iu':'m','Id':'lime','E':'darkslategrey','G': 'salmon'}
        c2 = {'royalblue','springgreen','aqua', 'pink','yellowgreen','teal'}
        for i,NN in enumerate(self.NN):
            if i==domain:
                if not plot_w:
                    w = {'R': 1.0, 'D': 1.0, 'N': 1.0, 'K': 1.0, 'I': 1.0, 'E': 1.0, 'Q': 1.0, 'G': 1.0, 'Iu': 1.0, 'Id': 1.0}
                elif plot_w:
                    w = NN.w_hist
                
                if plot_w==False and (loss=='TL' or loss=='all'):
                    ax.semilogy(range(1,len(self.NN[domain].losses['TL'])+1), self.NN[domain].losses['TL'],'k-',label='Loss_NN')
                for t in NN.Mesh_names:
                    if t in loss or loss=='all':
                        ax.semilogy(range(1,len(NN.losses[t])+1), w[t]*np.array(NN.losses[t]),c[t],label=f'Loss_{t}')

        ax.legend()
        ax.set_xlabel('$n: iterations$')
        ax.set_ylabel(r'$\mathcal{L}: Losses$')
        ax.set_title(f'Loss History of NN{domain+1}, Iterations: {self.XPINN.N_iters}, Loss: {self.loss_last[domain+1]}')
        ax.grid()
        if self.save:
            path = f'loss_history_{domain+1}_loss{loss}.png' if not plot_w  else f'loss_history_{domain+1}_w.png' 
            path_save = os.path.join(self.directory,self.path_plots_losses,path)
            fig.savefig(path_save)
            logger.info(f'Loss history Plot saved: {path}')


    def plot_loss_validation_history(self, domain=1, loss='TL'):
        fig,ax = plt.subplots()
        domain -= 1
        for i,NN in enumerate(self.NN):
            if i==domain:               
                if loss=='TL' or loss=='all':
                    ax.semilogy(range(1,len(self.NN[domain].losses['vTL'])+1), self.NN[domain].losses['vTL'],'b-',label=f'{loss}_training')
                    ax.semilogy(range(1,len(self.NN[domain].validation_losses['TL'])+1), self.NN[domain].validation_losses['TL'],'r-',label=f'{loss}_validation')
                elif loss in NN.Mesh_names:
                    ax.semilogy(range(1,len(self.NN[domain].losses[loss])+1), self.NN[domain].losses[loss],'b-',label=f'{loss}_training')
                    ax.semilogy(range(1,len(self.NN[domain].validation_losses[loss])+1), self.NN[domain].validation_losses[loss],'r-',label=f'{loss}_validation')  

        ax.legend()
        ax.set_xlabel('$n: iterations$')
        ax.set_ylabel(r'$\mathcal{L}: Losses$')
        ax.set_title(f'Loss History of NN{domain+1}, Iterations: {self.XPINN.N_iters}')
        ax.grid()
        if self.save:
            path = f'loss_val_history_{domain+1}_loss{loss}.png' 
            path_save = os.path.join(self.directory,self.path_plots_losses,path)
            fig.savefig(path_save)
            logger.info(f'Loss history Plot saved: {path}')


    def plot_weights_history(self, domain=1):
        fig,ax = plt.subplots()
        domain -= 1
        c = {'TL': 'k','R':'r','D':'b','N':'g', 'K': 'gold','Q': 'c','Iu':'m','Id':'lime','E':'darkslategrey','G': 'salmon'}
        for i,NN in enumerate(self.NN):
            if i==domain:
                w = NN.w_hist
                for t in NN.Mesh_names:
                    ax.semilogy(range(1,len(w[t])+1), w[t], c[t],label=f'w_{t}')
                
        ax.legend()
        ax.set_xlabel('$n: iterations$')
        ax.set_ylabel('w: weights')
        ax.set_title(f'Weights History of NN{domain+1}, Iterations: {self.XPINN.N_iters}')
        ax.grid()

        if self.save:
            path = f'weights_history_{domain+1}.png'
            path_save = os.path.join(self.directory,self.path_plots_weights,path)
            fig.savefig(path_save)
            logger.info(f'Loss history Plot saved: {path}')


    def plot_G_solv_history(self):
        fig,ax = plt.subplots()
        ax.plot(np.array(list(self.XPINN.G_solv_hist.keys()), dtype=self.DTYPE), self.XPINN.G_solv_hist.values(),'k-',label='G_solv')
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
            path_save = os.path.join(self.directory,self.path_plots_solution,path)
            fig.savefig(path_save)


    def plot_collocation_points_3D(self):

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
        fig.write_html(os.path.join(self.directory,self.path_plots_meshes, 'collocation_points_plot_3d.html'))


    def plot_mesh_3D(self):

        vertices = self.mesh.mol_verts
        elements = self.mesh.mol_faces

        element_trace = go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=elements[:, 0],
            j=elements[:, 1],
            k=elements[:, 2],
            facecolor=['grey'] * len(elements), 
            opacity=0.97
        )
        edge_x = []
        edge_y = []
        edge_z = []

        for element in elements:
            for i in range(3):
                edge_x.extend([vertices[element[i % 3], 0], vertices[element[(i + 1) % 3], 0], None])
                edge_y.extend([vertices[element[i % 3], 1], vertices[element[(i + 1) % 3], 1], None])
                edge_z.extend([vertices[element[i % 3], 2], vertices[element[(i + 1) % 3], 2], None])

        edge_trace = go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode='lines',
            line=dict(color='blue', width=2),
        )

        fig = go.Figure(data=[element_trace,edge_trace])
        fig.update_layout(scene=dict(aspectmode='data'))

        fig.write_html(os.path.join(self.directory,self.path_plots_meshes,f'mesh_plot_3D.html'))


    def plot_interface_3D(self,variable='phi', values=None):
        
        vertices = self.mesh.mol_verts
        elements = self.mesh.mol_faces
         
        if variable == 'phi':
            values = self.PDE.get_phi_interface(*self.NN)
        elif variable == 'dphi':
            values,_,_ = self.PDE.get_dphi_interface(*self.NN)
        values = values.flatten()

        fig = go.Figure()
        fig.add_trace(go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                            i=elements[:, 0], j=elements[:, 1], k=elements[:, 2],
                            intensity=values, colorscale='RdBu_r'))

        fig.update_layout(scene=dict(aspectmode='data'))

        fig.write_html(os.path.join(self.directory, self.path_plots_solution, f'Interface_{variable}.html'))


    def plot_phi_line(self, N=100, x0=np.array([0,0,0]), theta=0, phi=np.pi/2):
        fig, ax = plt.subplots()
        
        r = np.linspace(-self.mesh.R_exterior,self.mesh.R_exterior,N)
        x = x0[0] + r * np.sin(phi) * np.cos(theta) + self.mesh.centroid[0]
        y = x0[1] + r * np.sin(phi) * np.sin(theta) + self.mesh.centroid[1]
        z = x0[2] + r * np.cos(phi) + self.mesh.centroid[2]
        points = np.stack((x.ravel(), y.ravel(), z.ravel()), axis=1)
        X_in,X_out,_ = self.get_interior_exterior(points)
        
        u_in = self.XPINN.solver1.model(tf.constant(X_in))
        u_out = self.XPINN.solver2.model(tf.constant(X_out))

        X_in -= self.mesh.centroid
        x_diff, y_diff, z_diff = x0[:, np.newaxis] - X_in.transpose()
        r_in = np.sqrt(x_diff**2 + y_diff**2 + z_diff**2)
        n = np.argmin(r_in)
        r_in[:n] = -r_in[:n]

        X_out -= self.mesh.centroid
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
            path_save = os.path.join(self.directory,self.path_plots_solution,path)
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
        x = x0[0] + T*u[0] + S*v[0] + self.mesh.centroid[0]
        y = x0[1] + T*u[1] + S*v[1] + self.mesh.centroid[1]
        z = x0[2] + T*u[2] + S*v[2] + self.mesh.centroid[2]
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
            path_save = os.path.join(self.directory,self.path_plots_solution,path)
            fig.savefig(path_save)


    def get_max_min(self,u1,u2):
        U = tf.concat([u1,u2], axis=0)
        vmax = tf.reduce_max(U)
        vmin = tf.reduce_min(U)
        return vmax,vmin

    def get_interior_exterior(self,points,R_exterior=None):
        if R_exterior==None:
            R_exterior = self.mesh.R_exterior
        interior_points_bool = self.mesh.mol_mesh.contains(points)
        interior_points = points[interior_points_bool]
        exterior_points_bool = ~interior_points_bool  
        exterior_points = points[exterior_points_bool]
        exterior_distances = np.linalg.norm(exterior_points-self.mesh.centroid, axis=1)
        exterior_points = exterior_points[exterior_distances <= R_exterior]
        bool_2 = np.linalg.norm(points-self.mesh.centroid, axis=1) <= R_exterior*exterior_points_bool
        bools = (interior_points_bool,bool_2)
        return interior_points,exterior_points, bools
    
    def L2_error_interface_continuity(self):
        verts = tf.constant(self.XPINN.mesh.mol_verts)
        s1,s2 = self.XPINN.solvers
        u1 = s1.model(verts).numpy()
        u2 = s2.model(verts).numpy()
        u_dif = (u1-u2)
        error = np.sqrt(np.sum(u_dif**2)/np.sum(u1**2))
        return error
    
    def save_values_file(self,save=True):
     
        max_iter = max(map(int,list(self.XPINN.G_solv_hist.keys())))
        Gsolv_value = self.XPINN.G_solv_hist[str(max_iter)]

        dict_pre = {
            'Gsolv_value': Gsolv_value,
            'L2_continuity_u': np.sqrt(self.XPINN.losses['Iu'][-1]),
            'L2_continuity_du': np.sqrt(self.XPINN.losses['Id'][-1]),
            'Loss_XPINN': self.loss_last[0],
            'Loss_NN1': self.loss_last[1],
            'Loss_NN2': self.loss_last[2]
        } 

        df_dict = {}
        for key, value in dict_pre.items():
            if key=='Gsolv_value':
                df_dict[key] = np.format_float_positional(float(value), unique=False, precision=3)
                continue
            df_dict[key] = '{:.3e}'.format(float(value))

        if not save:
            return df_dict
        
        path_save = os.path.join(self.directory,'results_values.json')
        with open(path_save, 'w') as json_file:
            json.dump(df_dict, json_file, indent=4)

    def save_model_summary(self):
        path_save = os.path.join(self.directory,self.path_plots_model,'models_summary.txt')
        with open(path_save, 'w') as f:
            print_func = lambda x: print(x, file=f)
            self.NN[0].model.summary(print_fn=print_func)
            print("\n\n", file=f)
            self.NN[1].model.summary(print_fn=print_func)

    def plot_architecture(self,domain=1):
        
        domain -= 1
        input_layer = tf.keras.layers.Input(shape=self.XPINN.solvers[domain].model.input_shape_N[1:], name='input')
        visual_model = tf.keras.models.Model(inputs=input_layer, outputs=self.XPINN.solvers[domain].model.call(input_layer))

        if self.save:
            path = f'model_architecture_{domain+1}.png'
            path_save = os.path.join(self.directory,self.path_plots_model,path)

            tf.keras.utils.plot_model(visual_model, to_file=path_save,
                                        show_shapes=True,
                                        show_dtype=False,
                                        show_layer_names=True,
                                        expand_nested=True,
                                        show_layer_activations=True,
                                        dpi = 150)
        
        self.XPINN.solvers[domain].model.build_Net()


class Born_Ion_Postprocessing(Postprocessing):

    def __init__(self,*kargs,**kwargs):
        super().__init__(*kargs,**kwargs)

    def plot_aprox_analytic(self, N=8000, x0=np.array([0,0,0]), theta=0, phi=np.pi/2, zoom=False, lims=None, lims_zoom=None):
        
        fig, ax = plt.subplots()
        r = np.linspace(-self.mesh.R_exterior,self.mesh.R_exterior,N)
        x = x0[0] + r * np.sin(phi) * np.cos(theta) + self.mesh.centroid[0]
        y = x0[1] + r * np.sin(phi) * np.sin(theta) + self.mesh.centroid[1]
        z = x0[2] + r * np.cos(phi) + self.mesh.centroid[2]
        points = np.stack((x.ravel(), y.ravel(), z.ravel()), axis=1)
        X_in,X_out,_ = self.get_interior_exterior(points)
        
        u_in = self.XPINN.solver1.model(tf.constant(X_in))
        u_out = self.XPINN.solver2.model(tf.constant(X_out))

        X_in -= self.mesh.centroid
        x_diff, y_diff, z_diff = x0[:, np.newaxis] - X_in.transpose()
        r_in = np.sqrt(x_diff**2 + y_diff**2 + z_diff**2)
        n = np.argmin(r_in)
        r_in[:n] = -r_in[:n]

        X_out -= self.mesh.centroid
        x_diff, y_diff, z_diff = x0[:, np.newaxis] - X_out.transpose()
        r_out = np.sqrt(x_diff**2 + y_diff**2 + z_diff**2)
        n = np.argmin(r_out)
        r_out_1 = -r_out[:n]
        r_out_2 = r_out[n:]

        ax.plot(r_in,u_in[:,0], label='Aprox', c='b')
        ax.plot(r_out_1,u_out[:n,0], c='b')
        ax.plot(r_out_2,u_out[n:,0], c='b')

        points -= self.mesh.centroid
        r = np.sqrt(points[:,0]**2 + points[:,1]**2 + points[:,2]**2)
        r = r[r <= self.XPINN.mesh.R_exterior*0.7]
        r = r[r > 0.04]
        u_an = self.XPINN.PDE.analytic_Born_Ion(r)
        n2 = np.argmin(r)
        r[:n2] = -r[:n2]
        ax.plot(r,u_an, c='r', label='Analytic', linestyle='--')

        if zoom:
            axin = ax.inset_axes([0.6, 0.02, 0.38, 0.38])
            axin.plot(r_in,u_in[:,0], c='b')
            axin.plot(r_out_1,u_out[:n,0], c='b')
            axin.plot(r_out_2,u_out[n:,0], c='b')
            axin.plot(r,u_an, c='r', linestyle='--')
            R = self.XPINN.mesh.R_mol
            axin.set_xlim(0.9*R,1.1*R)
            axin.set_ylim(-0.05, 0.05)
            if lims_zoom != None:
                limx,limy = lims_zoom
                if limx != None:
                    axin.set_xlim(limx)
                if limy != None:
                    axin.set_ylim(lims)
            axin.grid()
            ax.indicate_inset_zoom(axin)
        
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
        if lims != None:
            limx,limy = lims
            if limx != None:
                ax.set_xlim(limx)
            if limy != None:
                ax.set_ylim(lims)

        if self.save:
            path = 'analytic.png' if zoom==False else 'analytic_zoom.png'
            path_save = os.path.join(self.directory,self.path_plots_solution,path)
            fig.savefig(path_save)


    def plot_line_interface(self,N=100,plot='u'):

        labels = ['Inside', 'Outside']
        colr = ['r','b']
        i = 0

        rr = self.XPINN.mesh.R_mol
        
        r_bl = np.linspace(rr, rr, N + 1, dtype=self.DTYPE)
        phi_bl = np.linspace(np.pi/2, np.pi/2, N + 1, dtype=self.DTYPE)
        theta_bl = np.linspace(0, 2*np.pi, N + 1, dtype=self.DTYPE)
        
        x_bl = tf.constant(r_bl*np.sin(phi_bl)*np.cos(theta_bl)) + self.mesh.centroid[0]
        y_bl = tf.constant(r_bl*np.sin(phi_bl)*np.sin(theta_bl)) + self.mesh.centroid[1]
        z_bl = tf.constant(r_bl*np.cos(phi_bl)) + self.mesh.centroid[2]
        
        x_bl = tf.reshape(x_bl,[x_bl.shape[0],1])
        y_bl = tf.reshape(y_bl,[y_bl.shape[0],1])
        z_bl = tf.reshape(z_bl,[z_bl.shape[0],1])

        theta_bl = tf.constant(theta_bl)
        theta_bl = tf.reshape(theta_bl,[theta_bl.shape[0],1])

        XX_bl = tf.concat([x_bl, y_bl, z_bl], axis=1)

        fig, ax = plt.subplots() 

        for model,solver in zip(self.models,self.NN):
            if plot=='u':
                U = model(XX_bl)
                ax.plot(theta_bl[:,0],U[:,0], label=labels[i], c=colr[i])
            elif plot=='du':
                radial_vector = XX_bl - self.mesh.centroid
                magnitude = tf.norm(radial_vector, axis=1, keepdims=True)
                normal_vector = radial_vector / magnitude
                n_v = solver.mesh.get_X(normal_vector)
                X = solver.mesh.get_X(XX_bl)
                du = self.PDE.directional_gradient(solver.mesh,solver.model,X,n_v)
                ax.plot(theta_bl[:,0],du[:,0]*solver.PDE.epsilon, label=labels[i], c=colr[i])
            i += 1

        if plot=='u':
            U2 = self.XPINN.PDE.analytic_Born_Ion(rr)
            u2 = np.ones((N+1,1))*U2
            ax.plot(theta_bl, u2, c='g', label='Analytic', linestyle='--')
        elif plot=='du':
            dU2 = self.XPINN.PDE.analytic_Born_Ion_du(rr)
            du2 = np.ones((N+1,1))*dU2*self.NN[0].PDE.epsilon
            ax.plot(theta_bl, du2, c='g', label='Analytic', linestyle='--')
        
        ax.set_xlabel(r'$\beta$')
        ax.set_ylabel(r'$\phi_{\theta}$')

        text_l = r'$\phi_{\theta}$' if plot=='w' else r'd$\phi_{\theta}$'
        ax.set_title(f'Solution {text_l} of PDE, Iterations: {self.XPINN.N_iters}')

        ax.grid()
        ax.legend()

        if self.save:
            path = f'interface_line_{plot}.png'
            path_save = os.path.join(self.directory,self.path_plots_solution,path)
            fig.savefig(path_save)


    def L2_error_interface_analytic(self):
        verts = self.XPINN.mesh.mol_verts
        s1,s2 = self.XPINN.solvers
        u1 = s1.model(tf.constant(verts)).numpy()
        u2 = s2.model(tf.constant(verts)).numpy()
        u_mean = (u1+u2)/2

        r = np.sqrt(verts[:,0]**2 + verts[:,1]**2 + verts[:,2]**2)
        u_an = self.XPINN.PDE.analytic_Born_Ion(r)
        u_dif = u_mean-u_an
        error = np.sqrt(np.sum(u_dif**2)/np.sum(u_an**2))
        return error
    
    def save_values_file(self,save=True):

        L2_analytic = self.L2_error_interface_analytic()
        
        df_dict = super().save_values_file(save=False)
        df_dict['L2_analytic'] = '{:.3e}'.format(float(L2_analytic))

        if not save:
            return df_dict
        
        path_save = os.path.join(self.directory,'results_values.json')
        with open(path_save, 'w') as json_file:
            json.dump(df_dict, json_file, indent=4)


if __name__=='__main__':
    pass