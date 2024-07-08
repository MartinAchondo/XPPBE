import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import json
import pandas as pd
import logging

matplotlib_logger = logging.getLogger('matplotlib')
matplotlib_logger.setLevel(logging.WARNING)

plt.rcParams['figure.max_open_warning'] = 50
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150

class Postprocessing():

    def __init__(self,PINN, save=False, directory='', solution_utils=False):

        if not solution_utils:
            self.DTYPE='float32'
            self.pi = tf.constant(np.pi, dtype=self.DTYPE)
            self.save = save
            self.directory = directory

            self.PINN = PINN
            self.model = PINN.model
            self.mesh = PINN.mesh
            self.PDE = PINN.PDE
            self.PDE.pqr_path = self.mesh.path_pqr
            self.to_V = self.PDE.to_V

            self.loss_last = [np.format_float_scientific(self.PINN.losses['TL'][-1], unique=False, precision=3),
                            np.format_float_scientific(self.PINN.losses['TL1'][-1], unique=False, precision=3),
                            np.format_float_scientific(self.PINN.losses['TL2'][-1], unique=False, precision=3)]

            save_folders = ['plots_solution', 'plots_losses', 'plots_weights', 'plots_meshes', 'plots_model']
            
            for folder in save_folders:
                setattr(self, f'path_{folder}', folder)
                path = os.path.join(self.directory, folder)
                os.makedirs(path, exist_ok=True)

    def get_phi(self,*args,**kwargs):
        return self.PDE.get_phi(*args,**kwargs)*self.to_V

    def get_dphi(self,*args,**kwargs):
        return tuple(dphi*self.to_V for dphi in self.PDE.get_dphi(*args,**kwargs))

    def phi_known(self,*args,**kwargs):
        return self.PDE.phi_known(*args,**kwargs)*self.to_V

    def get_phi_interface_verts(self,*args,**kwargs):
        return tuple(phi*self.to_V for phi in self.PDE.get_phi_interface_verts(*args,**kwargs))

    def get_dphi_interface_verts(self,*args,**kwargs):
        return tuple(dphi*self.to_V for dphi in self.PDE.get_dphi_interface_verts(*args,**kwargs))

    def get_solvation_energy(self,*args,**kwargs):
        return self.PDE.get_solvation_energy(self.model)
    
    def get_phi_ens(self,*args,**kwargs):
        ((X,X_values),flag) = self.mesh.domain_mesh_data['E2']
        q_L,_ = zip(*X_values)
        return self.PDE.get_phi_ens(self.model,(X,flag),q_L, **kwargs)
    
    def run_all(self,plot_mesh,known_method):
        
        self.plot_loss_history(domain=1);
        self.plot_loss_history(domain=2);
        
        self.plot_loss_validation_history(domain=1,loss='TL');
        self.plot_loss_validation_history(domain=2,loss='TL');

        if plot_mesh:
            self.plot_collocation_points_3D();
            self.plot_vol_mesh_3D();
            self.plot_surface_mesh_3D();
            self.plot_mesh_3D('R1');
            self.plot_mesh_3D('R2');
            self.plot_mesh_3D('I');
            self.plot_mesh_3D('D2');
            self.plot_surface_mesh_normals(plot='vertices');
            self.plot_surface_mesh_normals(plot='faces');

        self.plot_G_solv_history();
        self.plot_phi_line();
        self.plot_phi_line(value='react');
        self.plot_phi_contour();
        self.plot_phi_contour(value='react');
        self.plot_interface_3D(variable='phi');
        self.plot_interface_3D(variable='dphi');
        self.save_values_file();

        if not known_method is None:

            if known_method == 'analytic_Born_Ion':
                self.plot_G_solv_history('analytic_Born_Ion');
                self.save_values_file(L2_err_method='analytic_Born_Ion');
                self.plot_aprox_analytic();
                self.plot_aprox_analytic(value='react');
                self.plot_aprox_analytic(zoom=True);
                self.plot_aprox_analytic(zoom=True, value='react');
                self.plot_line_interface();
                self.plot_line_interface(value='react');
                self.plot_line_interface(plot='du');
            
            else:
                self.plot_G_solv_history(known_method);
                self.save_values_file(L2_err_method=known_method);
                self.plot_phi_line_aprox_known(known_method, value='react',theta=0, phi=np.pi/2)
                self.plot_phi_line_aprox_known(known_method, value='react',theta=np.pi/2, phi=np.pi/2)
                self.plot_phi_line_aprox_known(known_method, value='react', theta=np.pi/2, phi=np.pi)
                self.plot_interface_3D_known(known_method)
                self.plot_interface_error(known_method, type_e='relative',scale='log')
                self.plot_interface_error(known_method, type_e='absolute',scale='linear')
                
        self.save_model_summary();
        self.plot_architecture(domain=1);
        self.plot_architecture(domain=2);

    

    def plot_loss_history(self, domain=1, plot_w=False, loss='all'):
        fig,ax = plt.subplots()
        c = {'TL': 'k','R':'r','D':'b','N':'g', 'K': 'gold','Q': 'c','Iu':'m','Id':'lime', 'Ir': 'aqua', 'E':'darkslategrey','G': 'salmon','IB1':'lime','IB2': 'aqua'}
        c2 = {'royalblue','springgreen','aqua', 'pink','yellowgreen','teal'}
        for i in ['1','2']:
            if int(i)==domain:
                if not plot_w:
                    w = {'R'+i: 1.0, 'D'+i: 1.0, 'N'+i: 1.0, 'K'+i: 1.0, 'E'+i: 1.0, 'Q'+i: 1.0, 'G': 1.0, 'Iu': 1.0, 'Id': 1.0, 'Ir': 1.0,'IB1':1.0,'IB2':1.0}
                elif plot_w:
                    w = self.PINN.w_hist
                
                if plot_w==False and (loss=='TL' or loss=='all'):
                    ax.semilogy(range(1,len(self.PINN.losses['TL'+i])+1), self.PINN.losses['TL'+i],'k-',label='TL'+i)
                for t in self.PINN.losses_names_list[int(i)-1]:
                    t2 = t if t in ('Iu','Id','Ir','G','IB1','IB2') else t[0]
                    if (t2 in loss or loss=='all') and not t in 'TL' and t in self.mesh.domain_mesh_names:
                        cx = c[t] if t in ('Iu','Id','Ir','G','IB1','IB2') else c[t[0]]
                        ax.semilogy(range(1,len(self.PINN.losses[t])+1), w[t]*self.PINN.losses[t],cx,label=f'{t}')

        ax.legend()
        n_label = r'$n$'
        ax.set_xlabel(f'Iterations', fontsize='11')
        loss_label = r'$\mathcal{L}$'
        ax.set_ylabel(f'Losses {loss_label}', fontsize='11')
        # if loss=='TL' or loss=='all':
        #     ax.set_title(f'Loss History of NN{domain}, Loss: {self.loss_last[domain]}')
        # else:
        #     ax.set_title(f'Loss History of NN{domain}')
        ax.grid()
        if self.save:
            path = f'loss_history_{domain}_loss{loss}.png' if not plot_w  else f'loss_history_{domain}_w.png' 
            path_save = os.path.join(self.directory,self.path_plots_losses,path)
            fig.savefig(path_save, bbox_inches='tight')
        return fig,ax

    def plot_loss_validation_history(self, domain=1, loss='TL'):
        fig,ax = plt.subplots()
        for i in ['1','2']:
            if int(i)==domain:               
                if loss=='TL' or loss=='all':
                    ax.semilogy(range(1,len(self.PINN.losses['vTL'+i])+1), self.PINN.losses['vTL'+i],'b-',label=f'Training {i}')
                    ax.semilogy(range(1,len(self.PINN.validation_losses['TL'+i])+1), self.PINN.validation_losses['TL'+i],'r-',label=f'Validation {i}')
                else:
                    t = loss if loss in ('Iu','Id','Ir','G','IB1','IB2') else loss+i
                    if t in self.mesh.domain_mesh_names :
                        ax.semilogy(range(1,len(self.PINN.losses[t])+1), self.PINN.losses[t],'b-',label=f'{loss} training')
                        ax.semilogy(range(1,len(self.PINN.validation_losses[t])+1), self.PINN.validation_losses[t],'r-',label=f'{loss} validation')  

        ax.legend()
        n_label = r'$n$'
        ax.set_xlabel(f'Iterations', fontsize='11')
        loss_label = r'$\mathcal{L}$'
        ax.set_ylabel(f'Losses {loss_label}', fontsize='11')
        # if loss=='TL' or loss=='all':
        #     ax.set_title(f'Loss History of NN{domain}, Loss: {self.loss_last[domain]}')
        # else:
        #     ax.set_title(f'Loss History of NN{domain}')
        ax.grid()
        if self.save:
            path = f'loss_val_history_{domain}_loss{loss}.png' 
            path_save = os.path.join(self.directory,self.path_plots_losses,path)
            fig.savefig(path_save, bbox_inches='tight')
        return fig,ax


    def plot_weights_history(self, domain=1):
        fig,ax = plt.subplots()
        c = {'TL': 'k','R':'r','D':'b','N':'g', 'K': 'gold','Q': 'c','Iu':'m','Id':'lime', 'Ir': 'aqua', 'E':'darkslategrey','G': 'salmon','IB1':'lime','IB2': 'aqua'}
        for i in ['1','2']:
            if int(i)==domain:
                w = self.PINN.w_hist
                for t in self.PINN.losses_names_list[int(i)-1]:
                    if t in self.mesh.domain_mesh_names:
                        cx = c[t] if t in ('Iu','Id','Ir','G','IB1','IB2') else c[t[0]]
                        ax.semilogy(range(1,len(w[t])+1), w[t], cx,label=f'{t}')
                
        ax.legend()
        n_label = r'$n$'
        ax.set_xlabel(f'Iterations', fontsize='11')
        w_label = r'$w$'
        ax.set_ylabel(f'Weights', fontsize='11')
        # ax.set_title(f'Weights History of NN{domain}')
        ax.grid()
        if self.save:
            path = f'weights_history_{domain}.png'
            path_save = os.path.join(self.directory,self.path_plots_weights,path)
            fig.savefig(path_save, bbox_inches='tight')
        return fig,ax


    def plot_G_solv_history(self, known_method=None):
        fig,ax = plt.subplots()
        ax.plot(np.array(list(self.PINN.G_solv_hist.keys()), dtype=self.DTYPE), self.PINN.G_solv_hist.values(),'k-',label='PINN')
        if not known_method is None:
            G_known = self.PDE.solvation_energy_phi_qs(self.to_V**-1*self.phi_known(known_method,'react',tf.constant(self.PDE.x_qs, dtype=self.DTYPE),'molecule'))
            G_known = np.ones(len(self.PINN.G_solv_hist))*G_known
            label = known_method.replace('_',' ') if 'Born' not in known_method else 'Analytic'
            if label=='PBJ':
                label='BEM'
            elif label=='APBS':
                label='FDM'
            ax.plot(np.array(list(self.PINN.G_solv_hist.keys()), dtype=self.DTYPE), G_known,'r--',label=f'{label}')
        ax.legend()
        n_label = r'$n$'
        ax.set_xlabel(f'Iterations', fontsize='11')
        text_l = r'$\Delta G_{solv}$'
        ax.set_ylabel(f'{text_l} [kcal/mol]', fontsize='11')
        max_iter = max(map(int,list(self.PINN.G_solv_hist.keys())))
        #Gsolv_value = np.format_float_positional(self.PINN.G_solv_hist[str(max_iter)], unique=False, precision=2)
        # ax.set_title(f'Solution {text_l} of PBE')
        ax.grid()

        if self.save:
            path = 'Gsolv_history.png' if known_method is None else f'Gsolv_history_{known_method}.png'
            path_save = os.path.join(self.directory,self.path_plots_solution,path)
            fig.savefig(path_save, bbox_inches='tight')
        return fig,ax


    def plot_phi_line(self, N=8000, x0=np.array([0,0,0]), theta=0, phi=np.pi/2, value ='phi'):
        fig, ax = plt.subplots()
        
        X_in,X_out,r_in,r_out = self.get_points_line(N,x0,theta,phi)
        
        u_in = self.get_phi(tf.constant(X_in, dtype=self.DTYPE),'molecule',self.model,  value)[:,0]
        u_out = self.get_phi(tf.constant(X_out, dtype=self.DTYPE),'solvent',self.model,  value)[:,0]

        ax.plot(r_in,u_in[:], label='Solute', c='r')
        ax.plot(r_out[r_out<0],u_out[r_out<0], label='Solvent', c='b')
        ax.plot(r_out[r_out>0],u_out[r_out>0], c='b')
        
        text_A = r'$\AA$'
        text_r = r'$r$'
        ax.set_xlabel(f'{text_r} [{text_A}]', fontsize='11')
        text_l = r'$\phi$' if value=='phi' else r'$\phi_{react}$'
        ax.set_ylabel(f'{text_l} [V]', fontsize='11')
        text_theta = r'$\theta$'
        text_phi = r'$\phi$'
        theta = np.format_float_positional(theta, unique=False, precision=2)
        phi = np.format_float_positional(phi, unique=False, precision=2)
        text_x0 = r'$x_0$'
        # ax.set_title(f'Solution {text_l} of PBE;  ({text_x0}=[{x0[0]},{x0[1]},{x0[2]}] {text_theta}={theta}, {text_phi}={phi})')
        ax.grid()
        ax.legend()

        if self.save:
            path = f'solution_{value}_th{theta}_ph{phi}.png'
            path_save = os.path.join(self.directory,self.path_plots_solution,path)
            fig.savefig(path_save, bbox_inches='tight')
        return fig,ax


    def plot_phi_line_aprox_known(self, method, N=300, x0=np.array([0,0,0]), theta=0, phi=np.pi/2, value ='phi'):
        fig, ax = plt.subplots()
        
        X_in,X_out,r_in,r_out = self.get_points_line(N,x0,theta,phi)
        
        u_in = self.get_phi(tf.constant(X_in, dtype=self.DTYPE),'molecule',self.model,  value)[:,0]
        u_out = self.get_phi(tf.constant(X_out, dtype=self.DTYPE),'solvent',self.model,  value)[:,0]

        u_in_an = self.phi_known(method,value,tf.constant(X_in, dtype=self.DTYPE),'molecule',self.mesh.R_max_dist)
        u_out_an = self.phi_known(method,value,tf.constant(X_out, dtype=self.DTYPE),'solvent',self.mesh.R_max_dist)

        ax.plot(r_in,u_in[:], c='b')
        ax.plot(r_out[r_out<0],u_out[r_out<0], label='PINN', c='b')
        ax.plot(r_out[r_out>0],u_out[r_out>0], c='b')

        label = method.replace('_',' ')
        if label=='PBJ':
            label='BEM'
        elif label=='APBS':
            label='FDM'
        ax.plot(r_in,u_in_an[:], c='r', linestyle='--')
        ax.plot(r_out[r_out<0],u_out_an[r_out<0], label=label, c='r', linestyle='--')
        ax.plot(r_out[r_out>0],u_out_an[r_out>0], c='r', linestyle='--')
        
        text_A = r'$\AA$'
        text_r = r'$r$'
        ax.set_xlabel(f'{text_r} [{text_A}]', fontsize='11')
        text_l = r'$\phi$' if value=='phi' else r'$\phi_{react}$'
        ax.set_ylabel(f'{text_l} [V]', fontsize='11')
        text_theta = r'$\theta$'
        text_phi = r'$\phi$'
        theta = np.format_float_positional(theta, unique=False, precision=2)
        phi = np.format_float_positional(phi, unique=False, precision=2)
        text_x0 = r'$x_0$'
        # ax.set_title(f'Solution {text_l} of PBE;  ({text_x0}=[{x0[0]},{x0[1]},{x0[2]}] {text_theta}={theta}, {text_phi}={phi})')
        ax.grid()
        ax.legend()

        if self.save:
            path = f'solution_{value}_{method}_th{theta}_ph{phi}.png'
            path_save = os.path.join(self.directory,self.path_plots_solution,path)
            fig.savefig(path_save, bbox_inches='tight')
        return fig,ax


    def plot_phis_line(self,u_list, X, labels=None, domain='all', value='react'):
        fig, ax = plt.subplots()
        c = ['r','b','g','m']
        if labels == None:
            labels = ['1','2','3','4','5']
        r_in,r_out= X

        for i,(u_in,u_out) in enumerate(u_list):
            if domain == 'all':
                ax.plot(r_in,u_in[:], c=c[i])
            ax.plot(r_out[r_out<0],u_out[r_out<0], c=c[i], label=labels[i])
            ax.plot(r_out[r_out>0],u_out[r_out>0], c=c[i])

        text_A = r'$\AA$'
        text_r = r'$r$'
        ax.set_xlabel(f'{text_r} [{text_A}]', fontsize='11')
        text_l = r'$\phi$' if value=='phi' else r'$\phi_{react}$'
        ax.set_ylabel(f'{text_l} [V]', fontsize='11')
        # ax.set_title(f'Solution {text_l} of PBE')
        ax.grid()
        ax.legend()
        return fig,ax

    def plot_phi_contour(self, N=100, x0=np.array([0,0,0]), n=np.array([1,0,0]), value='phi'):
        
        fig,ax = plt.subplots()

        X_in,X_out,bools,vectors = self.get_points_plane(N, x0, n)
        T,S = vectors

        u_in = self.get_phi(tf.constant(X_in, dtype=self.DTYPE),'molecule',self.model, value)[:,0]
        u_out = self.get_phi(tf.constant(X_out, dtype=self.DTYPE),'solvent',self.model, value)[:,0]

        vmax,vmin = self.get_max_min(u_in,u_out)
        s = ax.scatter(T.ravel()[bools[0]], S.ravel()[bools[0]], c=u_in[:],vmin=vmin,vmax=vmax)
        s = ax.scatter(T.ravel()[bools[1]], S.ravel()[bools[1]], c=u_out[:],vmin=vmin,vmax=vmax)

        text_A = r'$\AA$'
        text_u = r'$u$'
        text_v = r'$v$'
        ax.set_xlabel(f'{text_u} [{text_A}]', fontsize='11')
        ax.set_ylabel(f'{text_v} [{text_A}]', fontsize='11')
        text_l = r'$\phi$' if value=='phi' else r'$\phi_{react}$'
        cbar = fig.colorbar(s, ax=ax)
        cbar.set_label(f'{text_l} [V]', rotation=270)
        text_n = r'$n$'
        text_x0 = r'$x_0$'
        # ax.set_title(f'Solution {text_l} of PBE;  ({text_x0}=[{x0[0]},{x0[1]},{x0[2]}] {text_n}=[{np.format_float_positional(n[0], unique=False, precision=1)},{np.format_float_positional(n[1], unique=False, precision=1)},{np.format_float_positional(n[2], unique=False, precision=1)}])')
        ax.grid()

        if self.save:
            path = f'contour_{value}.png'
            path_save = os.path.join(self.directory,self.path_plots_solution,path)
            fig.savefig(path_save, bbox_inches='tight')
        return fig,ax


    def plot_interface_3D(self,variable='phi', value='phi', domain='interface', jupyter=False, ext='html'):
        
        vertices = self.mesh.mol_verts.astype(np.float32)
        elements = self.mesh.mol_faces.astype(np.float32)
         
        if variable == 'phi':
            values,values_1,values_2 = self.get_phi_interface_verts(self.PINN.model,value=value)
            text_l = r'phi' if value == 'phi' else 'ϕ react'
        elif variable == 'dphi':
            values,values_1,values_2 = self.get_dphi_interface_verts(self.PINN.model)
            text_l = r'dphi' if value == 'phi' else '∂ϕ react'

        if domain =='interface':
            values = values.numpy().flatten()
        elif domain =='molecule':
            values = values_1.numpy().flatten()
        elif domain =='solvent':
            values = values_2.numpy().flatten()

        fig = go.Figure()
        fig.add_trace(go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                            i=elements[:, 0], j=elements[:, 1], k=elements[:, 2],
                            intensity=values, colorscale='RdBu_r',
                            colorbar=dict(title=f'{text_l} [V]')))

        fig.update_layout(scene=dict(aspectmode='data', xaxis_title='X [Å]', yaxis_title='Y [Å]', zaxis_title='Z [Å]'),margin=dict(l=30, r=40, t=20, b=20),font_family="Times New Roman")

        path_save = os.path.join(self.directory, self.path_plots_solution, f'Interface_{variable}_{value}_{domain}')
        if self.save:
            if ext=='html':
                fig.write_html(path_save+'.html')
            elif ext=='png':
                fig.write_image(path_save+'.png', scale=3)
        if jupyter:
            fig.show()
        return fig


    def plot_interface_3D_known(self, method, cmin=None,cmax=None, jupyter=False, ext='html'):
        
        vertices = self.mesh.mol_verts.astype(np.float32)
        elements = self.mesh.mol_faces.astype(np.float32)
        phi_known = self.phi_known(method,'react', vertices,'solvent')

        fig = self.plot_interface_3D_known_by(phi_known, vertices, elements, jupyter=False)
        path_save = os.path.join(self.directory, self.path_plots_solution, f'Interface_{method}')
        if self.save:
            if ext=='html':
                fig.write_html(path_save+'.html')
            elif ext=='png':
                fig.write_image(path_save+'.png', scale=3)
        if jupyter:
            fig.show()
        return fig

    @staticmethod
    def plot_interface_3D_known_by(phi_known, vertices, elements, cmin=None,cmax=None, jupyter=True):
        cmin = np.min(phi_known) if cmin is None else cmin
        cmax = np.max(phi_known) if cmax is None else cmax
        fig = go.Figure()
        fig.add_trace(go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                            i=elements[:, 0], j=elements[:, 1], k=elements[:, 2],
                            intensity=phi_known, colorscale='RdBu_r',cmin=cmin,cmax=cmax,
                            colorbar=dict(title='ϕ react [V]')))
        fig.update_layout(scene=dict(aspectmode='data', xaxis_title='X [Å]', yaxis_title='Y [Å]', zaxis_title='Z [Å]'), margin=dict(l=30, r=40, t=20, b=20),font_family="Times New Roman")
        if jupyter:
            fig.show()
        return fig

    def plot_interface_error(self,method, type_e='relative',scale='log',jupyter=False, ext='html'):
        vertices = self.mesh.mol_verts.astype(np.float32)
        elements = self.mesh.mol_faces.astype(np.float32)

        phi_known = self.phi_known(method,'react', vertices, flag='solvent')
        phi_pinn = self.get_phi(vertices,flag='interface',model=self.model,value='react')

        error = np.abs(phi_pinn.numpy() - phi_known.numpy().reshape(-1,1))
        if type_e == 'relative':
            error /= phi_known.numpy().reshape(-1,1)
        fig = self.plot_interface_error_by(error,vertices,elements,scale,jupyter=False)

        path_save = os.path.join(self.directory, self.path_plots_solution, f'Interface_error_{method}_{type_e}_{scale}')
        if self.save:
            if ext=='html':
                fig.write_html(path_save+'.html')
            elif ext=='png':
                fig.write_image(path_save+'.png', scale=3)

    @staticmethod
    def plot_interface_error_by(error,vertices,elements,scale='log',jupyter=True):
        fig = go.Figure()
        if scale=='log':
            epsilon = 1e-10
            log_intensity = np.log(np.abs(error) + epsilon)
            fig.add_trace(go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                                i=elements[:, 0], j=elements[:, 1], k=elements[:, 2],
                                intensity=log_intensity,
                                colorscale='Plasma',
                                colorbar=dict(
                                    title='Error',
                                    tickvals=np.log(np.array([1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2,1e3]) + epsilon),
                                    ticktext=['1e-7','1e-6','1e-5','1e-4','1e-3','1e-2','1e-1','1e0','1e1','1e2','1e3']
                                )))
        elif scale=='linear':
            fig.add_trace(go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                    i=elements[:, 0], j=elements[:, 1], k=elements[:, 2],
                    intensity=np.abs(error), colorscale='Plasma',
                    colorbar=dict(title='Error')))
        fig.update_layout(scene=dict(aspectmode='data', xaxis_title='X [Å]', yaxis_title='Y [Å]', zaxis_title='Z [Å]'), margin=dict(l=30, r=40, t=20, b=20),font_family="Times New Roman")

        if jupyter:
            fig.show()
        return fig


    def plot_collocation_points_3D(self, jupyter=False, ext='html'):

        color_dict = {
            'Q1_verts': 'red',
            'Q1_sample': 'red',
            'R1_verts': 'lightgreen',
            'R1_sample': 'lightgreen',
            'I_verts': 'purple',
            'I_sample': 'purple',
            'R2_verts': 'lightblue',
            'R2_sample': 'lightblue',
            'D2_verts': 'orange',
            'D2_sample': 'orange',
            'E2_verts': 'cyan',
            'K1_verts': 'yellow',
            'K2_verts': 'yellow'
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

        fig.update_layout(scene=dict(aspectmode="data", xaxis_title='X [Å]', yaxis_title='Y [Å]', zaxis_title='Z [Å]'), margin=dict(l=30, r=40, t=20, b=20),font_family="Times New Roman")
        
        path_save = os.path.join(self.directory,self.path_plots_meshes, 'collocation_points_plot_3d')
        if self.save:
            if ext=='html':
                fig.write_html(path_save+'.html')
            elif ext=='png':
                fig.write_image(path_save+'.png', scale=3)
        if jupyter:
            fig.show()
        return fig

    def plot_surface_mesh_3D(self, jupyter=False, ext='html'):

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
        fig.update_layout(scene=dict(aspectmode='data', xaxis_title='X [Å]', yaxis_title='Y [Å]', zaxis_title='Z [Å]'), margin=dict(l=30, r=40, t=20, b=20),font_family="Times New Roman")

        path_save = os.path.join(self.directory,self.path_plots_meshes,f'mesh_plot_surf_3D')
        if self.save:
            if ext=='html':
                fig.write_html(path_save+'.html')
            elif ext=='png':
                fig.write_image(path_save+'.png', scale=3)
        if jupyter:
            fig.show()
        return fig

    def plot_vol_mesh_3D(self, jupyter=False, ext='html'):
        toRemove = []
        ext_tetmesh = self.mesh.ext_tetmesh
        for vertexID in ext_tetmesh.vertexIDs:
            if vertexID.data()[1] > 0:
                toRemove.append(vertexID)
        for v in toRemove:
            ext_tetmesh.removeVertex(v)
        ext_surfmesh = ext_tetmesh.extractSurface()
        #ext_surfmesh.correctNormals()
        v_ex, e_ex, f_ex = ext_surfmesh.to_ndarray()

        toRemove = []
        int_tetmesh = self.mesh.int_tetmesh
        for vertexID in int_tetmesh.vertexIDs:
            if vertexID.data()[1] > 0:
                toRemove.append(vertexID)
        for v in toRemove:
            int_tetmesh.removeVertex(v)
        int_surfmesh = int_tetmesh.extractSurface()
        #int_surfmesh.correctNormals()
        v_in, e_in, f_in = int_surfmesh.to_ndarray()

        element_trace_in = go.Mesh3d(
                x=v_in[:, 0],
                y=v_in[:, 1],
                z=v_in[:, 2],
                i=f_in[:, 0],
                j=f_in[:, 1],
                k=f_in[:, 2],
                facecolor=['red'] * len(e_in),
                opacity=0.98,
                name='faces_in'
            )
        edge_x = []
        edge_y = []
        edge_z = []
        for element in f_in:
            for i in range(3):
                edge_x.extend([v_in[element[i % 3], 0], v_in[element[(i + 1) % 3], 0], None])
                edge_y.extend([v_in[element[i % 3], 1], v_in[element[(i + 1) % 3], 1], None])
                edge_z.extend([v_in[element[i % 3], 2], v_in[element[(i + 1) % 3], 2], None])
        edge_trace_in = go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode='lines',
            line=dict(color='black', width=3.6),
            name='edges_in'
        )

        element_trace_ex = go.Mesh3d(
            x=v_ex[:, 0],
            y=v_ex[:, 1],
            z=v_ex[:, 2],
            i=f_ex[:, 0],
            j=f_ex[:, 1],
            k=f_ex[:, 2],
            facecolor=['lightblue'] * len(e_ex), 
            opacity=0.98,
            name='faces_ex'
        )
        edge_x = []
        edge_y = []
        edge_z = []
        for element in f_ex:
            for i in range(3):
                edge_x.extend([v_ex[element[i % 3], 0], v_ex[element[(i + 1) % 3], 0], None])
                edge_y.extend([v_ex[element[i % 3], 1], v_ex[element[(i + 1) % 3], 1], None])
                edge_z.extend([v_ex[element[i % 3], 2], v_ex[element[(i + 1) % 3], 2], None])
        edge_trace_ex = go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode='lines',
            line=dict(color='black', width=3.6),
            name='edges_ex'
        )

        fig = go.Figure(data=[element_trace_ex,edge_trace_ex, element_trace_in, edge_trace_in])
        fig.update_layout(scene=dict(aspectmode="data", xaxis_title='X [Å]', yaxis_title='Y [Å]', zaxis_title='Z [Å]'), margin=dict(l=30, r=40, t=20, b=20),font_family="Times New Roman")
        
        path_save = os.path.join(self.directory,self.path_plots_meshes,f'mesh_plot_vol_3D')
        if self.save:
            if ext=='html':
                fig.write_html(path_save+'.html')
            elif ext=='png':
                fig.write_image(path_save+'.png', scale=3)
        if jupyter:
            fig.show()
        return fig


    def plot_mesh_3D(self,domain,element_indices=None, jupyter=False, ext='html'):

        element_indices_input = element_indices

        vertices = self.mesh.region_meshes[domain].vertices
        elements = self.mesh.region_meshes[domain].elements

        x = vertices[:, 0]
        y = vertices[:, 1]
        z = vertices[:, 2]

        trace_vertices = go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(
                size=1,
                color='rgb(255,0,0)',
                opacity=1
            )
        )

        edge_x = []
        edge_y = []
        edge_z = []
        if element_indices is None or len(element_indices) == 0:  
            element_indices = range(len(elements))

        for element_idx in element_indices:
            edges = set()
            if elements.shape[1] == 4: 
                for i in range(4):
                    for j in range(i+1, 4):
                        edge = tuple(sorted([elements[element_idx, i], elements[element_idx, j]]))
                        edges.add(edge)
            elif elements.shape[1] == 3:  
                for i in range(3):
                    for j in range(i+1, 3):
                        edge = tuple(sorted([elements[element_idx, i], elements[element_idx, j]]))
                        edges.add(edge)

            for edge in edges:
                edge_x += [x[edge[0]], x[edge[1]], None]
                edge_y += [y[edge[0]], y[edge[1]], None]
                edge_z += [z[edge[0]], z[edge[1]], None]

        trace_edges = go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode='lines',
            line=dict(
                color='rgb(0,0,0)',  
                width=2
            )
        )

        if element_indices_input is None or len(element_indices_input) == 0:
            fig = go.Figure(data=[trace_vertices, trace_edges])

        else:
            element_x = []
            element_y = []
            element_z = []
            for element_idx in element_indices:
                x = vertices[elements[element_idx], 0]
                y = vertices[elements[element_idx], 1]
                z = vertices[elements[element_idx], 2]
                element_x += list(x)
                element_y += list(y)
                element_z += list(z)

            trace_elements = go.Scatter3d(
                x=element_x,
                y=element_y,
                z=element_z,
                mode='markers',
                marker=dict(
                    size=3,
                    color='rgb(0,222,0)',  
                    opacity=1
                )
            )


            fig = go.Figure(data=[trace_vertices, trace_edges, trace_elements])

        fig.update_layout(scene=dict(aspectmode="data", xaxis_title='X [Å]', yaxis_title='Y [Å]', zaxis_title='Z [Å]'), margin=dict(l=30, r=40, t=20, b=20),font_family="Times New Roman")

        path_save = os.path.join(self.directory,self.path_plots_meshes,f'mesh_plot_3D_{domain}')
        if self.save:
            if ext=='html':
                fig.write_html(path_save+'.html')
            elif ext=='png':
                fig.write_image(path_save+'.png', scale=3)
        if jupyter:
            fig.show()
        return fig

    def plot_surface_mesh_normals(self,plot='vertices',jupyter=False, ext='html'):

        mesh_obj = self.mesh
        
        if plot=='vertices':
            vertices = mesh_obj.mol_verts
            normals = mesh_obj.mol_verts_normal
        elif plot=='faces':
            vertices = mesh_obj.mol_faces_centroid
            normals = mesh_obj.mol_faces_normal

        mesh_trace = go.Mesh3d(x=mesh_obj.mol_verts[:, 0], 
                       y=mesh_obj.mol_verts[:, 1], 
                       z=mesh_obj.mol_verts[:, 2], 
                       i=mesh_obj.mol_faces[:, 0], 
                       j=mesh_obj.mol_faces[:, 1], 
                       k=mesh_obj.mol_faces[:, 2],
                       color='lightblue', 
                       opacity=0.5)

        vertex_normals_trace = go.Cone(x=vertices[:, 0],
                                        y=vertices[:, 1],
                                        z=vertices[:, 2],
                                        u=normals[:, 0],
                                        v=normals[:, 1],
                                        w=normals[:, 2],
                                        sizemode="absolute",
                                        showscale=False,
                                        colorscale=[[0, 'red'], [1, 'red']],
                                        hoverinfo='none')

        edge_x = []
        edge_y = []
        edge_z = []

        for element in mesh_obj.mol_faces:
            for i in range(3):
                edge_x.extend([mesh_obj.mol_verts[element[i % 3], 0], mesh_obj.mol_verts[element[(i + 1) % 3], 0], None])
                edge_y.extend([mesh_obj.mol_verts[element[i % 3], 1], mesh_obj.mol_verts[element[(i + 1) % 3], 1], None])
                edge_z.extend([mesh_obj.mol_verts[element[i % 3], 2], mesh_obj.mol_verts[element[(i + 1) % 3], 2], None])

        edge_trace = go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode='lines',
            line=dict(color='black', width=2),
        )

        fig = go.Figure(data=[mesh_trace, vertex_normals_trace, edge_trace])
        fig.update_layout(scene=dict(aspectmode="data", xaxis_title='X [Å]', yaxis_title='Y [Å]', zaxis_title='Z [Å]'), margin=dict(l=30, r=40, t=20, b=20),font_family="Times New Roman")


        path_save = os.path.join(self.directory,self.path_plots_meshes,f'mesh_plot_surface_normals_{plot}')
        if self.save:
            if ext=='html':
                fig.write_html(path_save+'.html')
            elif ext=='png':
                fig.write_image(path_save+'.png', scale=3)
        if jupyter:
            fig.show()
        return fig


    @staticmethod
    def get_max_min(u1,u2):
        U = tf.concat([u1,u2], axis=0)
        vmax = tf.reduce_max(U)
        vmin = tf.reduce_min(U)
        return vmax,vmin

    def get_points_line(self,N,x0,theta,phi):
        r = np.linspace(-self.mesh.R_exterior,self.mesh.R_exterior,N)
        x = x0[0] + r * np.sin(phi) * np.cos(theta)
        y = x0[1] + r * np.sin(phi) * np.sin(theta)
        z = x0[2] + r * np.cos(phi)
        points = np.stack((x.ravel(), y.ravel(), z.ravel()), axis=1)
        X_in,X_out,bools = self.get_interior_exterior(points)
        return X_in,X_out,r[bools[0]],r[bools[1]]             

    def get_points_plane(self,N, x0, n):

        u,v = self.normal_vector_n(n)
    
        R_exterior = self.mesh.R_mol+self.mesh.dR_exterior/2
        t = np.linspace(-R_exterior,R_exterior,N)
        s = np.linspace(-R_exterior,R_exterior,N)
        T,S = np.meshgrid(t,s)
        x = x0[0] + T*u[0] + S*v[0]
        y = x0[1] + T*u[1] + S*v[1]
        z = x0[2] + T*u[2] + S*v[2]
        points = np.stack((x.ravel(), y.ravel(), z.ravel()), axis=1)
        X_in,X_out,bools = self.get_interior_exterior(points,R_exterior)
        return X_in,X_out,bools,(T,S)

    @staticmethod
    def normal_vector_n(n):
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
        return u,v

    def get_interior_exterior(self,points,R_exterior=None):
        if R_exterior==None:
            R_exterior = self.mesh.R_exterior
        interior_points_bool = self.mesh.mol_mesh.contains(points)
        interior_points = points[interior_points_bool]
        exterior_points_bool = ~interior_points_bool  
        exterior_points = points[exterior_points_bool]
        exterior_distances = np.linalg.norm(exterior_points, axis=1)
        exterior_points = exterior_points[exterior_distances <= R_exterior]
        bool_2 = np.linalg.norm(points, axis=1) <= R_exterior*exterior_points_bool
        bools = (interior_points_bool,bool_2)
        return interior_points,exterior_points, bools

    def L2_interface_known(self,known_method):
        vertices = self.mesh.mol_verts.astype(np.float32)
        phi_known = self.phi_known(known_method,'react', vertices, flag='solvent')
        phi_xpinn = self.get_phi_interface_verts(self.model,value='react')[0]

        if known_method == 'PBJ':
            vertices = self.PDE.pbj_vertices.astype(np.float32)
            phi_known = self.phi_known(known_method,'react', vertices, flag='solvent')
            phi_xpinn = self.get_phi(vertices,flag='interface',model=self.model,value='react')

        phi_dif = (phi_xpinn.numpy() - phi_known.numpy().reshape(-1,1))
        error = np.sqrt(np.sum(phi_dif**2)/np.sum(phi_known.numpy()**2))
        return error

    def save_values_file(self, save=True, L2_err_method=None):
     
        max_iter = max(map(int,list(self.PINN.G_solv_hist.keys())))
        Gsolv_value = self.PINN.G_solv_hist[str(max_iter)]

        dict_pre = {
            'Gsolv_value': Gsolv_value,
            'Loss_PINN': self.loss_last[0],
            'Loss_NN1': self.loss_last[1],
            'Loss_NN2': self.loss_last[2],
            'Loss_Val_NN1': self.PINN.validation_losses['TL1'][-1],
            'Loss_Val_NN2': self.PINN.validation_losses['TL2'][-1],
            'Loss_continuity_u': self.PINN.losses['Iu'][-1],
            'Loss_continuity_du': self.PINN.losses['Id'][-1],
            'Loss_Residual_R1': self.PINN.losses['R1'][-1],
            'Loss_Residual_R2': self.PINN.losses['R2'][-1],
            'Loss_Boundary_D2': self.PINN.losses['D2'][-1],
            'Loss_Data_K2': self.PINN.losses['K2'][-1]
        } 

        if not L2_err_method is None:
            dict_pre['L2_error'] = self.L2_interface_known(L2_err_method)

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
            self.PINN.model.summary(print_fn=print_func)
            print("\n\n", file=f)
            self.PINN.model.NNs[0].summary(print_fn=print_func)
            print("\n\n", file=f) 
            self.PINN.model.NNs[1].summary(print_fn=print_func)
        
        path_save = os.path.join(self.directory,self.path_plots_model,'hyperparameters.json')
        with open(path_save, "w") as json_file:
            json.dump({'Molecule_NN': self.PINN.hyperparameters[0], 'Solvent_NN': self.PINN.hyperparameters[1]}, json_file, indent=4)

    def plot_architecture(self,domain=1):
        
        domain -= 1
        input_layer = tf.keras.layers.Input(shape=self.PINN.model.NNs[domain].input_shape_N[1:], name='input')
        visual_model = tf.keras.models.Model(inputs=input_layer, outputs=self.PINN.model.NNs[domain].call(input_layer))

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



class Born_Ion_Postprocessing(Postprocessing):

    def __init__(self,*kargs,**kwargs):
        super().__init__(*kargs,**kwargs)

    def plot_aprox_analytic(self, N=8000, x0=np.array([0,0,0]), theta=0, phi=np.pi/2, zoom=False, value='phi'):
        
        fig, ax = plt.subplots()
        
        X_in,X_out,r_in,r_out = self.get_points_line(N,x0,theta,phi)
        
        u_in = self.get_phi(tf.constant(X_in, dtype=self.DTYPE),'molecule',self.model, value)[:,0]
        u_out = self.get_phi(tf.constant(X_out, dtype=self.DTYPE),'solvent',self.model, value)[:,0]

        ax.plot(r_in,u_in[:], c='b')
        ax.plot(r_out[r_out<0],u_out[r_out<0], label='PINN', c='b')
        ax.plot(r_out[r_out>0],u_out[r_out>0], c='b')

        u_in_an = self.phi_known('analytic_Born_Ion',value,tf.constant(X_in, dtype=self.DTYPE),'molecule')
        u_out_an = self.phi_known('analytic_Born_Ion',value,tf.constant(X_out, dtype=self.DTYPE),'solvent')

        ax.plot(r_in[np.abs(r_in) > 0.05],u_in_an[np.abs(r_in) > 0.05], c='r', linestyle='--')
        ax.plot(r_out[r_out<0],u_out_an[r_out<0], label='Analytic', c='r', linestyle='--')
        ax.plot(r_out[r_out>0],u_out_an[r_out>0], c='r', linestyle='--')

        if zoom:
            R = self.PINN.mesh.R_mol
            v = self.phi_known('analytic_Born_Ion',value,tf.constant([[R,0,0]], dtype=self.DTYPE),'solvent')

            if value == 'phi':
                axin = ax.inset_axes([0.65, 0.25, 0.28, 0.34])
                axin.set_ylim(-0.06*self.to_V+v, 0.06*self.to_V+v)
            else:
                axin = ax.inset_axes([0.68, 0.20, 0.28, 0.34])
                axin.set_ylim(-0.006*self.to_V+v, 0.006*self.to_V+v)
            axin.set_xlim(0.9*R,1.1*R)

            axin.plot(r_in,u_in[:], c='b')
            axin.plot(r_out[r_out<0],u_out[r_out<0], c='b')
            axin.plot(r_out[r_out>0],u_out[r_out>0], c='b')

            axin.plot(r_in[np.abs(r_in) > 0.04],u_in_an[np.abs(r_in) > 0.04], c='r', linestyle='--')
            axin.plot(r_out[r_out<0],u_out_an[r_out<0], c='r', linestyle='--')
            axin.plot(r_out[r_out>0],u_out_an[r_out>0], c='r', linestyle='--')

            axin.grid()
            ax.indicate_inset_zoom(axin)
        
        text_A = r'$\AA$'
        text_r = r'$r$'
        ax.set_xlabel(f'{text_r} [{text_A}]', fontsize='11')
        text_l = r'$\phi$' if value=='phi' else r'$\phi_{react}$'
        ax.set_ylabel(f'{text_l} [V]', fontsize='11')
        text_theta = r'$\theta$'
        text_phi = r'$\phi$'
        theta = np.format_float_positional(theta, unique=False, precision=2)
        phi = np.format_float_positional(phi, unique=False, precision=2)
        text_x0 = r'$x_0$'
        # ax.set_title(f'Solution {text_l} of PBE;  ({text_x0}=[{x0[0]},{x0[1]},{x0[2]}] {text_theta}={theta}, {text_phi}={phi})')
        ax.grid()
        ax.legend()        

        ax.set_xlim(-5,5)
        if value == 'phi':
            ax.set_ylim(-0.1*self.to_V,1.6*self.to_V)
        elif value == 'react':
            ax.set_ylim(-0.085*self.to_V,-0.01*self.to_V)

        if self.save:
            path = f'analytic_{value}.png' if zoom==False else f'analytic_zoom_{value}.png'
            path_save = os.path.join(self.directory,self.path_plots_solution,path)
            fig.savefig(path_save, bbox_inches='tight')
        return fig,ax


    def plot_line_interface(self,N=1000, plot='u',value='phi', nn=np.array([0,1,0])):

        labels = ['Solute', 'Solvent']
        colr = ['r','b']
        i = 0

        rr = self.PINN.mesh.R_mol
        uu,vv = self.normal_vector_n(nn)
        theta_bl = np.linspace(0, 2*np.pi, N, dtype=self.DTYPE)
        X = np.zeros((N,3))
        for i in range(N):
            X[i,:] = rr*np.cos(theta_bl[i])*uu + rr*np.sin(theta_bl[i])*vv
        
        theta_bl = tf.constant(theta_bl.reshape(-1,1))
        XX_bl = tf.constant(X, dtype=self.DTYPE)

        fig, ax = plt.subplots() 

        for i,flag in zip([0,1],['molecule','solvent']):
            if plot=='u':
                U = self.get_phi(XX_bl,flag,self.model,value)[:,0]
                ax.plot(theta_bl[:,0],U[:], label=labels[i], c=colr[i])
            elif plot=='du':
                radial_vector = XX_bl
                magnitude = tf.norm(radial_vector, axis=1, keepdims=True)
                normal_vector = radial_vector / magnitude
                du = self.get_dphi(XX_bl,normal_vector,flag,self.model,value)
                if i==0:
                    ax.plot(theta_bl[:,0],du[i][:,0]*self.PDE.epsilon_1, label=labels[i], c=colr[i])
                else:
                    ax.plot(theta_bl[:,0],du[i][:,0]*self.PDE.epsilon_2, label=labels[i], c=colr[i])
            i += 1

        if plot=='u':
            U2 = self.phi_known('analytic_Born_Ion',value,tf.constant([[rr,0,0]], dtype=self.DTYPE),'')
            u2 = np.ones((N,1))*U2
            ax.plot(theta_bl, u2, c='g', label='Analytic', linestyle='--')

        elif plot=='du':
            dU2 = self.PINN.PDE.analytic_Born_Ion_du(rr)*self.to_V
            du2 = np.ones((N,1))*dU2*self.PDE.epsilon_1
            if value=='react':
                n = du2.shape[0]
                nnvv = tf.concat([tf.ones((n, 1)), tf.zeros((n, 2))], axis=1)
                XX2 = tf.concat([tf.ones((n,1))*rr, tf.zeros((n, 2))], axis=1)
                du2 -= self.PINN.PDE.dG_n(*self.mesh.get_X(XX2),nnvv)*self.to_V
            ax.plot(theta_bl, du2, c='g', label='Analytic', linestyle='--')
        
        if plot=='u':
            unit = 'V'
            if value == 'phi':
                text_l = r'$\phi$'
            else:
                text_l = r'$\phi_{react}$'
        elif plot=='du':
            unit = r'V/$\AA$'
            if value == 'phi':
                text_l = r'$\partial_n\phi$'
            else:
                text_l = r'$\partial_n\phi_{react}$'  
        ax.set_xlabel(r'$\beta$ [rad]', fontsize='11')
        ax.set_ylabel(f'{text_l} [{unit}]', fontsize='11')
        text_n = r'$n$'
        # ax.set_title(f'Solution {text_l} of PBE at Interface;  ({text_n}=[{np.format_float_positional(nn[0], unique=False, precision=1)},{np.format_float_positional(nn[1], unique=False, precision=1)},{np.format_float_positional(nn[2], unique=False, precision=1)}])')

        ax.grid()
        ax.legend()

        if self.save:
            path = f'interface_line_{plot}_{value}.png'
            path_save = os.path.join(self.directory,self.path_plots_solution,path)
            fig.savefig(path_save, bbox_inches='tight')
        return fig,ax


    
    def save_values_file(self,save=True,L2_err_method='analytic_Born_Ion'):

        if L2_err_method is None:
            L2_err_method = 'analytic_Born_Ion'
        L2_analytic = self.L2_interface_known(L2_err_method)
        
        df_dict = super().save_values_file(save=False)
        df_dict['L2_analytic'] = '{:.3e}'.format(float(L2_analytic))

        if not save:
            return df_dict
        
        path_save = os.path.join(self.directory,'results_values.json')
        with open(path_save, 'w') as json_file:
            json.dump(df_dict, json_file, indent=4)
