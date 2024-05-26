import os
import sys
import yaml
import logging
import shutil
import numpy as np

from xppbe.Mesh.Mesh import Domain_Mesh
from xppbe.NN.XPINN import XPINN


class Simulation():
       
    def __init__(self, yaml_path, molecule_path=None, results_path=None):

        self.simulation_name, self.results_path, molecule_path, self.main_path,self.logger = self.get_simulation_paths(yaml_path,molecule_path,results_path)

        with open(os.path.join(self.main_path,'Simulation.yaml'), 'r') as yaml_file:
            yaml_data = yaml.safe_load(yaml_file)
        for key, value in yaml_data.items():
            setattr(self, key, value)

        with open(yaml_path, 'r') as yaml_file:
            yaml_data = yaml.safe_load(yaml_file)
        for key, value in yaml_data.items():
            setattr(self, key, value)

        self.molecule_path = os.path.join(molecule_path,self.domain_properties['molecule'])


    def create_simulation(self):

        self.logger.info(f"Solving PBE {self.equation}, in {self.pbe_model} form")
        self.logger.info(f"Molecule: {self.domain_properties['molecule']}")

        self.Mol_mesh = Domain_Mesh(self.domain_properties['molecule'], 
                        mesh_properties=self.mesh_properties, 
                        save_points=True,
                        path=self.main_path,
                        simulation=self.simulation_name,
                        result_path=self.results_path,
                        molecule_path=self.molecule_path
                        )

        if self.equation == 'standard':
                from xppbe.Model.Equations import PBE_Std as PBE
        elif self.equation == 'regularized_scheme_1':
                from xppbe.Model.Equations import PBE_Reg_1 as PBE
        elif self.equation == 'regularized_scheme_2':
                from xppbe.Model.Equations import PBE_Reg_2 as PBE

        self.PBE_model = PBE(self.domain_properties,
                mesh=self.Mol_mesh, 
                equation=self.pbe_model,
                main_path=self.main_path,
                molecule_path=self.molecule_path
                ) 

        meshes_domain = dict()           
        if self.equation == 'standard':
            meshes_domain['R1'] = {'domain': 'molecule', 'type':'R1', 'fun':lambda X: self.PBE_model.source(X)}
            meshes_domain['Q1'] = {'domain': 'molecule', 'type':'Q1', 'fun':lambda X: self.PBE_model.source(X)}
            meshes_domain['R2'] = {'domain': 'solvent', 'type':'R2', 'value':0.0}
            meshes_domain['D2'] = {'domain': 'solvent', 'type':'D2', 'fun':lambda X: self.PBE_model.G_Yukawa(X)}
                
        elif self.equation == 'regularized_scheme_1':
            meshes_domain['R1'] = {'domain': 'molecule', 'type':'R1', 'value':0.0}
            meshes_domain['R2'] = {'domain': 'solvent', 'type':'R2', 'value':0.0}
            meshes_domain['D2'] = {'domain': 'solvent', 'type':'D2', 'fun':lambda X: self.PBE_model.G_Yukawa(X)-self.PBE_model.G(X)}

        elif self.equation == 'regularized_scheme_2':
            meshes_domain['R1'] = {'domain': 'molecule', 'type':'R1', 'value':0.0}
            meshes_domain['R2'] = {'domain': 'solvent', 'type':'R2', 'value':0.0}
            meshes_domain['D2'] = {'domain': 'solvent', 'type':'D2', 'fun':lambda X: self.PBE_model.G_Yukawa(X)}

        meshes_domain['K1'] = {'domain': 'molecule', 'type':'K1', 'file':'data_known.dat'}
        meshes_domain['K2'] = {'domain': 'solvent', 'type':'K2', 'file':'data_known.dat'}
        meshes_domain['E2'] = {'domain': 'solvent', 'type': 'E2', 'file': 'data_experimental.dat', 'method': self.experimental_method}

        if self.network=='xpinn':
                meshes_domain['Iu'] = {'domain':'interface', 'type':'Iu'}
                meshes_domain['Id'] = {'domain':'interface', 'type':'Id'}
                meshes_domain['Ir'] = {'domain':'interface', 'type':'Ir'}
        meshes_domain['G'] = {'domain':'interface', 'type':'G'}

        self.meshes_domain = dict()
        for t in self.losses:
            self.meshes_domain[t] = meshes_domain[t]

        self.PBE_model.mesh.adapt_meshes_domain(self.meshes_domain,self.PBE_model.q_list)

        self.XPINN_solver = XPINN(results_path=self.results_path)
        self.XPINN_solver.adapt_PDE(self.PBE_model)


    def adapt_model(self):

        self.XPINN_solver.adapt_weights(
                self.weights,
                adapt_weights = self.adapt_weights,
                adapt_w_iter = self.adapt_w_iter,
                adapt_w_method = self.adapt_w_method,
                alpha_w = self.alpha_w
                )       
        self.XPINN_solver.adapt_optimizer(self.optimizer,self.lr, self.optimizer2,self.options_optimizer2) 
        self.XPINN_solver.set_points_methods(sample_method=self.sample_method)

        if self.network == 'xpinn':
            from xppbe.NN.NeuralNet import XPINN_NeuralNet as NeuralNet
        elif self.network == 'pinn':
            from xppbe.NN.NeuralNet import PINN_NeuralNet as NeuralNet     


        if self.starting_point == 'new':
            
            self.hyperparameters_in['scale'] = self.XPINN_solver.mesh.scale_1
            self.hyperparameters_out['scale'] = self.XPINN_solver.mesh.scale_2

            if self.scale_NN_q:
                self.hyperparameters_in['scale_NN'] = self.PBE_model.scale_q_factor
                self.hyperparameters_out['scale_NN'] = self.PBE_model.scale_q_factor 

            self.XPINN_solver.create_NeuralNet(NeuralNet,[self.hyperparameters_in,self.hyperparameters_out])
        
        elif self.starting_point == 'continue':
            self.XPINN_solver.load_NeuralNet(NeuralNet,self.results_path,f'iter_{self.continue_iteration}',self.continue_iteration)
        
        self.XPINN_solver.starting_point = self.starting_point


    def solve_model(self):
          
        self.XPINN_solver.solve(
            N=self.N_iters, 
            N2=self.N_steps_2,
            save_model = self.iters_save_model, 
            G_solve_iter= self.G_solve_iter
            )
        

    def postprocessing(self, jupyter=False, mesh=True, pbj=False):
          
        if self.domain_properties['molecule'] == 'born_ion':
            from xppbe.Post.Postcode import Born_Ion_Postprocessing as Postprocessing
        else:
            from xppbe.Post.Postcode import Postprocessing

        self.Post = Postprocessing(self.XPINN_solver, save=True, directory=self.results_path)
        
        if jupyter:
             return self.Post
        
        shutil.copy(os.path.join(self.main_path,'Post','Post_Template.ipynb'),os.path.join(self.results_path,'Post.ipynb'))

        self.Post.plot_loss_history(domain=1);
        self.Post.plot_loss_history(domain=2);
        self.Post.plot_loss_history(domain=1, plot_w=True);
        self.Post.plot_loss_history(domain=2, plot_w=True);
        # self.Post.plot_loss_history(domain=1,loss='RIu');
        # self.Post.plot_loss_history(domain=2,loss='RIuD');
        
        self.Post.plot_loss_validation_history(domain=1,loss='TL');
        self.Post.plot_loss_validation_history(domain=2,loss='TL');
        self.Post.plot_loss_validation_history(domain=1,loss='R');
        self.Post.plot_loss_validation_history(domain=2,loss='R');

        self.Post.plot_weights_history(domain=1);
        self.Post.plot_weights_history(domain=2);

        if mesh:
            self.Post.plot_collocation_points_3D();
            self.Post.plot_vol_mesh_3D();
            self.Post.plot_surface_mesh_3D();
            self.Post.plot_mesh_3D('R1');
            self.Post.plot_mesh_3D('R2');
            self.Post.plot_mesh_3D('I');
            self.Post.plot_mesh_3D('D2');
            self.Post.plot_surface_mesh_normals(plot='vertices');
            self.Post.plot_surface_mesh_normals(plot='faces');

        self.Post.plot_G_solv_history();
        self.Post.plot_phi_line();
        self.Post.plot_phi_line(value='react');
        self.Post.plot_phi_contour();
        self.Post.plot_phi_contour(value='react');
        self.Post.plot_interface_3D(variable='phi');
        self.Post.plot_interface_3D(variable='dphi');

        if self.domain_properties['molecule'] == 'born_ion':
            self.Post.plot_aprox_analytic();
            self.Post.plot_aprox_analytic(value='react');
            self.Post.plot_aprox_analytic(zoom=True);
            self.Post.plot_aprox_analytic(zoom=True, value='react');
            self.Post.plot_line_interface();
            self.Post.plot_line_interface(value='react');
            self.Post.plot_line_interface(plot='du');
            self.Post.plot_G_solv_history(known=True,method='analytic_Born_Ion');
            #self.Post.plot_line_interface(plot='du',value='react');
        else:
            if 'sphere' in self.domain_properties['molecule']:
                method = 'Spherical_Harmonics'
                plot_aprox = True
            elif self.domain_properties['molecule'] in ('methanol','arg'):
                method = 'PBJ'
                plot_aprox = pbj
             
            if plot_aprox:
                self.Post.plot_G_solv_history(known=True,method=method);
                self.Post.plot_phi_line_aprox_known(method, value='react',theta=0, phi=np.pi/2)
                self.Post.plot_phi_line_aprox_known(method, value='react',theta=np.pi/2, phi=np.pi/2)
                self.Post.plot_phi_line_aprox_known(method, value='react', theta=np.pi/2, phi=np.pi)

        self.Post.save_values_file();
        self.Post.save_model_summary();
        self.Post.plot_architecture(domain=1);
        self.Post.plot_architecture(domain=2);
    
        return self.Post


    def load_model_for_Post(self,Iter,save=False):

        if self.network == 'xpinn':
            from xppbe.NN.NeuralNet import XPINN_NeuralNet as NeuralNet
        elif self.network == 'pinn':
            from xppbe.NN.NeuralNet import PINN_NeuralNet as NeuralNet

        self.XPINN_solver.results_path = self.results_path
        self.XPINN_solver.load_NeuralNet(NeuralNet,self.results_path,f'iter_{Iter}',Iter)
        self.XPINN_solver.N_iters = self.N_iters
         
        if self.domain_properties['molecule'] == 'born_ion':
            from xppbe.Post.Postcode import Born_Ion_Postprocessing as Postprocessing
        else:
            from xppbe.Post.Postcode import Postprocessing

        self.Post = Postprocessing(self.XPINN_solver, save=save, directory=self.results_path)


    def get_simulation_paths(self,yaml_path, molecule_path=None, results_path=None):
    
        from xppbe import xppbe_path
        main_path = xppbe_path
        simulation_name = os.path.basename(yaml_path).split('.')[0]

        folder_name = simulation_name
        if results_path is None:
            folder_path = os.path.dirname(os.path.abspath(yaml_path))
            results_path = os.path.join(folder_path,'results',folder_name)
        else:
            results_path = os.path.join(results_path,'results',folder_name)

        if molecule_path is None:
            folder_path = xppbe_path
            molecule_path = os.path.join(folder_path,'Molecules')
        else:
            molecule_path = os.path.join(molecule_path)

        # if clean_results:
        #     if os.path.exists(results_path):
        #             shutil.rmtree(results_path)
        os.makedirs(results_path, exist_ok=True)
            
        self.starting_point = 'new'
        if os.path.exists(os.path.join(results_path,'iterations')):
            if os.listdir(os.path.join(results_path,'iterations')):
                self.starting_point = 'continue'
                folder_path = os.path.join(results_path,'iterations')
                subdirectories = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
                iteration_numbers = [int(d.split('_')[1]) for d in subdirectories if d.startswith('iter_')]
                self.continue_iteration = max(iteration_numbers)
                
        try:
            shutil.copy(yaml_path,os.path.join(results_path))
        except shutil.SameFileError:
            pass

        filename = os.path.join(results_path,'logfile.log')
        LOG_format = '%(levelname)s - %(name)s: %(message)s'
        logging.basicConfig(filename=filename, filemode='w', level=logging.INFO, format=LOG_format)
        logger = logging.getLogger(__name__)
        
        return simulation_name,results_path,molecule_path, main_path,logger



if __name__=='__main__':
    yaml_path = os.path.abspath(sys.argv[1])
    results_path = os.path.dirname(__file__)
    sim = Simulation(yaml_path, results_path=results_path)
    sim.create_simulation()
    sim.adapt_model()
    sim.solve_model()
    sim.postprocessing(mesh=False, pbj=True)

