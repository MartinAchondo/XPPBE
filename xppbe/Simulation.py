import os
import sys
import yaml
import logging
import shutil
import numpy as np


class Simulation():
       
    def __init__(self, yaml_path, molecule_dir, results_path=None):


        from xppbe import xppbe_path
        with open(os.path.join(xppbe_path,'Simulation.yaml'), 'r') as yaml_file:
            yaml_data = yaml.safe_load(yaml_file)
        for key, value in yaml_data.items():
            setattr(self, key, value)

        with open(yaml_path, 'r') as yaml_file:
            yaml_data = yaml.safe_load(yaml_file)
        for key, value in yaml_data.items():
            setattr(self, key, value)

        self.simulation_name, self.results_path, self.molecule_dir, self.main_path,self.logger = self.get_simulation_paths(yaml_path,molecule_dir,results_path)

    def create_simulation(self):

        self.logger.info(f"Solving PBE {self.equation}, in {self.pbe_model} form")
        self.logger.info(f"Molecule: {self.domain_properties['molecule']}")

        from xppbe.Mesh.Mesh import Domain_Mesh
        self.Mol_mesh = Domain_Mesh(self.domain_properties['molecule'], 
                        mesh_properties=self.mesh_properties, 
                        save_points=True,
                        path=self.main_path,
                        simulation=self.simulation_name,
                        results_path=self.results_path,
                        molecule_dir=self.molecule_dir
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
                molecule_dir=self.molecule_dir,
                results_path=self.results_path
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

        meshes_domain['K1'] = {'domain': 'molecule', 'type':'K1', 'file':f'known_data_{self.domain_properties["molecule"]}.dat'}
        meshes_domain['K2'] = {'domain': 'solvent', 'type':'K2', 'file':f'known_data_{self.domain_properties["molecule"]}.dat'}
        meshes_domain['E2'] = {'domain': 'solvent', 'type': 'E2', 'file': f'experimental_data_{self.domain_properties["molecule"]}.dat', 'method': self.experimental_method}

        if self.num_networks==2:
                meshes_domain['Iu'] = {'domain':'interface', 'type':'Iu'}
                meshes_domain['Id'] = {'domain':'interface', 'type':'Id'}
                meshes_domain['Ir'] = {'domain':'interface', 'type':'Ir'}
        meshes_domain['G'] = {'domain':'interface', 'type':'G'}

        self.meshes_domain = dict()
        for t in self.losses:
            self.meshes_domain[t] = meshes_domain[t]

        self.PBE_model.mesh.adapt_meshes_domain(self.meshes_domain,self.PBE_model.q_list)

        from xppbe.NN.PINN import PINN
        self.PINN_solver = PINN(results_path=self.results_path)
        self.PINN_solver.adapt_PDE(self.PBE_model)


    def adapt_model(self):

        self.PINN_solver.adapt_weights(
                self.weights,
                adapt_weights = self.adapt_weights,
                adapt_w_iter = self.adapt_w_iter,
                adapt_w_method = self.adapt_w_method,
                alpha_w = self.alpha_w
                )       
        self.PINN_solver.adapt_optimizer(self.optimizer,self.lr, self.optimizer2,self.options_optimizer2) 
        self.PINN_solver.set_points_methods(sample_method=self.sample_method)

        if self.num_networks == 2:
            from xppbe.NN.NeuralNet import PINN_2Dom_NeuralNet as NeuralNet
        elif self.num_networks == 1:
            from xppbe.NN.NeuralNet import PINN_1Dom_NeuralNet as NeuralNet     


        if self.starting_point == 'new':
            
            self.hyperparameters_in['scale'] = self.PINN_solver.mesh.scale_1
            self.hyperparameters_out['scale'] = self.PINN_solver.mesh.scale_2

            if self.scale_NN_q:
                self.hyperparameters_in['scale_NN'] = self.PBE_model.scale_q_factor
                self.hyperparameters_out['scale_NN'] = self.PBE_model.scale_q_factor 

            self.PINN_solver.create_NeuralNet(NeuralNet,[self.hyperparameters_in,self.hyperparameters_out])
        
        elif self.starting_point == 'continue':
            self.PINN_solver.load_NeuralNet(NeuralNet,self.results_path,f'iter_{self.continue_iteration}',self.continue_iteration)
        
        self.PINN_solver.starting_point = self.starting_point


    def solve_model(self):
          
        self.PINN_solver.solve(
            N=self.N_iters, 
            N2=self.N_steps_2,
            save_model = self.iters_save_model, 
            G_solve_iter= self.G_solve_iter
            )
        

    def postprocessing(self, run_all=False, jupyter=False, plot_mesh=False, known_method=None):
          
        if self.domain_properties['molecule'] == 'born_ion':
            from xppbe.Post.Postcode import Born_Ion_Postprocessing as Postprocessing
        else:
            from xppbe.Post.Postcode import Postprocessing

        self.Post = Postprocessing(self.PINN_solver, save=True, directory=self.results_path)
        
        if jupyter:
             return self.Post
        
        shutil.copy(os.path.join(self.main_path,'Post','Post_Template.ipynb'),os.path.join(self.results_path,'Post.ipynb'))

        if run_all:
            self.Post.run_all(plot_mesh,known_method)
    
        return self.Post


    def load_model_for_Post(self,Iter,save=False):

        if self.num_networks == 2:
            from xppbe.NN.NeuralNet import PINN_2Dom_NeuralNet as NeuralNet
        elif self.num_networks == 1:
            from xppbe.NN.NeuralNet import PINN_1Dom_NeuralNet as NeuralNet  

        self.PINN_solver.results_path = self.results_path
        self.PINN_solver.load_NeuralNet(NeuralNet,self.results_path,f'iter_{Iter}',Iter)
        self.PINN_solver.N_iters = self.N_iters
         
        if self.domain_properties['molecule'] == 'born_ion':
            from xppbe.Post.Postcode import Born_Ion_Postprocessing as Postprocessing
        else:
            from xppbe.Post.Postcode import Postprocessing

        self.Post = Postprocessing(self.PINN_solver, save=save, directory=self.results_path)


    def get_simulation_paths(self,yaml_path, molecule_dir=None, results_path=None):
    
        from xppbe import xppbe_path
        main_path = xppbe_path
        simulation_name = os.path.basename(yaml_path).replace('.yaml','')

        folder_name = simulation_name
        if results_path is None:
            folder_path = os.path.dirname(os.path.abspath(yaml_path))
            results_path = os.path.join(folder_path,'results',folder_name)
        else:
            results_path = results_path

        if molecule_dir is None:
            folder_path = xppbe_path
            molecule_dir = os.path.join(folder_path,'Molecules',self.domain_properties['molecule'])
        else:
            molecule_dir = os.path.join(molecule_dir)

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
        
        return simulation_name,results_path,molecule_dir, main_path,logger
