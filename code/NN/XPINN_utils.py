import copy
import logging
import os
import pandas as pd


logger = logging.getLogger(__name__)

class XPINN_utils():

    DTYPE='float32'

    def __init__(self):

        self.losses = dict()
        self.losses_names = ['TL','R1','D1','N1','K1','Q','R2','D2','N2','K2','G','Iu','Id','E','P1','P2']
        self.losses_names_1 = ['TL','R','D','N','K','Q','Iu','Id','P']
        self.losses_names_2 = ['TL','R','D','N','K','Iu','Id','E','G','P']
        self.validation_names = ['TL','R1','D1','N1','Q','R2','D2','N2','Iu','Id']
        for t in self.losses_names:
            self.losses[t] = list()

        self.validation_losses = dict()
        for t in self.validation_names:
            self.validation_losses[t] = list()

        self.G_solv_hist = dict()
        self.G_solv_hist['0'] = 0.0

        self.iter = 0
        self.lr = None

    def adapt_PDEs(self,PDE):
        self.PDE = PDE
        self.mesh = self.PDE.mesh
        for solver,pde in zip(self.solvers, PDE.get_PDEs):
            solver.adapt_PDE(pde)
        self.adapt_datasets()
        self.set_mesh_names()
        print("PDEs and datasets ready")

    def adapt_weights(self,weights,adapt_weights=False,adapt_w_iter=2000,adapt_w_method='gradients',alpha=0.3):
        self.adapt_weights = adapt_weights
        self.adapt_w_iter = adapt_w_iter
        self.adapt_w_method = adapt_w_method
        self.alpha_w = alpha

        for solver,weight in zip(self.solvers,weights):
            solver.adapt_weights(**weight) 

    def create_NeuralNets(self,NN_class,hyperparameters):
        for solver,hyperparameter in zip(self.solvers,hyperparameters):
            solver.create_NeuralNet(NN_class,**hyperparameter)

    def adapt_optimizers(self,optimizer,lrs,lr_p=0.001):
        for solver,lr in zip(self.solvers,lrs):
            solver.adapt_optimizer(optimizer,lr,lr_p)
        self.optimizer_name = optimizer

    def set_points_methods(self, sample_method='batches', N_batches=1, sample_size=1000):
        self.sample_method = sample_method
        self.N_batches = N_batches
        self.sample_size = sample_size

    def load_NeuralNets(self,dir_load,names):   
        for solver,name in zip(self.solvers,names):
            solver.adapt_weights()
            solver.load_NeuralNet(dir_load,name)  
        
        path_load = os.path.join(dir_load,'loss.csv')
        df = pd.read_csv(path_load)
        for t in self.losses_names:
            self.losses[t] = list(df[t])
        self.iter = len(self.losses['TL']) 

        path_load = os.path.join(dir_load,'loss_validation.csv')
        df = pd.read_csv(path_load)
        for t in self.validation_names:
            self.validation_losses[t] = list(df[t])

        self.add_losses_NN()

        path_load = os.path.join(dir_load,'G_solv.csv')
        df_2 = pd.read_csv(path_load)
        for iters,G_solv in zip(list(df_2['iter']),list(df_2['G_solv'])):
            self.G_solv_hist[str(iters)] = G_solv

    def save_models(self,dir_save,names):
        for solver,name in zip(self.solvers,names):
            solver.save_model(dir_save,name)  

        df_dict = dict()
        for t in self.losses_names:
            df_dict[t] = list(map(lambda tensor: tensor.numpy(),self.losses[t]))
        df = pd.DataFrame.from_dict(df_dict)
        path_save = os.path.join(dir_save,'loss.csv')
        df.to_csv(path_save)

        df_dict = dict()
        for t in self.validation_names:
            df_dict[t] = list(map(lambda tensor: tensor.numpy(),self.validation_losses[t]))
        df = pd.DataFrame.from_dict(df_dict)
        path_save = os.path.join(dir_save,'loss_validation.csv')
        df.to_csv(path_save)

        df_2 = pd.DataFrame(self.G_solv_hist.items())
        df_2.columns = ['iter','G_solv']
        path_save = os.path.join(dir_save,'G_solv.csv')
        df_2.to_csv(path_save)        

    ######################################################

    def set_mesh_names(self):
        for solver in self.solvers:
            solver.Mesh_names = solver.mesh.solver_mesh_names.union(self.mesh.domain_mesh_names)
        if 'E' in self.solver1.Mesh_names:
            self.solver1.Mesh_names.remove('E')

    def adapt_datasets(self):
        self.L_X_solvers = list()
        for solver in self.solvers:
            L_batches = dict()
            for t in solver.mesh.solver_mesh_names:
                L_batches[t] = solver.mesh.solver_mesh_data[t]
            self.L_X_solvers.append(L_batches)

        self.L_X_domain = dict()
        for t in self.mesh.domain_mesh_names:
            if t in ('Iu','Id'):
                self.L_X_domain['I'] = self.mesh.domain_mesh_data['I']
            else:
                self.L_X_domain[t] = self.mesh.domain_mesh_data[t]

        
    def get_batches(self, sample_method='random_sample', validation=False):

        if sample_method == 'full_batch':
            X_b1,X_b2 = self.L_X_solvers
            X_d = self.L_X_domain
        
        elif sample_method == 'random_sample':
            for i in range(2):
                for bl in self.mesh.mesh_objs[i].meshes.values():
                    type_b = bl['type']
                    if type_b in ('R','D','N','Q'): 
                        t = 'R' + str(i+1) if type_b=='R' else type_b
                        x = self.mesh.region_meshes[t].get_dataset()
                        x,y = self.mesh.mesh_objs[i].get_XU(x,bl)
                        self.L_X_solvers[i][t] = (x,y)
            X_b1,X_b2 = self.L_X_solvers

            self.L_X_domain['I'] = (self.mesh.region_meshes['I'].get_dataset(),self.mesh.region_meshes['I'].normals)
            X_d = self.L_X_domain

        if not validation:
            return X_b1,X_b2,X_d
            
        elif validation:
            X_vb1 = copy.deepcopy(X_b1)
            X_vb2 = copy.deepcopy(X_b2)
            X_vd = copy.deepcopy(X_d)
            for t in ('K','E','G'):
                if t in X_vb1:
                    del X_vb1[t]
                if t in X_vb2:
                    del X_vb2[t]
                if t in X_vd:
                    del X_vd[t]
            return X_vb1,X_vb2,X_vd

    #utils
    def checkers_iterations(self):

        # solvation energy
        if (self.iter+1)%self.G_solv_iter==0 and self.iter>1:
            self.calc_Gsolv_now = True
        else:
            self.calc_Gsolv_now = False

        # adapt losses weights
        if self.adapt_weights and (self.iter+1)%self.adapt_w_iter==0 and (self.iter+1)<self.N_iters and not self.precondition:
            self.adapt_w_now = True
        else:
            self.adapt_w_now = False  

        # check precondition
        if self.iter>=self.N_precond and self.precondition:
            self.precondition = False
            for data,solver in zip(self.L_X_solvers,self.solvers):
                del data['P']
                solver.mesh.solver_mesh_data['P'] = None
                solver.mesh.solver_mesh_names.remove('P')
                solver.Mesh_names.remove('P')
        
        if self.iter % 2 == 0:
            self.pbar.set_description("Loss: {:6.4e}".format(self.current_loss))                

    def callback(self, L1,L2):

        for net,L in zip(['1','2'],[L1,L2]):
            for t in self.losses_names:
                t2 = t[0]
                if t2 in ('R','N','D','K','P'):
                    if t[1] == net:
                        self.losses[t2+net].append(L[1][t2])

        for t in ('Q','Iu','Id','G','E'):
            self.losses[t].append(L1[1][t])

        loss = L1[0] + L2[0]
        self.losses['TL'].append(loss)
        self.current_loss = loss.numpy()
 
        self.solver1.L = L1[1]
        self.solver2.L = L2[1]

        for solver in self.solvers:
            for t in solver.L_names:
                solver.w_hist[t].append(solver.w[t])

        if self.save_model_iter > 0:
            if self.iter % self.save_model_iter == 0 and self.iter>1:
                dir_save = os.path.join(self.folder_path,f'iter_{self.iter}')
                self.save_models(dir_save, [f'model_1',f'model_2'])


    def callback_validation(self,L1,L2):
        for net,L in zip(['1','2'],[L1,L2]):
            for t in self.losses_names:
                t2 = t[0]
                if t2 in ('R','N','D'):
                    if t[1] == net:
                        self.validation_losses[t2+net].append(L[1][t2])
        for t in ('Q','Iu','Id'):
            self.validation_losses[t].append(L1[1][t])
        loss = L1[0] + L2[0]
        self.validation_losses['TL'].append(loss)


    def add_losses_NN(self):

        for solver,names,cont in zip(self.solvers,[self.losses_names_1,self.losses_names_2],['1','2']):
            for t in names:
                if t in ('R','N','D','K','P'):
                    solver.losses[t] = self.losses[t+cont]
                elif t != 'TL': 
                    solver.losses[t] = self.losses[t]
                if t in ('R','N','D'):
                    solver.validation_losses[t] = self.validation_losses[t+cont]
                elif t in ('Iu','Id','Q'): 
                    solver.validation_losses[t] = self.validation_losses[t]
              
        for solver in self.solvers:
            zipped_lists = zip(*solver.losses.values())
            solver.losses['TL'] = [sum(values) for values in zipped_lists]

            zipped_lists = zip(*(solver.losses[key] for key in solver.validation_losses.keys()))
            solver.losses['vTL'] = [sum(values) for values in zipped_lists]

            zipped_lists = zip(*solver.validation_losses.values())
            solver.validation_losses['TL'] = [sum(values) for values in zipped_lists]


