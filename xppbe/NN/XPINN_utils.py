import os
import copy
import json
import numpy as np
import pandas as pd
import tensorflow as tf

class XPINN_utils():

    DTYPE='float32'

    def __init__(self, results_path):

        self.results_path = results_path
        self.losses_names = ['TL','TL1','TL2','vTL1','vTL2','R1','D1','N1','K1','Q1','R2','D2','N2','K2','G','Iu','Id','Ir','E2','P1','P2']
        self.losses_names_1 = ['TL1','R1','D1','N1','K1','Q1','Iu','Id','Ir','G','P1']
        self.losses_names_2 = ['TL2','R2','D2','N2','K2','Iu','Id','Ir','E2','G','P2']
        self.validation_names = ['TL','TL1','TL2','R1','D1','N1','Q1','R2','D2','N2','Iu','Id','Ir']
        self.w_names = self.losses_names[5:]
        self.losses_names_list = [self.losses_names_1,self.losses_names_2]
        
        self.losses = dict()
        self.validation_losses = dict()

        self.G_solv_hist = dict()
        self.G_solv_hist['0'] = 0.0

        self.iter = 0


    def create_NeuralNet(self, NN_class, hyperparameters, *args, **kwargs):
        self.hyperparameters = hyperparameters
        self.model = NN_class(hyperparameters, *args, **kwargs)
        self.model.build_Net()

    def adapt_optimizer(self,optimizer,lr,optimizer2=False,options2=None):
        self.optimizer_name = optimizer
        if lr['method']=='exponential_decay':
            self.lr = tf.keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=lr['initial_learning_rate'],
                    decay_steps=lr['decay_steps'],
                    decay_rate=lr['decay_rate'],
                    staircase=lr['staircase'])

        if optimizer2:
            self.optimizer_2_name = optimizer2
            self.optimizer_2_opts = options2
            self.use_optimizer_2 = True
        else:
            self.use_optimizer_2 = False

    def adapt_PDE(self,PDE):
        self.PDE = PDE
        self.mesh = self.PDE.mesh
        self.adapt_datasets()
        print("PDEs and datasets ready")

    def adapt_weights(self,weights,adapt_weights=False,adapt_w_iter=2000,adapt_w_method='gradients',alpha_w=0.7):
        self.adapt_weights = adapt_weights
        self.adapt_w_iter = adapt_w_iter
        self.adapt_w_method = adapt_w_method
        self.alpha_w = alpha_w

        self.w = dict()
        for w_name in self.w_names:
            if not w_name in weights:
                self.w[w_name] = 1.0
            else:
                self.w[w_name] = weights[w_name]

        self.w_hist = dict()

    def set_points_methods(self, sample_method='batches', N_batches=1, sample_size=1000):
        self.sample_method = sample_method
        self.N_batches = N_batches
        self.sample_size = sample_size

    def adapt_datasets(self):
        self.L_X_domain = dict()
        for t in self.mesh.domain_mesh_names:
            if t in ('Iu','Id','Ir'):
                self.L_X_domain['I'] = self.mesh.domain_mesh_data['I']
            else:
                self.L_X_domain[t] = self.mesh.domain_mesh_data[t]

    def create_losses_arrays(self, N):
        for t in self.losses_names:
            self.losses[t] = np.zeros(N, dtype=self.DTYPE)
        for t in self.validation_names:
            self.validation_losses[t] = np.zeros(N, dtype=self.DTYPE)
        for t in self.w:
            self.w_hist[t] = np.zeros(N, dtype=self.DTYPE)

    def extend_losses_arrays(self, N):
        N_extend = N - len(self.losses['TL'])
        for t in self.losses_names:
            self.losses[t] = np.concatenate((self.losses[t],np.zeros(N_extend, dtype=self.DTYPE)))
        for t in self.validation_names:
            self.validation_losses[t] = np.concatenate((self.validation_losses[t],np.zeros(N_extend, dtype=self.DTYPE)))
        for t in self.w:
            self.w_hist[t] = np.concatenate((self.w_hist[t],np.zeros(N_extend, dtype=self.DTYPE)))


    ##############################################################################################
                
    def create_optimizer(self, starting_point):
        
        if self.optimizer_name == 'Adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
            
        if starting_point == 'continue':
            zero_grads = [tf.zeros_like(w) for w in self.model.trainable_variables]
            saved_vars = [tf.identity(w) for w in self.model.trainable_variables]
            optimizer.apply_gradients(zip(zero_grads, self.model.trainable_variables))
            [x.assign(y) for x,y in zip(self.model.trainable_variables, saved_vars)]
            optimizer.set_weights(self.loaded_opt_weights)
            
        return optimizer

    def get_batches(self, sample_method='random_sample', validation=False):

        if sample_method == 'full_batch':
            X_d = self.L_X_domain
        
        elif sample_method == 'random_sample':
            for bl in self.mesh.meshes_info.values():
                type_b = bl['type']
                flag = bl['domain']
                if type_b[0] in ('R','D','N','Q'): 
                    x = self.mesh.region_meshes[type_b].get_dataset()
                    x,y = self.mesh.get_XU(x,bl)
                    self.L_X_domain[type_b] = ((x,y),flag)

            self.L_X_domain['I'] = ((self.mesh.region_meshes['I'].get_dataset(),self.mesh.region_meshes['I'].normals),'interface')
            X_d = self.L_X_domain

        if not validation:
            return X_d
            
        elif validation:
            X_vd = copy.deepcopy(X_d)
            for t in ('K1','K2','E2','G'):
                if t in X_vd:
                    del X_vd[t]
            return X_vd

    def get_weight_tensor(self):
        weight_list = []
        shape_list = []
        for v in self.model.trainable_variables:
            shape_list.append(v.shape)
            weight_list.extend(v.numpy().flatten())
        weight_list = tf.convert_to_tensor(weight_list)
        return weight_list, shape_list

    def set_weight_tensor(self,weight_list):
        idx = 0
        for v in self.model.trainable_variables:
            vs = v.shape
            if len(vs) == 2:  
                sw = vs[0]*vs[1]
                new_val = tf.reshape(weight_list[idx:idx+sw],(vs[0],vs[1]))
                idx += sw
            elif len(vs) == 1:
                new_val = weight_list[idx:idx+vs[0]]
                idx += vs[0]
            elif len(vs) == 0:
                new_val = weight_list[idx]
                idx += 1
            v.assign(tf.cast(new_val, self.DTYPE))


    #utils
    def checkers_iterations(self):

        # solvation energy
        if (self.iter)%self.G_solv_iter==0 and self.iter>1:
            self.calc_Gsolv_now = True
        else:
            self.calc_Gsolv_now = False

        # adapt losses weights
        if self.adapt_weights and (self.iter)%self.adapt_w_iter==0 and (self.iter)<self.N_iters:
            self.adapt_w_now = True
        else:
            self.adapt_w_now = False  
              


    def callback(self, L_loss,Lv_loss):

        loss,L = L_loss
        lossv,Lv = Lv_loss
        i = self.iter - 1

        for t in self.losses_names:
            if not 'TL' in t:
                self.losses[t][i] = L[t].numpy()
                if t in self.validation_names:
                    self.validation_losses[t][i] = Lv[t].numpy()

        self.losses['TL'][i] = loss.numpy()
        self.validation_losses['TL'][i] = lossv.numpy()
        self.current_loss = loss.numpy()

        loss1 = 0.0
        loss1v = 0.0
        loss1_vl = 0.0
        for t in self.losses_names_1:
            if t != 'TL1':
                loss1 += L[t]
                if t in self.validation_names:
                    loss1v += L[t]
                    loss1_vl += Lv[t]
        self.losses['TL1'][i] = loss1.numpy()
        self.losses['vTL1'][i] = loss1v.numpy()
        self.validation_losses['TL1'][i] = loss1_vl.numpy()

        loss2 = 0.0
        loss2v = 0.0
        loss2_vl = 0.0
        for t in self.losses_names_2:
            if t != 'TL2':
                loss2 += L[t]
                if t in self.validation_names:
                    loss2v += L[t]
                    loss2_vl += Lv[t]
        self.losses['TL2'][i] = loss2.numpy()
        self.losses['vTL2'][i] = loss2v.numpy()
        self.validation_losses['TL2'][i] = loss2_vl.numpy()
        
        for t in self.w_names:
            self.w_hist[t][i] = self.w[t]

        if self.save_model_iter > 0:
            if (self.iter % self.save_model_iter == 0 and self.iter>1) or self.iter==self.N_iters or self.iter==self.N_iters+self.N_iters_2:
                dir_save = os.path.join(self.results_path,'iterations',f'iter_{self.iter}')
                self.save_model(dir_save)


    ##############################################################################################

    def load_NeuralNet(self,NN_class,dir_load,iter_path,Iter):   

        path = os.path.join(dir_load,'iterations',iter_path)

        with open(os.path.join(path,'hyperparameters.json'), "r") as json_file:
            hyper = json.load(json_file)
        hyperparameters = [hyper['Molecule_NN'],hyper['Solvent_NN']]
        self.create_NeuralNet(NN_class, hyperparameters)

        self.model.load_weights(os.path.join(path,'weights'))
        self.iter = Iter
        
        path_load = os.path.join(path,'w_hist.csv')
        df = pd.read_csv(path_load)
        self.w_hist,self.w = dict(),dict()
        for t in self.w_names:
            self.w_hist[t] = np.array(df[t], dtype=self.DTYPE)
            self.w[t] = float(self.w_hist[t][self.iter-1])
       
        path_load = os.path.join(path,'loss.csv')
        df = pd.read_csv(path_load)
        for t in self.losses_names:
            self.losses[t] = np.array(df[t])

        path_load = os.path.join(path,'loss_validation.csv')
        df = pd.read_csv(path_load)
        for t in self.validation_names:
            self.validation_losses[t] = np.array(df[t])

        path_load = os.path.join(path,'G_solv.csv')
        df_2 = pd.read_csv(path_load)
        for iters,G_solv in zip(list(df_2['iter']),list(df_2['G_solv'])):
            self.G_solv_hist[str(iters)] = G_solv

        path_load = os.path.join(path, 'optimizer.npy')
        self.loaded_opt_weights = np.load(path_load, allow_pickle=True)


    def save_model(self,dir_save):

        if not os.path.exists(dir_save):
            os.makedirs(dir_save)
        self.model.save_weights(os.path.join(dir_save,'weights'))

        df = pd.DataFrame.from_dict(self.w_hist)
        path_save = os.path.join(dir_save,'w_hist.csv')
        df.to_csv(path_save)

        df = pd.DataFrame.from_dict(self.losses)
        path_save = os.path.join(dir_save,'loss.csv')
        df.to_csv(path_save)

        df = pd.DataFrame.from_dict(self.validation_losses)
        path_save = os.path.join(dir_save,'loss_validation.csv')
        df.to_csv(path_save)
        
        df_2 = pd.DataFrame(self.G_solv_hist.items())
        df_2.columns = ['iter','G_solv']
        path_save = os.path.join(dir_save,'G_solv.csv')
        df_2.to_csv(path_save)  

        path_save = os.path.join(dir_save,'hyperparameters.json')
        with open(path_save, "w") as json_file:
            json.dump({'Molecule_NN': self.hyperparameters[0], 'Solvent_NN': self.hyperparameters[1]}, json_file, indent=4)     

        np.save(os.path.join(dir_save, 'optimizer'), np.array(self.optimizer.get_weights(), dtype=object))

