import numpy as np
import scipy.optimize
import tensorflow as tf
from time import time
import logging
from tqdm import tqdm as log_progress

from xppbe.NN.XPINN_utils import XPINN_utils

class XPINN(XPINN_utils):
    
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)       
    
    def get_loss(self, X_batch, model, w, validation=False):
        loss = 0.0
        L = self.PDE.get_loss(X_batch, model, validation=validation)
        for t in self.mesh.domain_mesh_names:
            loss += w[t]*L[t]
        return loss,L

    def get_grad_loss(self,X_batch, model, trainable_variables, w):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(trainable_variables)
            loss,L = self.get_loss(X_batch, model, w)
        g = tape.gradient(loss, trainable_variables)
        del tape
        return loss, L, g
    
    def train_last_step(self, X, u, method='L-BFGS-B'):

        options={'maxiter': 50000,
            'maxfun': 50000,
            'maxcor': 50,
            'maxls': 50,
            'ftol': 1.0*np.finfo(float).eps}
        
        def get_weight_tensor():
            
            weight_list = []
            shape_list = []
            
            for v in self.model.variables:
                shape_list.append(v.shape)
                weight_list.extend(v.numpy().flatten())
                
            weight_list = tf.convert_to_tensor(weight_list)
            return weight_list, shape_list

        x0, shape_list = get_weight_tensor()
        
        def set_weight_tensor(weight_list):

            idx = 0
            for v in self.model.variables:
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
                    
                v.assign(tf.cast(new_val, DTYPE))
        
        def get_loss_and_grad(w):
            
            set_weight_tensor(w)

            loss, grad = self.get_grad(X, u)
                   
            loss = loss.numpy().astype(np.float64)
            self.current_loss = loss            
            
            grad_flat = []
            for g in grad:
                grad_flat.extend(g.numpy().flatten())
            
            grad_flat = np.array(grad_flat,dtype=np.float64)
            
            return loss, grad_flat
        
        
        return scipy.optimize.minimize(fun=get_loss_and_grad,
                                       x0=x0,
                                       jac=True,
                                       method=method,
                                       callback=self.callback,
                                       **options)
    
    def main_loop(self, N=1000):
        
        optimizer = self.create_optimizer()

        @tf.function
        def train_step(X_batch, ws):
            loss, L_loss, grad_theta = self.get_grad_loss(X_batch, self.model, self.model.trainable_variables, ws)
            optimizer.apply_gradients(zip(grad_theta, self.model.trainable_variables))
            del grad_theta
            L = [loss,L_loss]
            return L
        
        @tf.function
        def caclulate_validation_loss(X_v):
            loss,L_loss = self.get_loss(X_v,self.model,self.w, validation=True)
            L = [loss,L_loss]
            return L

        self.N_iters = N
        self.current_loss = 100

        self.create_losses_arrays(N)
        X_v = self.get_batches('full_batch', validation=True)
        X_d = self.get_batches(self.sample_method)

        self.pbar = log_progress(range(N))

        for i in self.pbar:

            self.checkers_iterations()

            if self.sample_method == 'random_sample':
                X_d = self.get_batches(self.sample_method)
            
            L = train_step(X_d, ws=self.w) 
            L_v = caclulate_validation_loss(X_v)

            self.iter+=1
            self.calculate_G_solv(self.calc_Gsolv_now)
            self.callback(L,L_v)
            self.check_adapt_new_weights(self.adapt_w_now)


    def check_adapt_new_weights(self,adapt_now):
        
        if adapt_now:
            X_d = self.get_batches(self.sample_method)
            self.modify_weights_by(self.model,X_d) 
            
    def modify_weights_by(self,model,X_domain):
        
        L = dict()
        if self.adapt_w_method == 'gradients':
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(model.trainable_variables)
                _,L_loss = self.get_loss(X_domain, model, self.w)

            for t in self.mesh.domain_mesh_names:
                loss = L_loss[t]
                grads = tape.gradient(loss, model.trainable_variables)
                grads = [grad if grad is not None else tf.zeros_like(var) for grad, var in zip(grads, model.trainable_variables)]
                gradient_norm = tf.sqrt(sum([tf.reduce_sum(tf.square(g)) for g in grads]))
                L[t] = gradient_norm
            del tape

        elif self.adapt_w_method == 'values':
            _,L = self.get_loss(X_domain, model, self.w) 

        eps = 1e-9
        loss_wo_w = sum(L.values())
        for t in self.mesh.domain_mesh_names:
            w = float(loss_wo_w/(L[t]+eps))
            self.w[t] = self.alpha_w*self.w[t] + (1-self.alpha_w)*w  

    def calculate_G_solv(self,calc_now):
        if calc_now:
            G_solv = self.PDE.get_solvation_energy(self.model)
            self.G_solv_hist[str(self.iter)] = G_solv   


    def solve(self,N=1000, save_model=0, G_solve_iter=100):

        self.save_model_iter = save_model if save_model != 0 else N

        self.G_solv_iter = G_solve_iter

        t0 = time()

        self.main_loop(N)

        logger = logging.getLogger(__name__)
        logger.info(f' Iterations: {self.iter}')
        logger.info(" Loss: {:6.4e}".format(self.current_loss))
        logger.info('Computation time: {} minutes'.format(int((time()-t0)/60)))
