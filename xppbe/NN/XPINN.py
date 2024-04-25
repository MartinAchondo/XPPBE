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
    

    def train_sgd(self, X_d, X_v):

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

        for i in range(self.N_iters):

            if self.sample_method == 'random_sample':
                X_d = self.get_batches(self.sample_method)

            L = train_step(X_d, ws=self.w) 
            L_v = caclulate_validation_loss(X_v)
            self.complete_callback(L,L_v)


    def train_newton(self, X_batch, X_batch_val):

        def train_step(X_batch,X_batch_val):

            def get_loss_grad(w):

                self.set_weight_tensor(w)
                loss, _ , grad_theta = self.get_grad_loss(X_batch, self.model, self.model.trainable_variables, self.w)

                grad_flat = np.concatenate([g.numpy().flatten() for g in grad_theta], axis=0).astype(np.float64)

                return loss, grad_flat
            
            def callback_ts(w):
                self.set_weight_tensor(w)
                L = self.get_loss(X_batch,self.model,self.w)
                L_v = self.get_loss(X_batch_val,self.model,self.w, validation=True)
                self.complete_callback(L,L_v)
            
            x0, _ = self.get_weight_tensor()
            scipy.optimize.minimize(
                            fun=get_loss_grad,
                            x0=x0,
                            jac=True,
                            method=self.optimizer_2_name,
                            options=self.optimizer_2_opts,
                            callback=callback_ts)

            
        for i in range(self.N_steps_2):

            if self.sample_method == 'random_sample':
                X_batch = self.get_batches(self.sample_method)
            
            train_step(X_batch,X_batch_val)
            
            
    def main_loop(self, N=1000, N2=0):
        
        self.N_iters = N
        self.N_steps_2 = N2

        if self.use_optimizer_2:
            self.N_iters_2 = self.N_steps_2 * self.optimizer_2_opts['maxiter'] 

        N_total = self.N_iters + self.N_iters_2
        self.pbar = log_progress(range(N_total))

        self.create_losses_arrays(N_total)
        X_v = self.get_batches('full_batch', validation=True)
        X_d = self.get_batches(self.sample_method)

        self.train_sgd(X_d, X_v)

        if self.use_optimizer_2:
            self.train_newton(X_d,X_v)   


    def complete_callback(self, L,L_v):

        self.iter+=1
        self.checkers_iterations()
        self.calculate_G_solv(self.calc_Gsolv_now)
        self.callback(L,L_v)
        self.check_adapt_new_weights(self.adapt_w_now)

        self.pbar.update()
        if self.iter % 2 == 0:
            opt_name = self.optimizer_name if self.iter<=self.N_iters else self.optimizer_2_name
            self.pbar.set_description("{} loop, Loss: {:6.4e}".format(opt_name, self.current_loss))  


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


    def solve(self,N=1000, N2=0, save_model=0, G_solve_iter=100):

        self.save_model_iter = save_model if save_model != 0 else N
        self.G_solv_iter = G_solve_iter
        t0 = time()

        self.main_loop(N,N2)

        logger = logging.getLogger(__name__)
        logger.info(f' Iterations: {self.iter}')
        logger.info(" Loss: {:6.4e}".format(self.current_loss))
        logger.info('Computation time: {} minutes'.format(int((time()-t0)/60)))
