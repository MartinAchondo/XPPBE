import tensorflow as tf
from time import time
import logging
from tqdm import tqdm as log_progress

from NN.XPINN_utils import XPINN_utils

class XPINN(XPINN_utils):
    
    def __init__(self):
        super().__init__()       
    
    def get_loss(self, X_batch, model, w, precond=False, validation=False):
        loss = 0.0
        if not precond:
            L = self.PDE.get_loss(X_batch, model, validation=validation)
            for t in self.mesh.domain_mesh_names:
                loss += w[t]*L[t]
            return loss,L
        elif precond:
            L = self.PDE.get_loss_preconditioner(X_batch, model)
            return L['P1']+L['P2'],L

    def get_grad_loss(self,X_batch, model, trainable_variables, w, precond=False):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(trainable_variables)
            loss,L = self.get_loss(X_batch, model, w, precond)
        g = tape.gradient(loss, trainable_variables)
        del tape
        return loss, L, g
    
    def main_loop(self, N=1000, N_precond=10):
        
        if not self.two_optimizers:
            optimizer = self.create_optimizer()
        else: 
            optimizer_1,optimizer_2 = self.create_optimizer()
        if self.precondition:
            optimizer_P = self.create_optimizer(precond=True)

        @tf.function
        def train_step(X_batch, ws,precond=False):
            loss, L_loss, grad_theta = self.get_grad_loss(X_batch, self.model, self.model.trainable_variables, ws, precond)
            optimizer.apply_gradients(zip(grad_theta, self.model.trainable_variables))
            L = [loss,L_loss]
            return L
        
        @tf.function
        def train_step_2opt(X_batch, ws,precond=False):
            loss_1, L_loss_1, grad_theta_1 = self.get_grad_loss(X_batch, self.model, self.model.NNs[0].trainable_variables, ws, precond)
            loss_2, L_loss_2, grad_theta_2 = self.get_grad_loss(X_batch, self.model, self.model.NNs[1].trainable_variables, ws, precond)
            optimizer_1.apply_gradients(zip(grad_theta_1, self.model.NNs[0].trainable_variables))
            optimizer_2.apply_gradients(zip(grad_theta_2, self.model.NNs[1].trainable_variables))
            L = [loss_1,L_loss_1]
            return L

        @tf.function
        def train_step_precond(X_batch, ws, precond=True):
            loss, L_loss, grad_theta = self.get_grad_loss(X_batch, self.model, self.model.trainable_variables, ws, precond)
            optimizer_P.apply_gradients(zip(grad_theta, self.model.trainable_variables))
            L = [loss,L_loss]
            return L
        
        @tf.function
        def caclulate_validation_loss(X_v, precond):
            loss,L_loss = self.get_loss(X_v,self.model,self.w, precond=precond, validation=True)
            L = [loss,L_loss]
            return L

        self.N_iters = N
        self.N_precond = N_precond
        self.current_loss = 100

        self.create_losses_arrays(N)
        X_v = self.get_batches('full_batch', validation=True)
        X_d = self.get_batches(self.sample_method)

        self.pbar = log_progress(range(N))

        for i in self.pbar:

            if self.sample_method == 'random_sample':
                X_d = self.get_batches(self.sample_method)
                
            self.checkers_iterations()
            
            if self.precondition:
                L = train_step_precond(X_d, ws=self.w)

            elif not self.two_optimizers:
                L = train_step(X_d, ws=self.w)   
            
            elif self.two_optimizers:
                L = train_step_2opt(X_d, ws=self.w)

            self.iter+=1
            L_v = caclulate_validation_loss(X_v, self.precondition)
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


    def solve(self,N=1000, precond=False, N_precond=10, save_model=0, G_solve_iter=100):

        self.precondition = precond
        self.save_model_iter = save_model if save_model != 0 else N

        self.G_solv_iter = G_solve_iter

        t0 = time()

        self.main_loop(N, N_precond)

        logger = logging.getLogger(__name__)
        logger.info(f' Iterations: {self.iter}')
        logger.info(" Loss: {:6.4e}".format(self.current_loss))
        logger.info('Computation time: {} minutes'.format(int((time()-t0)/60)))
