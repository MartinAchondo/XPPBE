import numpy as np
import tensorflow as tf
from time import time
from tqdm import tqdm as log_progress

class XPINN():
    
    def __init__(self, PINN):

        self.DTYPE = 'float32'

        self.solver1, self.solver2 = PINN(), PINN()
        self.solvers = [self.solver1,self.solver2]
        
        self.loss_hist = list()

        self.loss_r1 = list()
        self.loss_bD1 = list()
        self.loss_bN1 = list()
        self.loss_bI1 = list()

        self.loss_r2 = list()
        self.loss_bD2 = list()
        self.loss_bN2 = list()
        self.loss_bI2 = list()

        self.iter = 0
        self.lr = None


    def adapt_PDEs(self,PDEs,unions) -> None:
        for solver,pde,union in zip(self.solvers,PDEs,unions):
            solver.adapt_PDE(pde)
            solver.un = union

    def adapt_meshes(self,meshes,weights):
        for solver,mesh,weight in zip(self.solvers,meshes,weights):
            solver.adapt_mesh(mesh,**weight)

    def create_NeuralNets(self,NN_class,lrs,hyperparameters):
        for solver,lr,hyperparameter in zip(self.solvers,lrs,hyperparameters):
            solver.create_NeuralNet(NN_class,lr,**hyperparameter)

    def load_NeuralNets(self,dirs,names,lrs):   
        for solver,lr,name in zip(self.solvers,lrs,names):
            solver.load_NeuralNet(dirs,name,lr)  

    def save_models(self,dirs,names):
        for solver,name in zip(self.solvers,names):
            solver.save_model(dirs,name)   


    def loss_I(self,solver,solver_ex):
        loss = 0
        for j in range(len(solver.XI_data)):
            x_i,y_i,z_i = solver.mesh.get_X(solver.XI_data[j])
            
            with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape1:
                tape1.watch(x_i)
                tape1.watch(y_i)
                tape1.watch(z_i)
                R = solver.mesh.stack_X(x_i,y_i,z_i)
                u_pred_1 = solver.model(R)
            ux_pred_1 = tape1.gradient(u_pred_1,x_i)
            uy_pred_1 = tape1.gradient(u_pred_1,y_i)
            uz_pred_1 = tape1.gradient(u_pred_1,z_i)

            del tape1

            with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape2:
                tape2.watch(x_i)
                tape2.watch(y_i)
                tape2.watch(z_i)
                R = solver_ex.mesh.stack_X(x_i,y_i,z_i)
                u_pred_2 = solver_ex.model(R)
            ux_pred_2 = tape2.gradient(u_pred_2,x_i)
            uy_pred_2 = tape2.gradient(u_pred_2,y_i)
            uz_pred_2 = tape2.gradient(u_pred_2,z_i)

            del tape2

            u_prom = (u_pred_1+u_pred_2)/2
            loss += tf.reduce_mean(tf.square(u_pred_1 - u_prom)) 
            
            norm_vn = tf.sqrt(x_i**2 + y_i**2 + z_i**2)
            v1 = (x_i*ux_pred_1 + y_i*uy_pred_1 + z_i*uz_pred_1)/norm_vn
            v2 = (x_i*ux_pred_2 + y_i*uy_pred_2 + z_i*uz_pred_2)/norm_vn
            loss += tf.reduce_mean(tf.square(v1*solver.un - v2*solver_ex.un))
            
        return loss

    def get_grad(self,solver,solver_ex):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(solver.model.trainable_variables)
            loss,L = solver.loss_fn()
            loss_I = solver.w_i*self.loss_I(solver,solver_ex)
            loss += loss_I 
            L['I'] = loss_I
        g = tape.gradient(loss, solver.model.trainable_variables)
        del tape
        return loss, L, g
    
    
    def solve_with_TFoptimizer(self, optimizer, N=1000):
        optimizer1,optimizer2 = optimizer

        @tf.function
        def train_step():
            loss1, L_loss1, grad_theta1 = self.get_grad(self.solver1,self.solver2)
            loss2, L_loss2, grad_theta2 = self.get_grad(self.solver2,self.solver1)

            optimizer1.apply_gradients(zip(grad_theta1, self.solver1.model.trainable_variables))
            optimizer2.apply_gradients(zip(grad_theta2, self.solver2.model.trainable_variables))

            L1 = [loss1,L_loss1]
            L2 = [loss2,L_loss2]
            return L1,L2
        
        pbar = log_progress(range(N))
        pbar.set_description("Loss: %s " % 100)
        for i in pbar:
            L1,L2 = train_step()
            self.callback(L1,L2)

            if self.iter % 10 == 0:
                pbar.set_description("Loss: {:6.4e}".format(self.current_loss))
        

    def callback(self, L1,L2):
        self.loss_r1.append(L1[1]['r'])
        self.loss_bD1.append(L1[1]['D'])
        self.loss_bN1.append(L1[1]['N'])
        self.loss_bI1.append(L1[1]['I'])

        self.loss_r2.append(L2[1]['r'])
        self.loss_bD2.append(L2[1]['D'])
        self.loss_bN2.append(L2[1]['N'])
        self.loss_bI2.append(L2[1]['I'])

        loss = L1[0] + L2[0]
        self.current_loss = loss.numpy()
        self.loss_hist.append(self.current_loss)
        self.iter+=1

    def add_losses_NN(self):
        self.solver1.loss_r = self.loss_r1
        self.solver1.loss_bD = self.loss_bD1
        self.solver1.loss_bN = self.loss_bN1
        self.solver1.loss_bI = self.loss_bI1

        self.solver2.loss_r = self.loss_r2
        self.solver2.loss_bD = self.loss_bD2
        self.solver2.loss_bN = self.loss_bN2
        self.solver2.loss_bI = self.loss_bI2


    def solve(self,N=1000):
        optim1 = tf.keras.optimizers.Adam(learning_rate=self.solver1.lr)
        optim2 = tf.keras.optimizers.Adam(learning_rate=self.solver2.lr)
        optim = [optim1,optim2]

        t0 = time()
        self.solve_with_TFoptimizer(optim, N)
        print('\nComputation time: {} seconds'.format(time()-t0))

        self.add_losses_NN()


if __name__=='__main__':
    pass

