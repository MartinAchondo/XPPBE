import json
import logging

from DCM.Mesh import Mesh
from NN.NeuralNet import PINN_NeuralNet

from NN.PINN import PINN
from NN.XPINN import XPINN

from Post.Postprocessing import View_results
from Post.Postprocessing import View_results_X

logger = logging.getLogger(__name__)



class Simulation():
    
    def __init__(self, PDE):
          self.problem = None
          self.mesh = None
          self.weights = None
          self.lr = None
          self.hyperparameters = None
          self.PDE_Interface = PDE
          self.precondition = False
          self.N_batches = 1

    def setup_algorithm(self):
        
        logger.info("> Starting PINN Algorithm")
        logger.info(json.dumps(self.problem, indent=4))
        logger.info(json.dumps({'q': self.q}))
        
        PDE_in = self.PDE_in
        domain_in = PDE_in.set_domain(self.domain_in)

        PDE_out = self.PDE_out
        domain_out = PDE_out.set_domain(self.domain_out)
   

        logger.info(f'Batches: {self.N_batches}')
        mesh_in = Mesh(domain_in, mesh_N=self.mesh_in, N_batches=self.N_batches, precondition=self.precondition)
        mesh_in.create_mesh(self.extra_meshes_in, self.ins_domain_in)
        logger.info(json.dumps({'Domain_Mesh_in': self.mesh_in}, indent=4))
        L = dict()
        for t in self.extra_meshes_in:
            L[self.extra_meshes_in[t]['type']] = self.extra_meshes_in[t]['N']
        logger.info(json.dumps({'Extra_Meshes_in': L}, indent=4))
        # mesh_in.plot_points_2d(self.folder_path, 'Mesh_2d_in')

        mesh_out = Mesh(domain_out, mesh_N=self.mesh_out, N_batches=self.N_batches, precondition=self.precondition)
        mesh_out.create_mesh(self.extra_meshes_out, self.ins_domain_out)
        logger.info(json.dumps({'Domain_Mesh_out': self.mesh_out}, indent=4))
        L = dict()
        for t in self.extra_meshes_out:
            L[self.extra_meshes_out[t]['type']] = self.extra_meshes_out[t]['N']
        logger.info(json.dumps({'Extra_Meshes_out': L}, indent=4))
        # mesh_out.plot_points_2d(self.folder_path, 'Mesh_2d_out')

        PDE = self.PDE_Interface()
        PDE.adapt_PDEs([PDE_in,PDE_out],[PDE_in.epsilon,PDE_out.epsilon])
        PDE.epsilon_G = PDE_in.epsilon_G
        PDE.q = PDE_in.q
        PDE.problem = self.problem

        self.XPINN_solver = XPINN(PINN)
        self.XPINN_solver.adapt_PDEs(PDE)

        self.XPINN_solver.adapt_meshes([mesh_in,mesh_out],[self.weights,self.weights])

        self.XPINN_solver.create_NeuralNets(PINN_NeuralNet,[self.lr,self.lr],[self.hyperparameters_in,self.hyperparameters_out])
        
        logger.info(json.dumps({'hyperparameters in': self.hyperparameters_in}, indent=4))
        logger.info(json.dumps({'hyperparameters out': self.hyperparameters_out}, indent=4))
        logger.info(json.dumps({'weights': self.weights}, indent=4))
        logger.info(json.dumps({'Learning Rate': self.lr}))

        self.XPINN_solver.folder_path = self.folder_path


    def solve_algorithm(self,N_iters, precond=False, N_precond=10, save_model=0, adapt_weights=False, adapt_w_iter=1, shuffle = True, shuffle_iter = 500):
        logger.info("> Solving XPINN")
        if precond:
            logger.info(f'Preconditioning {N_precond} iterations')
        if save_model != 0:
            logger.info(f'Save model every {save_model} iterations')
        if shuffle:
            logger.info(f'Shuffling batches every {shuffle_iter} iterations')
        if adapt_weights:
            logger.info(f'Adapting weights every {adapt_w_iter} iterations')
        self.XPINN_solver.solve(N=N_iters, 
                                precond=precond, 
                                N_precond=N_precond,  
                                save_model=save_model, 
                                adapt_weights=adapt_weights, 
                                adapt_w_iter=adapt_w_iter,
                                shuffle = shuffle, 
                                shuffle_iter = shuffle_iter )


    def postprocessing(self,folder_path):
        
        Post = View_results_X(self.XPINN_solver, View_results, save=True, directory=folder_path, data=False)
        logger.info("> Ploting Solution")

        Post.plot_loss_history();
        Post.plot_loss_history(plot_w=True);
        Post.plot_u_plane();
        Post.plot_u_domain_contour();
        Post.plot_aprox_analytic();
        Post.plot_interface();
        Post.plot_weights_history();

        if Post.data:
            Post.close_file()

        logger.info('================================================')
