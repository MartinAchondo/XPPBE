import json
import logging

from DCM.Mesh import Mesh
from NN.NeuralNet import PINN_NeuralNet

from NN.PINN import PINN

from DCM.Postprocessing import View_results

logger = logging.getLogger(__name__)



class Simulation():
    
    def __init__(self, PDE):
          self.problem = None
          self.mesh = None
          self.weights = None
          self.lr = None
          self.hyperparameters = None
          self.PDE_EQ = PDE
          self.precondition = False

    def setup_algorithm(self):
        
        logger.info("> Starting PINN Algorithm")
        logger.info(json.dumps(self.problem, indent=4))
        
        PDE_in = self.PDE_in
        domain_in = PDE_in.set_domain(self.domain_in)
      
        mesh_in = Mesh(domain_in, mesh_N=self.mesh, precondition=self.precondition)
        mesh_in.create_mesh(self.borders_in)
        mesh_in.plot_points_2d(self.folder_path, 'Mesh_2d_in')
        #mesh_in.plot_points_3d(self.folder_path, 'Mesh_3d_in')

        PDE = self.PDE_EQ()
        PDE.rho = PDE_in.rho
        PDE.mu = PDE_in.mu
        PDE.problem = self.problem

        self.PINN_solver = PINN()

        self.PINN_solver.adapt_PDE(PDE)

        logger.info(json.dumps({'Mesh': self.mesh}, indent=4))
        self.PINN_solver.adapt_mesh(mesh_in,**self.weights)

        self.PINN_solver.create_NeuralNet(PINN_NeuralNet,self.lr,**self.hyperparameters_in)
        
        logger.info(json.dumps({'hyperparameters in': self.hyperparameters_in}, indent=4))
        logger.info(json.dumps({'weights': self.weights}, indent=4))
        logger.info(json.dumps({'Learning Rate': self.lr}))

        self.PINN_solver.folder_path = self.folder_path


    def solve_algorithm(self,N_iters, precond=False, N_precond=10, N_batches=1, save_model=0):
        logger.info("> Solving PINN")
        if precond:
            logger.info(f'Preconditioning {N_precond} iterations')
        logger.info(f'Number Batches: {N_batches}')
        self.PINN_solver.solve(N=N_iters, precond=precond, N_precond=N_precond, N_batches=N_batches, save_model=save_model)


    def postprocessing(self,folder_path):
        
        Post = View_results(self.PINN_solver, save=True, directory=folder_path, data=False)

        logger.info("> Ploting Solution")

        Post.plot_loss_history();
        Post.plot_u_plane(tv=0);
        Post.plot_u_plane(tv=1);

        if Post.data:
            Post.close_file()

        logger.info('================================================')
