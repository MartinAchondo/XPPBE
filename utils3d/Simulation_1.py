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

    def setup_algorithm(self):
        
        logger.info("> Starting PINN Algorithm")
        logger.info(json.dumps(self.problem, indent=4))
        logger.info(json.dumps({'q': self.q}))
        
        PDE_in = self.PDE_in
        domain_in = PDE_in.set_domain(self.domain_in)

        self.N_b = self.mesh['N_b']
        self.N_r = self.mesh['N_r']        
        mesh_in = Mesh(domain_in, N_b=self.N_b, N_r=self.N_r)
        mesh_in.create_mesh(self.borders_in, self.ins_domain_in)

        PDE = self.PDE_EQ()
        PDE.epsilon_G = PDE_in.epsilon_G
        PDE.q = PDE_in.q
        PDE.problem = self.problem

        self.PINN_solver = PINN()

        self.PINN_solver.adapt_PDE(PDE)

        logger.info(json.dumps({'Mesh': self.mesh}, indent=4))
        self.PINN_solver.adapt_mesh(mesh_in,**self.weights)

        self.PINN_solver.create_NeuralNet(PINN_NeuralNet,self.lr,**self.hyperparameters_in)
        
        logger.info(json.dumps({'hyperparameters in': self.hyperparameters_in}, indent=4))
        logger.info(json.dumps({'weights': self.weights}, indent=4))
        logger.info(json.dumps({'Learning Rate': self.lr}))


    def solve_algorithm(self,N_iters, precond=False, N_precond=10):
        logger.info("> Solving PINN")
        if precond:
            logger.info(f'Preconditioning {N_precond} iterations')
        self.PINN_solver.solve(N=N_iters, precond=precond, N_precond=N_precond)


    def postprocessing(self,folder_path):
        
        Post = View_results(self.PINN_solver, save=True, directory=folder_path, data=True)

        logger.info("> Ploting Solution")

        Post.plot_loss_history();
        Post.plot_u_plane();
        #Post.plot_u_domain_contour();
        Post.plot_aprox_analytic();

        if Post.data:
            Post.close_file()

        logger.info('================================================')
