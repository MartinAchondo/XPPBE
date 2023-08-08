import json
import logging

from DCM.Mesh import Mesh
from NN.NeuralNet import PINN_NeuralNet

from NN.PINN import PINN
from NN.XPINN import XPINN

from DCM.Postprocessing import View_results
from DCM.Postprocessing import View_results_X

logger = logging.getLogger(__name__)



class Simulation():
    
    def __init__(self, PDE):
          self.problem = None
          self.mesh = None
          self.weights = None
          self.lr = None
          self.hyperparameters = None
          self.PDE_Interface = PDE

    def setup_algorithm(self):
        
        logger.info("> Starting PINN Algorithm")
        logger.info(json.dumps(self.problem, indent=4))
        logger.info(json.dumps({'q': self.q}))
        
        PDE_in = self.PDE_in
        domain_in = PDE_in.set_domain(self.domain_in)

        PDE_out = self.PDE_out
        domain_out = PDE_out.set_domain(self.domain_out)

        self.N_b = self.mesh['N_b']
        self.N_r = self.mesh['N_r']        
        mesh_in = Mesh(domain_in, N_b=self.N_b, N_r=self.N_r)
        mesh_in.create_mesh(self.borders_in, self.ins_domain_in)
        
        mesh_out = Mesh(domain_out, N_b=self.N_b, N_r=self.N_r)
        mesh_out.create_mesh(self.borders_out, self.ins_domain_out)

        PDE = self.PDE_Interface()
        PDE.adapt_PDEs([PDE_in,PDE_out],[PDE_in.epsilon,PDE_out.epsilon])
        PDE.epsilon_G = PDE_in.epsilon_G
        PDE.q = PDE_in.q
        PDE.problem = self.problem

        self.XPINN_solver = XPINN(PINN)

        self.XPINN_solver.adapt_PDEs(PDE)

        logger.info(json.dumps({'Mesh': self.mesh}, indent=4))
        self.XPINN_solver.adapt_meshes([mesh_in,mesh_out],[self.weights,self.weights])

        self.XPINN_solver.create_NeuralNets(PINN_NeuralNet,[self.lr,self.lr],[self.hyperparameters_in,self.hyperparameters_out])
        
        logger.info(json.dumps({'hyperparameters in': self.hyperparameters_in}, indent=4))
        logger.info(json.dumps({'hyperparameters out': self.hyperparameters_out}, indent=4))
        logger.info(json.dumps({'weights': self.weights}, indent=4))
        logger.info(json.dumps({'Learning Rate': self.lr}))


    def solve_algorithm(self,N_iters, precond=False, N_precond=10):
        logger.info("> Solving XPINN")

        self.XPINN_solver.solve(N=N_iters, precond=precond, N_precond=N_precond)


    def postprocessing(self,folder_path):
        
        Post = View_results_X(self.XPINN_solver, View_results, save=True, directory=folder_path, data=True)

        logger.info("> Ploting Solution")

        Post.plot_loss_history();
        Post.plot_u_plane();
        Post.plot_u_domain_contour();
        Post.plot_aprox_analytic();
        Post.plot_interface();

        if Post.data:
            Post.close_file()

        logger.info('================================================')
