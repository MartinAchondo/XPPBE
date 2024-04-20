from Simulation import Simulation
import tensorflow as tf

# Create simulation object
simulation = Simulation(__file__)

# Equation to solve
simulation.equation = 'regularized_scheme_2'
simulation.pbe_model = 'linear'

# Domain properties
simulation.domain_properties = {
        'molecule': 'arg',
        'epsilon_1':  1,
        'epsilon_2': 80,
        'kappa': 0.125,
        'T' : 300 
        }

# Mesh properties
simulation.mesh_properties = {
        'vol_max_interior': 1.1,
        'vol_max_exterior': 2.5,
        'density_mol': 1.5,
        'density_border': 3,
        'dx_experimental': 2,
        'N_pq': 100,
        'G_sigma': 0.04,
        'mesh_generator': 'msms',
        'dR_exterior': 3
        }
simulation.sample_method='random_sample'

# Frecuency of solvation energy calculation
simulation.G_solve_iter=6

# Losses to add, and initial weights
simulation.losses = ['R1','R2','D2','Iu','Id','K2']
simulation.weights = {
        'E2': 10**-10,
        }

# Weights adapting algorithm inputs
simulation.adapt_weights = True,
simulation.adapt_w_iter = 1000
simulation.adapt_w_method = 'gradients'
simulation.alpha_w = 0.7

# Architecture
simulation.network = 'xpinn'
simulation.hyperparameters_in = {
        'input_shape': (None,3),
        'num_hidden_layers': 2,
        'num_neurons_per_layer': 15,
        'output_dim': 1,
        'activation': 'tanh',
        'adaptative_activation': True,
        'architecture_Net': 'FCNN',
        'fourier_features': True,
        'num_fourier_features': 256
        }
simulation.hyperparameters_out = {
        'input_shape': (None,3),
        'num_hidden_layers': 2,
        'num_neurons_per_layer': 15,
        'output_dim': 1,
        'activation': 'tanh',
        'adaptative_activation': True,
        'architecture_Net': 'FCNN',
        'fourier_features': True
        }

# Scale charges
simulation.scale_NN_q = True

# Optimizer properties
simulation.optimizer = 'Adam'
simulation.lr = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=0.001,
                decay_steps=2000,
                decay_rate=0.9,
                staircase=True)
simulation.lr_p = 0.001

# Solve parameters
simulation.N_iters = 2

simulation.precondition = False
simulation.N_precond = 0
simulation.iters_save_model = 5000


if __name__=='__main__':
        # Create and solve simulation
        simulation.create_simulation()
        simulation.adapt_simulation()
        simulation.solve_model()
        simulation.postprocessing()


