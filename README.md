
# XPINN Solver for 3D Poisson-Boltzmann Equation 

Simple Python Poisson-Boltzmann equation solver for real macromolecules in a polarizable media, using Extended Physics Informed Neural Networks. 

$$\nabla^2 \phi_1 = -\frac{1}{\epsilon_1}\sum_k q_k\delta(x_k) \quad x \in \Omega_1 $$

$$\nabla^2 \phi_2 = \kappa^2\phi_2 \quad x \in \Omega_2 $$


<p align="center">
  <img height="200" src="img/Implicit-solvent-tr.png">
</p>

## Features

- Solves the electrostatic potential for the solute and solvent domain.
- Simple molecule definition by .pqr file.
- Different loss terms can be added to the model.
- Use of different architectures is available, very customizable.
- Weigths balancing algortithm implemented.
- Build in Python/Tensorflow.

<p align="center">
  <img height="200" src="img/molecule.png">
</p>


## Resources

- [Documents](./documents/): Check this folder for relevant papers and additional project documentation.
- [Tutorials](./tutorials/): Check this folder for notebooks tutorials.

## Installation

To install and run this project locally, follow these steps:

1. Clone the repository to your local machine.

   ```bash
   git clone https://github.com/MartinAchondo/XPINN-for-PBE-Simulation
   ```
2. Navigate to the project directory
   ```bash
   cd XPINN-for-PBE-Simulation
    ```
3. Install project dependecies:
    ```bash
   poetry install
    ```

## Usage
To use this project, start by following the [Tutorial.ipynb](./tutorials/tutorial.ipynb) notebook, or modifying the [Main_Molecule.py](./code/Main_Molecule.py) or the [Main_BornIon.py](./code/Main_BornIon.py) template codes. If you intend to run multiple simulations, add your customized `Main.py` files to the `code/simulations_list` folder. Finally, execute the following command:


```bash
bash run_simulations.bash
```

An explanation of a `Main.py` code is as follows:

1. Define the molecule, the properties and the equation to solve:
    ```py
    equation = 'regularized_scheme_1'

    inputs = {'molecule': 'methanol',
              'epsilon_1':  1,
              'epsilon_2': 80,
              'kappa': 0.125,
              'T' : 300 
              }
    ```     
2. Define the number of collocation points (mesh properties):
    ```py
    N_points = {'vol_max_interior': 0.04,
                'vol_max_exterior': 0.1,
                'density_mol': 40,
                'density_border': 4,
                'dx_experimental': 2,
                'N_pq': 100,
                'G_sigma': 0.04,
                'mesh_generator': 'msms',
                'dR_exterior': 8
                }
    ```

3. Define the different loss terms (solute domain, solvent domain and combination of boths)
    ```py
    self.meshes_domain = dict()   
    self.meshes_domain['1'] = {'domain': 'molecule', 'type':'R1', 'value':0.0}
    self.meshes_domain['3'] = {'domain': 'molecule', 'type':'K1', 'file':'data_known_reg.dat'}
    self.meshes_domain['5'] = {'domain': 'solvent', 'type':'R2', 'value':0.0}
    self.meshes_domain['6'] = {'domain': 'solvent', 'type':'D2', 'fun':lambda x,y,z: self.PBE_model.border_value(x,y,z)}
    self.meshes_domain['7'] = {'domain': 'solvent', 'type':'K2', 'file':'data_known.dat'}
    self.meshes_domain['9'] = {'domain': 'solvent', 'type': 'E2', 'file': 'data_experimental.dat'}
    self.meshes_domain['10'] = {'domain':'interface', 'type':'I'}
    self.meshes_domain['11'] = {'domain':'interface', 'type':'G'}
    ```
4. Define the architectures:
    ```py
    hyperparameters_in = {
                    'input_shape': (None,3),
                    'num_hidden_layers': 4,
                    'num_neurons_per_layer': 20,
                    'output_dim': 1,
                    'activation': 'tanh',
                    'adaptative_activation': True,
                    'architecture_Net': 'FCNN',
                    'fourier_features': True,
                    'num_fourier_features': 12,
                    'scale': XPINN_solver.mesh.scale_1
            }

    hyperparameters_out = {
                    'input_shape': (None,3),
                    'num_hidden_layers': 4,
                    'num_neurons_per_layer': 20,
                    'output_dim': 1,
                    'activation': 'tanh',
                    'adaptative_activation': True,
                    'architecture_Net': 'FCNN',
                    'fourier_features': False,
                    'scale': XPINN_solver.mesh.scale_2
            }
    ```

5. Finally, specify the optimization algorithm, the weights algorithm and the batches/samples approach.
    ```py
    XPINN_solver.adapt_weights([weights,weights],
                                adapt_weights = True,
                                adapt_w_iter = 20,
                                adapt_w_method = 'gradients',
                                alpha = 0.7)             

    XPINN_solver.set_points_methods(sample_method='random_sample')

    optimizer = 'Adam'
    lr = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=2000,
            decay_rate=0.9,
            staircase=True)
    XPINN_solver.adapt_optimizers(optimizer,lr)
    ```

6. And solve the PDE:
    ```py
    N_iters = 10000

    XPINN_solver.solve(N=N_iters, 
                    precond = False, 
                    save_model = 0, 
                    G_solve_iter=100)
    ```

7. For quick postprocessing, use the Postprocessing class:
    ```py
    Post = Postprocessing(XPINN_solver, save=True, directory=folder_path)
    ```
    Or the [Postprocessing Jupyter Notebook](./code/Post/post.ipynb).

## Citing

If you find this project useful for your research or work, please consider citing it. Here is an example BibTeX entry:

```bibtex
@misc{XPINN-for-PBE,
  author    = {Martín Achondo},
  title     = {XPINN Solver for 3D Poisson-Boltzmann Equation},
  howpublished = {GitHub repository},
  year      = {2024},
  url       = {https://github.com/MartinAchondo/XPINN-for-PBE-Simulation},
}
```