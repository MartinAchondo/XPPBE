
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
To use this project, start by following the [Tutorial.ipynb](./tutorials/tutorial.ipynb) notebook, or modifying the [Main.py](./code/Main.py) template code. If you intend to run multiple simulations, add your customized `Main.py` files to the `code/simulations_list` folder. Finally, execute the following command:


```bash
bash Allrun
```

An explanation of a `Main.py` code is as follows:

1. Import the simulation object and initialize it:
    ```py
    from Simulation import Simulation
    simulation = Simulation(__file__)
    ```

2. Define the molecule, the properties and the equation to solve:
    ```py
    simulation.equation = 'standard'
    simulation.pbe_model = 'linear'

    simulation.domain_properties = {
            'molecule': 'born_ion',
            'epsilon_1':  1,
            'epsilon_2': 80,
            'kappa': 0.125,
            'T' : 300 
            }
    ```     
3. Define the number of collocation points (mesh properties):
    ```py
    simulation.mesh_properties = {
            'vol_max_interior': 0.04,
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

4. Define the different loss terms (solute domain, solvent domain and combination of boths)
    ```py
    simulation.losses = ['R1','R2','D2','I','K1','K2']
    ```
5. Define the architectures:
    ```py

    simulation.network = 'xpinn'
    simulation.hyperparameters_in = {
            'input_shape': (None,3),
            'num_hidden_layers': 4,
            'num_neurons_per_layer': 200,
            'output_dim': 1,
            'activation': 'tanh',
            'adaptative_activation': True,
            'architecture_Net': 'FCNN',
            'fourier_features': True,
            'num_fourier_features': 256
            }
    simulation.hyperparameters_out = {
            'input_shape': (None,3),
            'num_hidden_layers': 4,
            'num_neurons_per_layer': 200,
            'output_dim': 1,
            'activation': 'tanh',
            'adaptative_activation': True,
            'architecture_Net': 'FCNN',
            'fourier_features': False
            }
    ```

6. Finally, specify the optimization algorithm, the weights algorithm, the batches/samples approach and the number of iterations.
    ```py
    simulation.adapt_weights = True,
    simulation.adapt_w_iter = 1000
    simulation.adapt_w_method = 'gradients'
    simulation.alpha_w = 0.7           

    simulation.sample_method='random_sample'

    simulation.optimizer = 'Adam'
    simulation.lr = tf.keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=0.001,
                    decay_steps=2000,
                    decay_rate=0.9,
                    staircase=True)

    simulation.N_iters = 10000
    ```

7. Run the simulation:
    ```py
    simulation.create_simulation()
    simulation.adapt_simulation()
    simulation.solve_model()
    simulation.postprocessing()
    ```

8. For quick postprocessing, use the [Postprocessing Jupyter Notebook](./code/Post/post.ipynb).

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