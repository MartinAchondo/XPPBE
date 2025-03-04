
# XPPBE: PINN Solver for 3D Poisson-Boltzmann Equation 

![XPPBE](https://img.shields.io/badge/dynamic/toml?label=XPPBE&url=https%3A%2F%2Fraw.githubusercontent.com%2FMartinAchondo%2FXPPBE%2Fmaster%2Fpyproject.toml&query=%24.project.version&prefix=version%20&color=blue&logo=moleculer&logoColor=white)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/MartinAchondo/XPPBE/.github%2Fworkflows%2FCI.yml)
![Python version](https://img.shields.io/badge/dynamic/toml?url=https%3A%2F%2Fraw.githubusercontent.com%2FMartinAchondo%2FXPPBE%2Fmaster%2Fpyproject.toml&query=%24.project%5B'requires-python'%5D&logo=python&label=python&color=lightgrey)


Physics-Informed Neural Network solver for the Poisson-Boltzmann equation applied to real macromolecules in polarizable media.

$$\nabla^2 \phi_m = -\frac{1}{\epsilon_m}\sum_k q_k\delta(x_k) \quad x \in \Omega_m $$

$$\nabla^2 \phi_w = \kappa^2_w\phi_w \quad x \in \Omega_w $$


<!-- <p align="center">
  <img height="200" src="img/Implicit-solvent-tr.png">
</p> -->

<!-- <p align="center">
<picture>
    <source media="(prefers-color-scheme: dark)" srcset="img/Implicit-solvent-tr.png">
    <source media="(prefers-color-scheme: light)" srcset="img/Implicit-solvent.png">
    <img height="200" src="https://user-images.githubusercontent.com/25423296/163456779-a8556205-d0a5-45e2-ac17-42d089e3c3f8.png">
</picture>
</p> -->

## Features

- Solves the electrostatic potential for the solute and solvent domain.
- Simple molecule definition by .pdb or .pqr file.
- Different loss terms can be added to the model.
- Use of different architectures is available, very customizable.
- Loss balancing algorithm implemented.
- Build in Python/Tensorflow.

<p align="center">
  <img height="200" src="img/molecule.png">
</p>


## Resources

- [Papers](./papers/): Check this folder for relevant papers and additional project documentation.
- [Tutorials](./notebooks/tutorials/): Check this folder for notebooks tutorials.

## Installation

To install and run this project locally, follow these steps:

1. Clone the repository to your local machine.

   ```bash
   git clone https://github.com/MartinAchondo/XPPBE
   ```
2. Navigate to the project directory
   ```bash
   cd XPPBE
   ```
3. Create a virtual environment
   ```bash
   conda create --name xppbe python=3.9
   ```
4. Activate the virtual environment
   ```bash
   conda activate xppbe
   ```
5. Install the project
    ```bash
   pip install .
    ```

## Usage
To use this project, start by following the [Tutorial.ipynb](./tutorials/tutorial.ipynb) notebook. An explanation of this notebook is as follows:

1. Import the simulation object, the YAML file, and initialize it:
    ```py
    from xppbe import Simulation
    simulation = Simulation(yaml_path, molecule_dir)
    ```
2. Run the simulation:
    ```py
    simulation.create_simulation()
    simulation.adapt_model()
    simulation.solve_model()
    simulation.postprocessing(run_all=True)
    ```
The Simulation object import a YAML file with all the problem definitions. An explanation is as follows:

1. Define the molecule, the properties and the equation to solve:
    ```yaml
    equation: regularized_scheme_2
    pbe_model: linear
    pinns_method: DCM

    domain_properties:
        molecule: methanol
        epsilon_1: 1
        epsilon_2: 80
        kappa: 0.125
        T: 300
    ```     
2. Define the number of collocation points (mesh properties):
    ```yaml
    mesh_properties:
        vol_max_interior: 0.04
        vol_max_exterior: 0.1
        density_mol: 10
        density_border: 0.5
        mesh_generator: msms
        dR_exterior: 3
    ```

3. Define the different loss terms (solute domain, solvent domain and combination of both)
    ```yaml
    losses:
        - R1
        - R2
        - D2
        - Iu
        - Id
    ```
4. Define the architectures:
    ```yaml
    num_networks: 2

    hyperparameters_in:
        architecture_Net: FCNN
        num_hidden_layers: 4
        num_neurons_per_layer: 200
        activation: tanh
        adaptive_activation: true
        fourier_features: true
        weight_factorization: false
        scale_input: true
        scale_output: true

    hyperparameters_out:
        architecture_Net: FCNN
        num_hidden_layers: 4
        num_neurons_per_layer: 200
        activation: tanh
        adaptive_activation: true
        fourier_features: true
        weight_factorization: false
        scale_input: true
        scale_output: true
    ```

5. Finally, specify the optimization algorithm, the weights algorithm, the batches/samples approach and the number of iterations.
    ```yaml
    adapt_weights: true
    adapt_w_iter: 1000
    adapt_w_method: gradients
    alpha_w: 0.7         

    sample_method: random_sample
    
    optimizer: Adam
    lr:
        method: exponential_decay
        initial_learning_rate: 0.001
        decay_steps: 2000
        decay_rate: 0.9
        staircase: true

    N_iters: 20000
    ```

## Recent Publication

Our recent work investigates the application of Physics-Informed Neural Networks (PINNs) to solve the Poisson-Boltzmann equation (PBE) using XPPBE. In this study, we highlight the impact of incorporating advanced neural network features, such as input and output scaling, random Fourier features, trainable activation functions, and a loss balancing algorithm. 

<!-- Our findings show that these enhancements achieve accuracies of the order of 10⁻²—10⁻³, comparable to state-of-the-art methods. -->

- **Title**: An Investigation of Physics Informed Neural Networks to solve the Poisson-Boltzmann Equation in Molecular Electrostatics.
- **Authors**: Martín A. Achondo, Jehanzeb H. Chaudhry, Christopher D. Cooper.
- **Journal/Conference**: Under review.
- **DOI**: Pending.

All input files and scripts to reproduce the results are available in the [Paper Folder](./papers/PINN_PB_paper).


## Citing

If you find this project useful for your research or work, please consider citing it. Here is an example BibTeX entry:

```bibtex
@article{achondo2024investigation,
  title={An Investigation of Physics Informed Neural Networks to solve the Poisson-Boltzmann Equation in Molecular Electrostatics},
  author={Achondo, Martin A and Chaudhry, Jehanzeb H and Cooper, Christopher D},
  journal={arXiv preprint arXiv:2410.12810},
  year={2024}
}
```
