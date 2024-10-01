# Reproducing Paper Results

This folder contains all the necessary files and configurations to reproduce the results from our recent publication: *An Investigation of Physics Informed Neural Networks to solve the Poisson-Boltzmann Equation in Molecular Electrostatics*.

## How to Run the Simulations

1. **Set Up the Environment**: Make sure you have installed all dependencies as specified in the main [README](../../README.md).
   
2. **Choose the YAML Configuration File**: Select the desired configuration file from this [directory](./input_files/) (e.g., `input_files/BI/BI_WA_TF_FF_SI_SO.yaml`).

3. **Run the Simulation**: Use the example Python [script](./main.py) to start the simulation, specifying the chosen YAML file.

4. **Postprocess the Results**: After running the simulation, check the output directory for plots, model summaries, and saved results.
