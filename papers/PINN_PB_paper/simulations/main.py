import os
# os.environ['CUDA_VISIBLE_DEVICES']="-1"  # turn ON to run on the GPU

from xppbe import Simulation

yaml_path = 'path_to_input_file.yaml'   # example 'input_files/BI/BI_WA_TF_FF_SI_SO.yaml'
yaml_path = 'input_files/BI/BI_WA_TF_FF_SI_SO.yaml'
molecule_dir = None

# Creation of the simulation object
simulation = Simulation(yaml_path, molecule_dir)

# Setup of the algorithm
simulation.create_simulation()
simulation.adapt_model()

# Solve the model
simulation.solve_model()

# Postprocessing
post = simulation.postprocessing()
post.save = True

# Plot of loss history
post.plot_loss_history_total();
post.plot_loss_hisotry_training_validation();

# Plot of solvation energy evolution
post.plot_G_solv_history();

# Different plots of the reaction field
post.plot_phi_line(value='react');
post.plot_phi_contour(value='react');
post.plot_interface_3D(value='react', ext='png');

# Save relevant information
post.save_values_file()
post.save_model_summary();





