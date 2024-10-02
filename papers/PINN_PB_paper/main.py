import os
# os.environ['CUDA_VISIBLE_DEVICES']="-1"  # turn ON to run on the GPU

from xppbe import Simulation

yaml_path = 'path_to_input_file.yaml'   # example 'input_files/BI/BI_WA_TF_FF_SI_SO.yaml'
yaml_path = 'input_files/BI/BI_WA_TF_FF_SI_SO.yaml'

molecule_dir = None                     # Use default molecule directory if None, calling xppbe/Molecules/

results_path = f'results/{yaml_path.split("/")[-1].replace(".yaml","")}'   # example to use the name of the .yaml file

# Creation of the simulation object
simulation = Simulation(yaml_path, molecule_dir, results_path)

# Setup the PINN model and parameters from the YAML configuration
simulation.create_simulation()

# Adapt the architecture based on configuration options
simulation.adapt_model()

# Solve the model
simulation.solve_model()

# Postprocessing
post = simulation.postprocessing()
post.save = True

# Plots of loss history
post.plot_loss_history_total();
post.plot_loss_history_training_validation();

# Plot of solvation energy evolution
post.plot_G_solv_history();

# Different plots of the reaction field
import numpy as np
post.plot_phi_line(value='react', theta=0, phi=np.pi/2);
post.plot_phi_contour(value='react');
post.plot_interface_3D(value='react', ext='png');
post.plot_interface_3D(value='react', ext='html');

# Save relevant information
post.save_values_file()
post.save_model_summary();

# To compare to a known method, use known_method as 'analytic_Born_Ion', 'Spherical_Harmonics', 'PBJ', 'APBS'
known_method = 'analytic_Born_Ion'
post.values_for_paper(err_method=known_method)
post.plot_G_solv_history(known_method);
post.plot_phi_line_aprox_known(known_method, value='react',theta=0, phi=np.pi/2)
post.plot_phi_line_aprox_known(known_method, value='react',theta=np.pi/2, phi=np.pi/2)
post.plot_phi_line_aprox_known(known_method, value='react', theta=np.pi/2, phi=np.pi)
post.plot_interface_error(known_method, type_e='absolute', scale='log', ext='png')
post.plot_interface_error(known_method, type_e='absolute', scale='linear', ext='png')



