import os
from xppbe.Simulation import Simulation
yaml_path = os.path.join(os.getcwd(),'Main.yaml')

simulation = Simulation(yaml_path)

simulation.create_simulation()
simulation.adapt_simulation()
simulation.solve_model()
simulation.postprocessing()
