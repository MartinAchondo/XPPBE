
from xppbe.Simulation import Simulation


simulation = Simulation('Main.yaml')

simulation.create_simulation()
simulation.adapt_simulation()
simulation.solve_model()
simulation.postprocessing()
