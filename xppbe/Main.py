import os

from xppbe.Simulation import Simulation


# Create simulation object

yaml_path = os.path.join(os.getcwd(),'Main.yaml')

simulation = Simulation(yaml_path)


if __name__=='__main__':
        # Create and solve simulation
        simulation.create_simulation()
        simulation.adapt_simulation()
        simulation.solve_model()
        simulation.postprocessing()


