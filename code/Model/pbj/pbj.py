
from Model.pbj.electrostatics.solute import Solute
from Model.pbj.electrostatics.simulation import Simulation

class pbj():

    def __init__(self,domain_properties,pqr_path,mesh_density,mesh_generator):

        mol_object = Solute(pqr_path, mesh_density=mesh_density, mesh_generator=mesh_generator, save_mesh_build_files=False)
        self.simulation = Simulation()
        self.simulation.add_solute(mol_object)

        self.simulation.solutes[0].ep_in = domain_properties['epsilon_1']
        self.simulation.solutes[0].ep_ex = domain_properties['epsilon_2']
        self.simulation.solutes[0].kappa = domain_properties['kappa']

        self.simulation.calculate_surface_potential()

    def calculate_potential(self,X,domain):

        if domain == 'molecule':
            phir_solute, _ = self.simulation.calculate_reaction_potential_solute(X, units='e_eps0_angs')
            return phir_solute

        elif domain == 'solvent':
            phi_solv, _ = self.simulation.calculate_potential_solvent(X,units='e_eps0_angs')
            return phi_solv
    