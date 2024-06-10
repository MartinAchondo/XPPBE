import os
import shutil


class APBS():

    def __init__(self,domain_properties,equation,pqr_path):

        self.apbs_path = os.path.dirname(os.path.abspath(__file__))

        with open(os.path.join(self.apbs_path,'config.in'), 'r') as config_file:
            text = config_file.read()
        
        self.molecule = domain_properties['molecule']
        text = text.replace('$MOLECULE',self.molecule)
        if equation == 'linear':
            equation = 'lpbe'
        elif equation == 'nonlinear':
            equation = 'npbe'
        text = text.replace('$PBE_EQUATION',equation)
        text = text.replace('$EPSILON_1',str(float(domain_properties['epsilon_1'])))
        text = text.replace('$EPSILON_2',str(float(domain_properties['epsilon_2'])))
        text = text.replace('$TEMPERATURE',str(float(domain_properties['T'])))

        self.apbs_file = text

        original_dir = os.getcwd()

        self.path_files = os.path.join(self.apbs_path,'Temp')
        if os.path.exists(self.path_files):
            shutil.rmtree(self.path_files)
        os.makedirs(self.path_files)

        shutil.copy(pqr_path,self.path_files)
        mol_file_name = f'{domain_properties["molecule"]}.in'
        with open(os.path.join(self.path_files,mol_file_name), 'w') as mol_file:
            mol_file.write(text)
        
        os.chdir(self.path_files)
        os.system(f"apbs {mol_file_name} --output-file=results.txt")
        os.chdir(original_dir)


    def calculate_potential(self,X):
        
        from gridData import Grid 
        phi_total = Grid(os.path.join(self.path_files,f"phi_total_{self.molecule}.dx"))
        phi_vacuum = Grid(os.path.join(self.path_files,f"phi_vacuum_{self.molecule}.dx"))

        phi_react = phi_total - phi_vacuum

        phi = phi_react.interpolated(X[:,0],X[:,1],X[:,2])

        return phi
   

    def calculate_solvation_energy(self):

        import re
        energy_patterns = {
            'phi': re.compile(r'totEnergy\s+(\S+)\s+kJ/mol', re.MULTILINE),
            'vacuum': re.compile(r'totEnergy\s+(\S+)\s+kJ/mol', re.MULTILINE),
        }

        with open(os.path.join(self.path_files,'results.txt'), 'r') as file:
            log_content = file.read()

        energies = dict()
        E_total_pbe = float(energy_patterns['phi'].findall(log_content)[0])
        E_vacuum = float(energy_patterns['vacuum'].findall(log_content)[1])

        return (E_total_pbe - E_vacuum)/4.184

