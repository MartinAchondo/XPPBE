import os
os.environ['TF_RUN_EAGER_OP_AS_FUNCTION']='false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

xppbe_path = os.path.dirname(os.path.abspath(__file__))
scripts_path = os.path.join(xppbe_path,'Scripts')


def RunSimulation(yaml_path,results_path,molecule_dir,plot_mesh,known_method=None):
    sim = Simulation(yaml_path, results_path=results_path, molecule_dir=molecule_dir)
    sim.create_simulation()
    sim.adapt_model()
    sim.solve_model()
    sim.postprocessing(plot_all=True, mesh=plot_mesh, known_method=known_method)

def Allrun(sims_path, results_path, molecule_dir, plot_mesh=False, known_method=None):
    sims_path = os.path.abspath(sims_path)
    results_path = os.path.abspath(results_path)
    command = f"bash {scripts_path} Allrun --sims-path={sims_path} --results-path={results_path} --molecule-dir={molecule_dir if molecule_dir else ''}"
    if plot_mesh:
        command += " --mesh"
    if not known_method is None:
        command += f" --known-method={known_method}"
    os.system(command) 

def Allclean(results_path):
    results_path = os.path.abspath(results_path)
    command = f"bash {scripts_path} Allclean --results-path={results_path}"
    os.system(command) 

def SimsStatus(sims_path,results_path):
    sims_path = os.path.abspath(sims_path)
    results_path = os.path.abspath(results_path)
    command = f"bash {scripts_path} SimsStatus --sims-path={sims_path} --results-path={results_path}"
    os.system(command) 


from xppbe.Simulation import Simulation
import xppbe.Molecules
import xppbe.Model
import xppbe.Mesh
import xppbe.NN
import xppbe.Post
