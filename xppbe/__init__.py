import os

os.environ['TF_RUN_EAGER_OP_AS_FUNCTION']='false'

xppbe_path = os.path.dirname(os.path.abspath(__file__))
scripts_path = os.path.join(xppbe_path,'Scripts')

def Allrun(sims_path, results_path, plot_mesh=True, plot_pbj=False):
    sims_path = os.path.abspath(sims_path)
    results_path = os.path.abspath(results_path)
    command = f"bash {scripts_path} Allrun --sims-path={sims_path} --results-path={results_path}"
    if plot_mesh:
        command += " --mesh"
    if plot_pbj:
        command += " --pbj"
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
