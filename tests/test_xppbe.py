import os
import tempfile
import pytest
import csv
import shutil
from xppbe import Simulation
from xppbe import Allrun,Allclean
import subprocess

import xppbe

def test_bash():
    print(xppbe.Molecules)
    os.system(f"bash sims.bash")
    #print("\n\n\n")
    #subprocess.Popen("pip list")


def run_checkers(sim,sim_name,temp_dir):

    results_path = os.path.join(temp_dir,'results')
    assert os.path.isdir(results_path)
    assert len(os.listdir(results_path)) > 0

    results_path = os.path.join(results_path,sim_name)
    for name in ['iterations','mesh','plots_losses','plots_model','plots_solution','plots_weights','Post.ipynb','results_values.json',f'{sim_name}.yaml']:
        assert name in os.listdir(results_path)

    mesh_path = os.path.join(results_path,'mesh')
    assert len(os.listdir(mesh_path)) == 10 or len(os.listdir(mesh_path)) == 11 

    iterations_path = os.path.join(results_path,'iterations')
    assert len(os.listdir(iterations_path)) == 1

    last_iteration = os.path.join(iterations_path,f'iter_{sim.N_iters}')
    listdir_last_iteration = os.listdir(last_iteration)

    for file_name in ['optimizer.npy', 'weights.index', 'w_hist.csv', 'checkpoint', 'loss.csv', 'hyperparameters.json', 'G_solv.csv', 'loss_validation.csv', 'weights.data-00000-of-00001']:
        assert file_name in listdir_last_iteration

    for file_name in ['loss.csv','loss_validation.csv','w_hist.csv','G_solv.csv']:
        file = os.path.join(last_iteration,file_name)
        with open(file, 'r') as csvfile:
            reader = csv.reader(csvfile)
            lines = list(reader)
            assert len(lines) == sim.N_iters + 1
            last_line = lines[-1]
            assert not '' in last_line
    
    print('Checkers Passed!')

def test_scripts():
    with tempfile.TemporaryDirectory() as temp_dir:
        sim_name = f'test_born_ion'
        yaml_path = os.path.join(os.path.dirname(__file__),'simulations_yaml',sim_name+'.yaml')
        sims_path = os.path.join(temp_dir,'sims')
        os.mkdir(sims_path)
        shutil.copy(yaml_path,os.path.join(sims_path,sim_name+'.yaml'))
        Allrun(sims_path=sims_path, results_path=temp_dir)
        sim = Simulation(yaml_path, results_path=temp_dir)
        run_checkers(sim,sim_name,temp_dir)
        Allclean(results_path=temp_dir)
        assert len(os.listdir(os.path.join(temp_dir,'results'))) == 0