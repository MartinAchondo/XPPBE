import os
import tempfile
import pytest
import csv
import shutil
from xppbe import Simulation
from xppbe import Allrun,Allclean

def run_checkers(sim,sim_name,temp_dir,num_iters=1):

    results_path = os.path.join(temp_dir,'results')
    assert os.path.isdir(results_path)
    assert len(os.listdir(results_path)) > 0

    results_path = os.path.join(results_path,sim_name)
    for name in ['iterations','mesh','plots_losses','plots_model','plots_solution','plots_weights','Post.ipynb','results_values.json',f'{sim_name}.yaml']:
        assert name in os.listdir(results_path)

    mesh_path = os.path.join(results_path,'mesh')
    assert len(os.listdir(mesh_path)) > 0 

    iterations_path = os.path.join(results_path,'iterations')
    assert len(os.listdir(iterations_path)) == num_iters

    last_iteration = os.path.join(iterations_path,f'iter_{int(sim.N_iters+sim.N_steps_2*sim.options_optimizer2["maxiter"])}')
    listdir_last_iteration = os.listdir(last_iteration)

    for file_name in ['optimizer.npy', 'weights.index', 'w_hist.csv', 'checkpoint', 'loss.csv', 'hyperparameters.json', 'G_solv.csv', 'loss_validation.csv', 'weights.data-00000-of-00001']:
        assert file_name in listdir_last_iteration

    for file_name in ['loss.csv','loss_validation.csv','w_hist.csv','G_solv.csv']:
        file = os.path.join(last_iteration,file_name)
        with open(file, 'r') as csvfile:
            reader = csv.reader(csvfile)
            lines = list(reader)
            assert len(lines) == int(sim.N_iters+sim.N_steps_2*sim.options_optimizer2["maxiter"]) + 1
            last_line = lines[-1]
            assert not '' in last_line
    
    print('Checkers Passed!')


def test_scripts():
    with tempfile.TemporaryDirectory() as temp_dir:
        sim_name = f'test_born_ion'
        yaml_path = os.path.join(os.path.dirname(__file__),'simulations_yaml',sim_name+'.yaml')
        sims_path = os.path.join(temp_dir,'sims')
        results_path = os.path.join(temp_dir,'results',sim_name)
        os.mkdir(sims_path)
        shutil.copy(yaml_path,os.path.join(sims_path,sim_name+'.yaml'))
        Allrun(sims_path=sims_path, results_path=temp_dir, molecule_dir=None)
        sim = Simulation(yaml_path, results_path=results_path, molecule_dir=None)
        run_checkers(sim,sim_name,temp_dir)
        Allclean(results_path=temp_dir)
        assert len(os.listdir(os.path.join(temp_dir,'results'))) == 0
    


@pytest.mark.parametrize(
 ('pinns_method','model','scheme'),
 (
     ('DCM','linear','direct'),
     ('DCM','linear', 'regularized_scheme_1'),
     ('DCM','nonlinear','regularized_scheme_2'),
     ('DVM','linear','direct'),
     ('DVM','linear','regularized_scheme_1'),
     ('DVM','nonlinear','regularized_scheme_1'),
     ('DBM','linear','direct')
 )       
)
def test_pinns_method_and_schemes(pinns_method,model,scheme):
    with tempfile.TemporaryDirectory() as temp_dir:
        sim_name = f'test_{pinns_method}_{scheme}'
        yaml_path = os.path.join(os.path.dirname(__file__),'simulations_yaml',sim_name+'.yaml')
        yaml_prev_path = os.path.join(os.path.dirname(__file__),'simulations_yaml','test_born_ion.yaml')
        shutil.copy(yaml_prev_path,yaml_path)
        results_path = os.path.join(temp_dir,'results',sim_name)
        sim = Simulation(yaml_path, results_path=results_path, molecule_dir=None)
        sim.pinns_method = pinns_method
        sim.pbe_model = model
        sim.equation = scheme
        if pinns_method == 'DBM':
            sim.num_networks = 1
            sim.losses = ['IB1','IB2']
        sim.create_simulation()
        sim.adapt_model()
        sim.solve_model()
        sim.postprocessing(run_all=True)
        run_checkers(sim,sim_name,temp_dir)


@pytest.mark.parametrize(
 ('molecule'),
 (
     ('methanol'),
     ('arg')
 )       
)
def test_other_molecules(molecule):
    with tempfile.TemporaryDirectory() as temp_dir:
        sim_name = f'test_{molecule}'
        yaml_path = os.path.join(os.path.dirname(__file__),'simulations_yaml',sim_name+'.yaml')
        results_path = os.path.join(temp_dir,'results',sim_name)
        sim = Simulation(yaml_path, results_path=results_path, molecule_dir=None)
        sim.create_simulation()
        sim.adapt_model()
        sim.solve_model()
        sim.postprocessing(run_all=True)
        run_checkers(sim,sim_name,temp_dir)


def test_optimizers():
    with tempfile.TemporaryDirectory() as temp_dir:
        sim_name = f'test_optimizers'
        yaml_path = os.path.join(os.path.dirname(__file__),'simulations_yaml',sim_name+'.yaml')
        yaml_prev_path = os.path.join(os.path.dirname(__file__),'simulations_yaml','test_born_ion.yaml')
        shutil.copy(yaml_prev_path,yaml_path)
        results_path = os.path.join(temp_dir,'results',sim_name)
        sim = Simulation(yaml_path, results_path=results_path, molecule_dir=None)
        sim.options_optimizer2['maxiter'] = 3
        sim.N_steps_2 = 2
        sim.create_simulation()
        sim.adapt_model()
        sim.solve_model()
        sim.postprocessing(run_all=True)
        run_checkers(sim,sim_name,temp_dir,num_iters=2)


@pytest.mark.parametrize(
 ('arch'),
 (
     ('ModMLP'),
     ('ResNet')
 )       
)
def test_other_architectures(arch):
    with tempfile.TemporaryDirectory() as temp_dir:
        sim_name = f'test_{arch}'
        yaml_path = os.path.join(os.path.dirname(__file__),'simulations_yaml',sim_name+'.yaml')
        yaml_prev_path = os.path.join(os.path.dirname(__file__),'simulations_yaml','test_born_ion.yaml')
        shutil.copy(yaml_prev_path,yaml_path)
        results_path = os.path.join(temp_dir,'results',sim_name)
        sim = Simulation(yaml_path, results_path=results_path, molecule_dir=None)
        sim.hyperparameters_in['architecture_Net'] = arch
        sim.create_simulation()
        sim.adapt_model()
        sim.solve_model()
        sim.postprocessing(run_all=True)
        run_checkers(sim,sim_name,temp_dir)



@pytest.mark.parametrize(
 ('loss'),
 (
     ('K2'),
     ('E2'),
     ('G')
 )       
)
def test_additional_losses(loss):
    with tempfile.TemporaryDirectory() as temp_dir:
        sim_name = f'test_{loss}'
        yaml_path = os.path.join(os.path.dirname(__file__),'simulations_yaml',sim_name+'.yaml')
        yaml_prev_path = os.path.join(os.path.dirname(__file__),'simulations_yaml','test_born_ion.yaml')
        shutil.copy(yaml_prev_path,yaml_path)
        results_path = os.path.join(temp_dir,'results',sim_name)
        sim = Simulation(yaml_path, results_path=results_path, molecule_dir=None)
        sim.losses.append(loss)
        if loss == 'E2':
            sim.mesh_properties['dR_exterior'] = 5
        sim.create_simulation()
        sim.adapt_model()
        sim.solve_model()
        sim.postprocessing(run_all=True)
        run_checkers(sim,sim_name,temp_dir)


def test_mesh_post():
    with tempfile.TemporaryDirectory() as temp_dir:
        sim_name = f'test_mesh_post'
        yaml_path = os.path.join(os.path.dirname(__file__),'simulations_yaml',sim_name+'.yaml')
        yaml_prev_path = os.path.join(os.path.dirname(__file__),'simulations_yaml','test_born_ion.yaml')
        shutil.copy(yaml_prev_path,yaml_path)
        results_path = os.path.join(temp_dir,'results',sim_name)
        sim = Simulation(yaml_path, results_path=results_path, molecule_dir=None)
        sim.create_simulation()
        sim.adapt_model()
        sim.solve_model()
        sim.postprocessing(run_all=True, plot_mesh=True)
        run_checkers(sim,sim_name,temp_dir)


def test_iteration_continuation():
    with tempfile.TemporaryDirectory() as temp_dir:
        sim_name = 'test_iteration'
        yaml_path = os.path.join(os.path.dirname(__file__),'simulations_yaml',sim_name+'.yaml')
        yaml_prev_path = os.path.join(os.path.dirname(__file__),'simulations_yaml','test_born_ion.yaml')
        shutil.copy(yaml_prev_path,yaml_path)
        results_path = os.path.join(temp_dir,'results',sim_name)
        sim = Simulation(yaml_path, results_path=results_path, molecule_dir=None)
        sim.create_simulation()
        sim.adapt_model()
        sim.solve_model()
        sim.postprocessing(run_all=True)

        assert os.path.isdir(results_path)
        assert len(os.listdir(results_path)) > 0

        iterations_path = os.path.join(results_path,'iterations')
        assert len(os.listdir(iterations_path)) == 1

        sim_2 = Simulation(yaml_path, results_path=results_path, molecule_dir=None)
        sim_2.N_iters = sim.N_iters*2
        sim_2.create_simulation()
        sim_2.adapt_model()
        sim_2.solve_model()
        sim_2.postprocessing(run_all=True)

        last_iteration = os.path.join(iterations_path,f'iter_{sim_2.N_iters}')

        for file_name in ['loss.csv','loss_validation.csv','w_hist.csv','G_solv.csv']:
            file = os.path.join(last_iteration,file_name)
            with open(file, 'r') as csvfile:
                reader = csv.reader(csvfile)
                lines = list(reader)
                assert len(lines) == sim.N_iters*2 + 1
                last_line = lines[-1]
                assert not '' in last_line
