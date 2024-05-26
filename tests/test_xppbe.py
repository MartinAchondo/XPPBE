import os
import tempfile
import pytest
import csv
import shutil
from xppbe.Simulation import Simulation


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


@pytest.mark.parametrize(
 ('molecule'),
 (
     ('born_ion'),
     ('sphere_+1-1')
     ('methanol'),
     ('arg')
 )       
)
def test_xppbe_solver(molecule):
    with tempfile.TemporaryDirectory() as temp_dir:
        sim_name = f'test_{molecule}'
        yaml_path = os.path.join(os.path.dirname(__file__),'simulations_yaml',sim_name+'.yaml')
        sim = Simulation(yaml_path, results_path=temp_dir)
        sim.create_simulation()
        sim.adapt_model()
        sim.solve_model()
        sim.postprocessing(mesh=False, pbj=False)
        run_checkers(sim,sim_name,temp_dir)


@pytest.mark.parametrize(
 ('loss'),
 (
     ('K2'),
     ('E2'),
     ('G'),
     ('Ir')
 )       
)
def test_additional_losses(loss):
    with tempfile.TemporaryDirectory() as temp_dir:
        sim_name = f'test_{loss}'
        yaml_path = os.path.join(os.path.dirname(__file__),'simulations_yaml',sim_name+'.yaml')
        yaml_prev_path = os.path.join(os.path.dirname(__file__),'simulations_yaml','test_born_ion.yaml')
        shutil.copy(yaml_prev_path,yaml_path)
        sim = Simulation(yaml_path, results_path=temp_dir)
        sim.losses.append(loss)
        if loss == 'E2':
            sim.mesh_properties['dR_exterior'] = 5
        sim.create_simulation()
        sim.adapt_model()
        sim.solve_model()
        sim.postprocessing(mesh=False, pbj=False)
        run_checkers(sim,sim_name,temp_dir)


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
        sim = Simulation(yaml_path, results_path=temp_dir)
        sim.hyperparameters_in['architecture_Net'] = arch
        sim.create_simulation()
        sim.adapt_model()
        sim.solve_model()
        sim.postprocessing(mesh=False, pbj=False)
        run_checkers(sim,sim_name,temp_dir)


@pytest.mark.parametrize(
 ('model','scheme'),
 (
     ('nonlinear','regularized_scheme_2'),
     ('linear', 'regularized_scheme_1'),
     ('linear','standard')
 )       
)
def test_non_linear_and_schemes(model,scheme):
    with tempfile.TemporaryDirectory() as temp_dir:
        sim_name = f'test_{model}_{scheme}'
        yaml_path = os.path.join(os.path.dirname(__file__),'simulations_yaml',sim_name+'.yaml')
        yaml_prev_path = os.path.join(os.path.dirname(__file__),'simulations_yaml','test_born_ion.yaml')
        shutil.copy(yaml_prev_path,yaml_path)
        sim = Simulation(yaml_path, results_path=temp_dir)
        sim.pbe_model = model
        sim.equation = scheme
        sim.create_simulation()
        sim.adapt_model()
        sim.solve_model()
        sim.postprocessing(mesh=False, pbj=False)
        run_checkers(sim,sim_name,temp_dir)


def test_mesh_post():
    with tempfile.TemporaryDirectory() as temp_dir:
        sim_name = f'test_mesh_post'
        yaml_path = os.path.join(os.path.dirname(__file__),'simulations_yaml',sim_name+'.yaml')
        yaml_prev_path = os.path.join(os.path.dirname(__file__),'simulations_yaml','test_born_ion.yaml')
        shutil.copy(yaml_prev_path,yaml_path)
        sim = Simulation(yaml_path, results_path=temp_dir)
        sim.create_simulation()
        sim.adapt_model()
        sim.solve_model()
        sim.postprocessing(mesh=True, pbj=False)
        run_checkers(sim,sim_name,temp_dir)


def test_iteration_continuation():
    with tempfile.TemporaryDirectory() as temp_dir:
        sim_name = f'test_iteration'
        yaml_path = os.path.join(os.path.dirname(__file__),'simulations_yaml',sim_name+'.yaml')
        yaml_prev_path = os.path.join(os.path.dirname(__file__),'simulations_yaml','test_born_ion.yaml')
        shutil.copy(yaml_prev_path,yaml_path)
        sim = Simulation(yaml_path, results_path=temp_dir)
        sim.create_simulation()
        sim.adapt_model()
        sim.solve_model()
        sim.postprocessing(mesh=False, pbj=True)

        results_path = os.path.join(temp_dir,'results')
        assert os.path.isdir(results_path)
        assert len(os.listdir(results_path)) > 0

        results_path = os.path.join(results_path,'test_born_ion')
        iterations_path = os.path.join(results_path,'iterations')
        assert len(os.listdir(iterations_path)) == 1

        sim_2 = Simulation(yaml_path, results_path=temp_dir)
        sim_2.N_iters = sim.N_iters*2
        sim_2.create_simulation()
        sim_2.adapt_model()
        sim_2.solve_model()
        sim_2.postprocessing(mesh=False, pbj=False)

        last_iteration = os.path.join(iterations_path,f'iter_{sim_2.N_iters}')

        for file_name in ['loss.csv','loss_validation.csv','w_hist.csv','G_solv.csv']:
            file = os.path.join(last_iteration,file_name)
            with open(file, 'r') as csvfile:
                reader = csv.reader(csvfile)
                lines = list(reader)
                assert len(lines) == sim.N_iters*2 + 1
                last_line = lines[-1]
                assert not '' in last_line