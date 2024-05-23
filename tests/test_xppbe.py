import os
import tempfile
import pytest
import csv
from xppbe.Simulation import Simulation


@pytest.mark.parametrize(
 ('molecule'),
 (
     ('born_ion'),
     ('methanol'),
     ('arg')
 )       
)
def test_xppbe_solver(molecule):

    with tempfile.TemporaryDirectory() as temp_dir:
        
        yaml_path = os.path.join(os.path.dirname(__file__),f'test_{molecule}.yaml')
        sim = Simulation(yaml_path, results_path=temp_dir)
        sim.create_simulation()
        sim.adapt_model()
        sim.solve_model()
        sim.postprocessing(mesh=False, pbj=False)

        results_path = os.path.join(temp_dir,'results')
        assert os.path.isdir(results_path)
        assert len(os.listdir(results_path)) > 0

        results_path = os.path.join(results_path,f'test_{molecule}')
        for name in ['iterations','mesh','plots_losses','plots_model','plots_solution','plots_weights','Post.ipynb','results_values.json',f'test_{molecule}.yaml']:
            assert name in os.listdir(results_path)

        mesh_path = os.path.join(results_path,'mesh')
        assert len(os.listdir(mesh_path)) == 10

        iterations_path = os.path.join(results_path,'iterations')
        assert len(os.listdir(iterations_path)) == 1

        last_iteration = os.path.join(iterations_path,f'iter_{sim.N_iters}')
        listdir_last_iteration = os.listdir(last_iteration)

        for file_name in ['optimizer.npy', 'weights.index', 'w_hist.csv', 'checkpoint', 'loss.csv', 'hyperparameters.json', 'G_solv.csv', 'loss_validation.csv', 'weights.data-00000-of-00001']:
            assert file_name in listdir_last_iteration

        for file_name in ['loss.csv','loss_validation.csv','w_hist.csv']:
            file = os.path.join(last_iteration,file_name)
            with open(file, 'r') as csvfile:
                reader = csv.reader(csvfile)
                lines = list(reader)
                assert len(lines) == sim.N_iters + 1


def test_iteration_continuation():

    with tempfile.TemporaryDirectory() as temp_dir:
        
        yaml_path = os.path.join(os.path.dirname(__file__),'test_born_ion.yaml')
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

        for file_name in ['loss.csv','loss_validation.csv','w_hist.csv']:
            file = os.path.join(last_iteration,file_name)
            with open(file, 'r') as csvfile:
                reader = csv.reader(csvfile)
                lines = list(reader)
                assert len(lines) == sim_2.N_iters + 1