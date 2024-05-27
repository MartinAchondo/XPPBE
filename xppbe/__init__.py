import os

os.environ['TF_RUN_EAGER_OP_AS_FUNCTION']='false'

xppbe_path = os.path.dirname(os.path.abspath(__file__))

from xppbe.Simulation import Simulation
import xppbe.Molecules
import xppbe.Model
import xppbe.Mesh
import xppbe.NN
import xppbe.Post
