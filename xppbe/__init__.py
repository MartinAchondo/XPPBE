import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_RUN_EAGER_OP_AS_FUNCTION']='false'
# import tensorflow as tf
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from xppbe.Simulation import Simulation

xppbe_path = os.path.dirname(os.path.abspath(__file__))

