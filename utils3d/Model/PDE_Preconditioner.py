import os
import numpy as np
import tensorflow as tf


class Preconditioner():

    def __init__(self):
        pass


    def get_loss_preconditioner(self, X_batches, model):
        L = dict()
        L['R'] = 0.0
        L['D'] = 0.0
        L['N'] = 0.0
        L['I'] = 0.0
        L['K'] = 0.0
        L['P'] = 0.0

        #residual
        if 'P' in self.mesh.meshes_names:
            X = X_batches['P'] 
            loss_p = self.preconditioner(self.mesh,model,self.mesh.get_X(X))
            L['P'] += loss_p   
            
        return L