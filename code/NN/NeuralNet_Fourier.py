import tensorflow as tf
import numpy as np
import os

class NeuralNet(tf.keras.Model):

    def __init__(self, lb, ub, 
                 input_shape=(None, 3),
                 output_dim=1,
                 num_hidden_layers=2,
                 num_neurons_per_layer=20,
                 num_hidden_blocks=2,
                 activation='tanh',
                 kernel_initializer='glorot_normal',
                 architecture_Net='FCNN',
                 num_fourier_features=256, 
                 fourier_sigma=1,
                 **kwargs):
        super().__init__(**kwargs)

        self.input_shape_N = input_shape
        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_blocks = num_hidden_blocks
        self.output_dim = output_dim
        self.lb = lb
        self.ub = ub
        self.architecture_Net = architecture_Net
        self.num_fourier_features = num_fourier_features
        self.sigma = fourier_sigma

        # Scale layer
        self.scale = tf.keras.layers.Lambda(
            lambda x: 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0, 
            name=f'layer_input')

        # Fourier feature layer
        self.fourier_features = tf.keras.Sequential(name=f'fourier_layer')
        self.fourier_features.add(tf.keras.layers.Dense(num_fourier_features, 
                                                      activation=None, 
                                                      use_bias=False,
                                                      trainable=False, 
                                                      kernel_initializer=tf.initializers.RandomNormal(stddev=self.sigma),
                                                      name='fourier_features'))
        class SinCosLayer(tf.keras.layers.Layer):
            def call(self, Z):
                return tf.concat([tf.sin(2.0*np.pi*Z), tf.cos(2.0*np.pi*Z)], axis=-1)
        self.fourier_features.add(SinCosLayer(name='fourier_sincos_layer'))

        # FCNN architecture
        if self.architecture_Net == 'FCNN':
            self.hidden_layers = list()
            for i in range(self.num_hidden_layers):
                layer = tf.keras.layers.Dense(num_neurons_per_layer,
                                              activation=tf.keras.activations.get(activation),
                                              kernel_initializer=kernel_initializer,
                                              name=f'layer_{i}')
                self.hidden_layers.append(layer)

        # ResNet architecture
        elif self.architecture_Net == 'ResNet':
            self.first = tf.keras.layers.Dense(num_neurons_per_layer,
                                               activation=tf.keras.activations.get(activation),
                                               kernel_initializer=kernel_initializer,
                                               name=f'layer_0')
            self.hidden_blocks = list()
            self.hidden_blocks_activations = list()
            for i in range(self.num_hidden_blocks):
                block = tf.keras.Sequential(name=f"block_{i}")
                block.add(tf.keras.layers.Dense(num_neurons_per_layer,
                                                activation=tf.keras.activations.get(activation),
                                                kernel_initializer=kernel_initializer))
                block.add(tf.keras.layers.Dense(num_neurons_per_layer,
                                                activation=None,
                                                kernel_initializer=kernel_initializer))
                self.hidden_blocks.append(block)
                activation_layer = tf.keras.layers.Activation(tf.keras.activations.get(activation))
                self.hidden_blocks_activations.append(activation_layer)
            
            self.last = tf.keras.layers.Dense(num_neurons_per_layer,
                                              activation=tf.keras.activations.get(activation),
                                              kernel_initializer=kernel_initializer,
                                              name=f'layer_1')

        # Output layer
        self.out = tf.keras.layers.Dense(output_dim, name=f'layer_output')
        


    def build_Net(self):
        self.build(self.input_shape_N)

    def call(self, X):
        if self.architecture_Net == 'FCNN':
            return self.call_FCNN(X)
        elif self.architecture_Net == 'ResNet':
           return self.call_ResNet(X)

    # Call NeuralNet functions with the desired architecture

    def call_FCNN(self, X):
        Z = self.scale(X)
        Z = self.fourier_features(Z) 
        for layer in self.hidden_layers:
            Z = layer(Z)
        return self.out(Z)

    def call_ResNet(self, X):
        Z = self.scale(X)
        Z = self.fourier_features(Z)  
        Z = self.first(Z)
        for block,activation in zip(self.hidden_blocks,self.hidden_blocks_activations):
            Z = activation(block(Z) + Z)
        Z = self.last(Z)
        return self.out(Z)
    

    def plot_model(self):
        input_layer = tf.keras.layers.Input(shape=self.input_shape_N[1:], name='input')
    
        visual_model = tf.keras.models.Model(inputs=input_layer, outputs=self.call(input_layer))

        main_path = os.path.join(os.path.dirname(os.path.abspath(__file__)))
        tf.keras.utils.plot_model(visual_model, to_file=os.path.join(main_path,'model_achitecture.png'),
                                    show_shapes=True,
                                    show_dtype=False,
                                    show_layer_names=True,
                                    expand_nested=True,
                                    show_layer_activations=True,
                                    dpi = 150)
        
        self.build_Net()
        print(self.summary(expand_nested=True))


if __name__=='__main__':
    hyperparameters = {
                'input_shape': (None,3),
                'num_hidden_layers': 4,
                'num_neurons_per_layer': 200,
                'output_dim': 1,
                'activation': 'tanh',
                'architecture_Net': 'FCNN'
        }
    model = NeuralNet(lb=-1,ub=1,**hyperparameters)
    model.plot_model()
