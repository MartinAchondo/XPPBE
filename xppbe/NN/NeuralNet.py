import numpy as np
import tensorflow as tf


class PINN_2Dom_NeuralNet(tf.keras.Model):

    def __init__(self, hyperparameters, **kwargs):
        super().__init__(name='PINN_NN', **kwargs)
        param_1, param_2 = hyperparameters
        self.NNs = [NeuralNet(**param_1, name='Molecule_NN'), NeuralNet(**param_2, name='Solvent_NN')]

    def call(self, X, flag):
        outputs = tf.zeros([tf.shape(X)[0], 2])   
        if flag == 'molecule':
            output = self.NNs[0](X)
            outputs = tf.concat([output, tf.zeros_like(output)], axis=1)
        elif flag == 'solvent':
            output = self.NNs[1](X)
            outputs = tf.concat([tf.zeros_like(output), output], axis=1)
        elif flag =='interface':
            outputs = tf.concat([self.NNs[0](X), self.NNs[1](X)], axis=1)
        return outputs
    
    def build_Net(self):
        self.NNs[0].build_Net()
        self.NNs[1].build_Net()


class PINN_1Dom_NeuralNet(tf.keras.Model):

    def __init__(self, hyperparameters, **kwargs):
        super().__init__(name='PINN_NN', **kwargs)
        param_1, param_2 = hyperparameters
        self.NN = NeuralNet(**param_1, name='NN')
        self.NNs = [self.NN,self.NN]

    def call(self, X, flag):
        output = self.NN(X)
        outputs = tf.concat([output, output], axis=1)
        return outputs
    
    def build_Net(self):
        self.NN.build_Net()


class NeuralNet(tf.keras.Model):

    def __init__(self, 
                 input_shape=(None, 3),
                 output_dim=1,
                 num_hidden_layers=2,
                 num_neurons_per_layer=20,
                 num_hidden_blocks=2,
                 activation='tanh',
                 adaptative_activation=False,
                 kernel_initializer='glorot_normal',
                 architecture_Net='FCNN',
                 fourier_features=False, 
                 num_fourier_features=128, 
                 fourier_sigma=1,
                 weight_factorization=False,
                 scale_input=True,
                 scale_NN=1.0,
                 scale=([-1.,-1.,-1.],[1.,1.,1.]),
                 **kwargs):
        super().__init__(**kwargs)

        self.input_shape_N = input_shape
        self.output_dim = output_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_neurons_per_layer = num_neurons_per_layer
        self.num_hidden_blocks = num_hidden_blocks
        
        self.activation = activation
        self.adaptative_activation = adaptative_activation

        self.kernel_initializer = kernel_initializer
        self.architecture_Net = architecture_Net
        self.use_fourier_features = fourier_features
        self.num_fourier_features = num_fourier_features
        self.fourier_sigma = fourier_sigma
        self.weight_factorization = weight_factorization

        self.scale_input = scale_input
        self.scale_NN = scale_NN
        self.lb = tf.constant(scale[0])
        self.ub = tf.constant(scale[1])

        if not self.weight_factorization:
            Dense_Layer = tf.keras.layers.Dense
        elif self.weight_factorization:
            Dense_Layer = CustomDenseLayer

        # Scale layer
        if self.scale_input:
            self.scale = tf.keras.layers.Lambda(
                lambda x: 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0, 
                name=f'scale_layer')

        # Fourier feature layer
        if self.use_fourier_features:
            self.fourier_features = tf.keras.Sequential(name=f'fourier_layer')
            self.fourier_features.add(tf.keras.layers.Dense(num_fourier_features, 
                                                          activation=None, 
                                                          use_bias=False,
                                                          trainable=False, 
                                                          kernel_initializer=tf.initializers.RandomNormal(stddev=self.fourier_sigma),
                                                          name='fourier_features'))
            class SinCosLayer(tf.keras.layers.Layer):
                def call(self, Z):
                    return tf.concat([tf.sin(2.0*np.pi*Z), tf.cos(2.0*np.pi*Z)], axis=-1)
            self.fourier_features.add(SinCosLayer(name='fourier_sincos_layer'))
        
        # FCNN or ModMLP architectures
        if self.architecture_Net in ('FCNN','ModMLP','MLP'):
            self.hidden_layers = list()
            for i in range(self.num_hidden_layers):
                layer = Dense_Layer(self.num_neurons_per_layer,
                            activation=CustomActivation(units=self.num_neurons_per_layer,
                                                        activation=self.activation, 
                                                        adaptative_activation=self.adaptative_activation),
                            kernel_initializer=self.kernel_initializer,
                            name=f'layer_{i}')
                self.hidden_layers.append(layer)

            # ModMLP architecture
            if self.architecture_Net == 'ModMLP':
                self.U = Dense_Layer(self.num_neurons_per_layer,
                            activation=CustomActivation(units=self.num_neurons_per_layer,
                                                        activation=self.activation, 
                                                        adaptative_activation=self.adaptative_activation),
                            kernel_initializer=self.kernel_initializer,
                            name=f'layer_u')
                self.V = Dense_Layer(self.num_neurons_per_layer,
                            activation=CustomActivation(units=self.num_neurons_per_layer,
                                                        activation=self.activation, 
                                                        adaptative_activation=self.adaptative_activation),
                            kernel_initializer=self.kernel_initializer,
                            name=f'layer_v')

        # ResNet architecture
        elif self.architecture_Net == 'ResNet':
            self.first = Dense_Layer(self.num_neurons_per_layer,
                            activation=CustomActivation(units=self.num_neurons_per_layer,
                                                        activation=self.activation, 
                                                        adaptative_activation=self.adaptative_activation),
                            kernel_initializer=self.kernel_initializer,
                            name=f'layer_0')
            self.hidden_blocks = list()
            self.hidden_blocks_activations = list()
            for i in range(self.num_hidden_blocks):
                block = tf.keras.Sequential(name=f"block_{i}")
                block.add(Dense_Layer(self.num_neurons_per_layer,
                                activation=CustomActivation(units=self.num_neurons_per_layer,
                                                            activation=self.activation,
                                                            adaptative_activation=self.adaptative_activation),
                                kernel_initializer=self.kernel_initializer))
                block.add(Dense_Layer(self.num_neurons_per_layer,
                                activation=None,
                                kernel_initializer=self.kernel_initializer))
                self.hidden_blocks.append(block)
                activation_layer = tf.keras.layers.Activation(activation=CustomActivation(units=self.num_neurons_per_layer,
                                                                                          activation=self.activation,
                                                                                          adaptative_activation=self.adaptative_activation))
                self.hidden_blocks_activations.append(activation_layer)
            
            self.last = Dense_Layer(self.num_neurons_per_layer,
                            activation=CustomActivation(units=self.num_neurons_per_layer,
                                                        activation=self.activation, 
                                                        adaptative_activation=self.adaptative_activation),
                            kernel_initializer=self.kernel_initializer,
                            name=f'layer_1')

        # Output layer
        self.out = Dense_Layer(output_dim, name=f'output_layer')

        # Scale output layer
        self.scale_out = tf.keras.layers.Lambda(
            lambda x: x*self.scale_NN, 
            name=f'scale_output_layer')

    def build_Net(self):
        self.build(self.input_shape_N)

    def call(self, X):
        if self.architecture_Net in ('FCNN','MLP'):
            return self.call_FCNN(X)
        elif self.architecture_Net == 'ModMLP':
            return self.call_ModMLP(X)
        elif self.architecture_Net == 'ResNet':
           return self.call_ResNet(X)

    # Call NeuralNet functions with the desired architecture

    def call_FCNN(self, X):
        if self.scale_input:
            X = self.scale(X)
        if self.use_fourier_features:
            X = self.fourier_features(X) 
        for layer in self.hidden_layers:
            X = layer(X)
        X = self.out(X)
        return self.scale_out(X)

    def call_ModMLP(self, X):
        if self.scale_input:
            X = self.scale(X)
        if self.use_fourier_features:
            X = self.fourier_features(X) 
        U = self.U(X)
        V = self.V(X)
        for layer in self.hidden_layers:
            X = layer(X)*U + (1-layer(X))*V
        X = self.out(X)
        return self.scale_out(X)

    def call_ResNet(self, X):
        if self.scale_input:
            X = self.scale(X)
        if self.use_fourier_features:
            X = self.fourier_features(X)  
        X = self.first(X)
        for block,activation in zip(self.hidden_blocks,self.hidden_blocks_activations):
            X = activation(block(X) + X)
        X = self.last(X)
        X = self.out(X)
        return self.scale_out(X)
    

class CustomActivation(tf.keras.layers.Layer):

    def __init__(self, units=1, activation='tanh', adaptative_activation=False, **kwargs):
        super(CustomActivation, self).__init__(**kwargs)
        self.units = units
        self.activation = activation
        self.adaptative_activation = adaptative_activation

    def build(self, input_shape):
        self.a = self.add_weight(name='a',
                                shape=(self.units,),
                                initializer='ones',
                                trainable=self.adaptative_activation)

    def call(self, inputs):
        a_expanded = tf.expand_dims(self.a, axis=0) 
        activation_func = tf.keras.activations.get(self.activation)
        return activation_func(inputs * a_expanded)

class CustomDenseLayer(tf.keras.layers.Layer):

    def __init__(self, units, activation=None, kernel_initializer='glorot_normal', **kwargs):
        super(CustomDenseLayer, self).__init__(**kwargs)
        self.units = units
        self.activation = activation
        self.kernel_initializer= tf.keras.initializers.get(kernel_initializer)
                                        
    def build(self, input_shape):
        self.input_dim = input_shape[-1]
        W = tf.Variable(initial_value=self.kernel_initializer(shape=(input_shape[-1], self.units)),
                trainable=False)
        S, V = self.weight_factorization(W)
        self.S = self.add_weight(name='S', shape=S.shape, initializer=tf.constant_initializer(S.numpy()), trainable=True)
        self.V = self.add_weight(name='V', shape=V.shape, initializer=tf.constant_initializer(V.numpy()), trainable=True)
        self.b = self.add_weight(name='b', shape=(self.units,), initializer='zeros', trainable=True)

    def weight_factorization(self, W, mean=1.0, stddev=0.1):
        S = mean + tf.random.normal(shape=[tf.shape(W)[-1]], stddev=stddev)
        S = tf.exp(S)
        V = W / S
        return S,V

    def call(self, inputs):
        SV = tf.multiply(self.S, self.V)
        outputs = tf.matmul(inputs, SV) + self.b
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs

