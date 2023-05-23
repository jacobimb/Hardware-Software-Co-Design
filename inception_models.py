# -*- coding: utf-8 -*-
# Inception Neural Network Models
import tensorflow as tf

# Inception module
class InceptionModule(tf.keras.layers.Layer):
    # Note the added `**kwargs`, as Keras supports many arguments
    def __init__(self, features_1x1_1x1, features_3x3_1x1, features_3x3_3x3, features_5x5_1x1, features_5x5_5x5, features_pool_1x1, **kwargs):
        super().__init__(**kwargs)
        
        # 1x1 convolution branch
        self.branch_1x1_1x1 = tf.keras.layers.Conv2D(filters=features_1x1_1x1, kernel_size=1, strides=1, padding="same")
        
        # 3x3 convolution branch
        self.branch_3x3_1x1 = tf.keras.layers.Conv2D(filters=features_3x3_1x1, kernel_size=1, strides=1, padding="same")
        self.branch_3x3_3x3 = tf.keras.layers.Conv2D(filters=features_3x3_3x3, kernel_size=3, strides=1, padding="same")
        
        # 5x5 convolution branch
        self.branch_5x5_1x1 = tf.keras.layers.Conv2D(filters=features_5x5_1x1, kernel_size=1, strides=1, padding="same")
        self.branch_5x5_5x5 = tf.keras.layers.Conv2D(filters=features_5x5_5x5, kernel_size=5, strides=1, padding="same")
        
        # Max pooling branch (only the 1x1 convolution layer needs created)
        self.branch_pool_1x1 = tf.keras.layers.Conv2D(filters=features_pool_1x1, kernel_size=1, strides=1, padding="same")

    def call(self, inputs):
        
        branch_1x1_output = self.branch_1x1_1x1(inputs)
        branch_1x1_output = tf.keras.layers.ReLU()(branch_1x1_output)
        
        branch_3x3_ouptut = self.branch_3x3_1x1(inputs)
        branch_3x3_ouptut = tf.keras.layers.ReLU()(branch_3x3_ouptut)
        branch_3x3_ouptut = self.branch_3x3_3x3(branch_3x3_ouptut)
        branch_3x3_ouptut = tf.keras.layers.ReLU()(branch_3x3_ouptut)
        
        branch_5x5_output = self.branch_5x5_1x1(inputs)
        branch_5x5_output = tf.keras.layers.ReLU()(branch_5x5_output)
        branch_5x5_output = self.branch_5x5_5x5(branch_5x5_output)
        branch_5x5_output = tf.keras.layers.ReLU()(branch_5x5_output)
        
        branch_pool_output = tf.keras.layers.MaxPool2D(pool_size=3, strides=1, padding="same")(inputs)
        branch_pool_output = self.branch_pool_1x1(branch_pool_output)
        branch_pool_output = tf.keras.layers.ReLU()(branch_pool_output)
        
        return tf.keras.layers.Concatenate(axis=-1)([branch_1x1_output, branch_3x3_ouptut, branch_5x5_output, branch_pool_output])
    
# Final global average pooling layer
# Takes NxNxC and outputs 1x1xC
class GlobalPoolLayer(tf.keras.layers.Layer):
    # Note the added `**kwargs`, as Keras supports many arguments
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    # To always pool h and w, input shape must be known
    def build(self, input_shape):
        h = input_shape[-3]
        w = input_shape[-2]
            
        self.avg_pool = tf.keras.layers.AveragePooling2D(pool_size=(h, w))
        
    def call(self, x):
        x = self.avg_pool(x)
        
        return x
    
# Full ResNet network
# param classes:            number of dataset classes
# param initial_features:   number of features created by first convolution
# param modules:            list indicating number of consecutive Inception 
#                           modules before a transition occurs              
# param parameters:         list of lists indicating the configuration of each
#                           set of inception modules
# Parameter configurations that will be tested:
# (35, 16, [2, 4, 2], [[8, 12, 16, 2, 4, 4], [24, 12, 26, 2, 6, 8], [48, 24, 48, 6, 16, 16]])
# (35, 16, [2, 4, 1], [[8, 12, 16, 2, 4, 4], [24, 12, 26, 2, 6, 8], [48, 24, 48, 6, 16, 16]])
# (35, 16, [2, 3, 1], [[8, 12, 16, 2, 4, 4], [24, 12, 26, 2, 6, 8], [48, 24, 48, 6, 16, 16]])
# (35, 16, [2, 2, 1], [[8, 12, 16, 2, 4, 4], [24, 12, 26, 2, 6, 8], [48, 24, 48, 6, 16, 16]])
# (35, 16, [2, 1, 1], [[8, 12, 16, 2, 4, 4], [24, 12, 26, 2, 6, 8], [48, 24, 48, 6, 16, 16]])
# (35, 16, [1, 1, 1], [[8, 12, 16, 2, 4, 4], [24, 12, 26, 2, 6, 8], [48, 24, 48, 6, 16, 16]])
# (35, 16, [1, 1], [[8, 12, 16, 2, 4, 4], [24, 12, 26, 2, 6, 8]])
# (35, 16, [1], [[8, 12, 16, 2, 4, 4]])
class InceptionV1(tf.keras.Model):
    def __init__(self, classes, initial_features, modules, parameters, **kwargs):
        super().__init__(**kwargs)
        
        # Save total number of residual layers
        self.layer_count = sum(modules)+len(modules)-1
        
        # Initial layer
        self.conv_01 = tf.keras.layers.Conv2D(filters=initial_features, kernel_size=7, strides=2, padding="same")
        
        # Lists to hold variable number of residual layers
        self.network_layers = []
        
        # Add Inception modules and max pooling layers
        for i in range(len(modules)-1):
            for j in range(modules[i]):
                self.network_layers.append(InceptionModule(features_1x1_1x1=parameters[i][0],
                                                           features_3x3_1x1=parameters[i][1],
                                                           features_3x3_3x3=parameters[i][2],
                                                           features_5x5_1x1=parameters[i][3],
                                                           features_5x5_5x5=parameters[i][4],
                                                           features_pool_1x1=parameters[i][5]))
            self.network_layers.append(tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same"))
            
        # Add Inception modules and max pooling layers
        for j in range(modules[-1]):
            self.network_layers.append(InceptionModule(features_1x1_1x1=parameters[-1][0],
                                                       features_3x3_1x1=parameters[-1][1],
                                                       features_3x3_3x3=parameters[-1][2],
                                                       features_5x5_1x1=parameters[-1][3],
                                                       features_5x5_5x5=parameters[-1][4],
                                                       features_pool_1x1=parameters[-1][5]))
        
        # Global average pooling layer (reduces to 1x1xC)
        self.avg_pool = GlobalPoolLayer()
        
        # Fully connected layer with classes param outputs
        self.full_con = tf.keras.layers.Dense(units=classes, activation="softmax")

    def call(self, x):
        x = self.conv_01(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Lambda(tf.nn.local_response_normalization)(x)
        
        for i in range(self.layer_count):
            x = self.network_layers[i](x)
        
        x = self.avg_pool(x)
        x = tf.keras.layers.Flatten()(x)
        x = self.full_con(x)
        return x
    
    # To get model summary, use ResNet(params).model(input_shape).summary()
    # NOTE: Exclude batch dimension from input shape
    def model(self, input_shape):
        x = tf.keras.layers.Input(shape=input_shape)
        return tf.keras.Model(inputs=x, outputs=self.call(x))
