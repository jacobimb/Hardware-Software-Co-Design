# -*- coding: utf-8 -*-
# CENet Neural Network Models
import tensorflow as tf
import math

# CENet bottleneck layer
class BottleneckLayer(tf.keras.layers.Layer):
    # Note the added `**kwargs`, as Keras supports many arguments
    def __init__(self, ratio, **kwargs):
        super().__init__(**kwargs)
        
        # Save layer feature ratio
        self.ratio = ratio
        
        # Batch normalization layers
        self.bn_1 = tf.keras.layers.BatchNormalization(axis=-1)
        self.bn_2 = tf.keras.layers.BatchNormalization(axis=-1)
        self.bn_3 = tf.keras.layers.BatchNormalization(axis=-1)
        
    # Convolution layers can only be built when the input shape is known
    def build(self, input_shape):
        self.conv_1 = tf.keras.layers.Conv2D(filters=math.floor(input_shape[-1]*self.ratio), kernel_size=1, strides=1, padding="same")
        self.conv_2 = tf.keras.layers.Conv2D(filters=math.floor(input_shape[-1]*self.ratio), kernel_size=3, strides=1, padding="same")
        self.conv_3 = tf.keras.layers.Conv2D(filters=input_shape[-1], kernel_size=1, strides=1, padding="same")

    def call(self, inputs):
        # Forward path
        forward = self.conv_1(inputs)
        forward = self.bn_1(forward)
        forward = tf.keras.layers.ReLU()(forward)
        forward = self.conv_2(forward)
        forward = self.bn_2(forward)
        forward = tf.keras.layers.ReLU()(forward)
        forward = self.conv_3(forward)
        forward = self.bn_3(forward)
        forward = tf.keras.layers.ReLU()(forward)
        
        return tf.keras.layers.add(inputs=[forward, inputs]) 
    
# CENet connection layer
class ConnectionLayer(tf.keras.layers.Layer):
    # Note the added `**kwargs`, as Keras supports many arguments
    def __init__(self, ratio, out_features, **kwargs):
        super().__init__(**kwargs)
        
        # Save layer feature ratio
        self.ratio = ratio
        
        # Convolution layers that do not require knowledge of input shape
        self.conv_3 = tf.keras.layers.Conv2D(filters=out_features, kernel_size=1, strides=2, padding="same")
        self.short = tf.keras.layers.Conv2D(filters=out_features, kernel_size=1, strides=2, padding="same")
        
        # Batch normalization layers
        self.bn_1 = tf.keras.layers.BatchNormalization(axis=-1)
        self.bn_2 = tf.keras.layers.BatchNormalization(axis=-1)
        self.bn_3 = tf.keras.layers.BatchNormalization(axis=-1)
        self.bn_short = tf.keras.layers.BatchNormalization(axis=-1)
        
    # Convolution layers can only be built when the input shape is known
    def build(self, input_shape):
        self.conv_1 = tf.keras.layers.Conv2D(filters=math.floor(input_shape[-1]*self.ratio), kernel_size=1, strides=1, padding="same")
        self.conv_2 = tf.keras.layers.Conv2D(filters=math.floor(input_shape[-1]*self.ratio), kernel_size=3, strides=1, padding="same")
        

    def call(self, inputs):
        # Forward path
        forward = self.conv_1(inputs)
        forward = self.bn_1(forward)
        forward = tf.keras.layers.ReLU()(forward)
        forward = self.conv_2(forward)
        forward = self.bn_2(forward)
        forward = tf.keras.layers.ReLU()(forward)
        forward = self.conv_3(forward)
        forward = self.bn_3(forward)
        forward = tf.keras.layers.ReLU()(forward)
        
        # Shortcut path
        shortcut = self.short(inputs)
        shortcut = self.bn_short(shortcut)
        shortcut = tf.keras.layers.ReLU()(shortcut)
        
        return tf.keras.layers.add(inputs=[forward, shortcut]) 
    
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
# param stages:             list indicating number of consecutive bottleneck 
#                           layers in each CENet stage             
# param parameters:         list of lists indicating the configuration of each
#                           CENet stage
# Parameter configurations that will be tested:
# (35, 16, [15, 15, 7], [[0.5, 0.5, 32], [0.25, 0.25, 48], [0.25, 0.25, 64]]) CENet-40
# (35, 16, [7, 7, 7], [[0.5, 0.5, 32], [0.25, 0.25, 48], [0.25, 0.25, 64]]) CENet-24
# (35, 16, [7, 7, 6], [[0.5, 0.5, 32], [0.25, 0.25, 48], [0.25, 0.25, 64]])
# (35, 16, [7, 7, 5], [[0.5, 0.5, 32], [0.25, 0.25, 48], [0.25, 0.25, 64]])
# (35, 16, [7, 7, 4], [[0.5, 0.5, 32], [0.25, 0.25, 48], [0.25, 0.25, 64]])
# (35, 16, [7, 7, 3], [[0.5, 0.5, 32], [0.25, 0.25, 48], [0.25, 0.25, 64]])
# (35, 16, [7, 7, 2], [[0.5, 0.5, 32], [0.25, 0.25, 48], [0.25, 0.25, 64]])
# (35, 16, [7, 7, 1], [[0.5, 0.5, 32], [0.25, 0.25, 48], [0.25, 0.25, 64]])
# (35, 16, [7, 6, 1], [[0.5, 0.5, 32], [0.25, 0.25, 48], [0.25, 0.25, 64]])
# (35, 16, [7, 5, 1], [[0.5, 0.5, 32], [0.25, 0.25, 48], [0.25, 0.25, 64]])
# (35, 16, [7, 4, 1], [[0.5, 0.5, 32], [0.25, 0.25, 48], [0.25, 0.25, 64]])
# (35, 16, [7, 3, 1], [[0.5, 0.5, 32], [0.25, 0.25, 48], [0.25, 0.25, 64]])
# (35, 16, [7, 2, 1], [[0.5, 0.5, 32], [0.25, 0.25, 48], [0.25, 0.25, 64]])
# (35, 16, [7, 1, 1], [[0.5, 0.5, 32], [0.25, 0.25, 48], [0.25, 0.25, 64]])
# (35, 16, [6, 1, 1], [[0.5, 0.5, 32], [0.25, 0.25, 48], [0.25, 0.25, 64]])
# (35, 16, [5, 1, 1], [[0.5, 0.5, 32], [0.25, 0.25, 48], [0.25, 0.25, 64]])
# (35, 16, [4, 1, 1], [[0.5, 0.5, 32], [0.25, 0.25, 48], [0.25, 0.25, 64]])
# (35, 16, [3, 1, 1], [[0.5, 0.5, 32], [0.25, 0.25, 48], [0.25, 0.25, 64]])
# (35, 16, [2, 1, 1], [[0.5, 0.5, 32], [0.25, 0.25, 48], [0.25, 0.25, 64]])
# (35, 16, [1, 1, 1], [[0.5, 0.5, 32], [0.25, 0.25, 48], [0.25, 0.25, 64]]) CENet-6
# (35, 16, [1, 1], [[0.5, 0.5, 32], [0.25, 0.25, 48]])
# (35, 16, [1], [[0.5, 0.5, 32]])
class CENet(tf.keras.Model):
    def __init__(self, classes, initial_features, stages, parameters, **kwargs):
        super().__init__(**kwargs)
        
        # Save total number of residual layers
        self.layer_count = sum(stages)+len(stages)
        
        # Initial layer
        self.conv_01 = tf.keras.layers.Conv2D(filters=initial_features, kernel_size=3, strides=1, padding="same")
        self.bn_01 = tf.keras.layers.BatchNormalization(axis=-1)
        
        # Lists to hold variable number of residual layers
        self.network_layers = []
        
        # Add Inception modules and max pooling layers
        for i in range(len(stages)):
            for j in range(stages[i]):
                self.network_layers.append(BottleneckLayer(ratio=parameters[i][0]))
            self.network_layers.append(ConnectionLayer(ratio=parameters[i][1], out_features=parameters[i][2]))
        
        # Global average pooling layer (reduces to 1x1xC)
        self.avg_pool = GlobalPoolLayer()
        
        # Fully connected layer with classes param outputs
        self.full_con = tf.keras.layers.Dense(units=classes, activation="softmax")

    def call(self, x):
        x = self.conv_01(x)
        x = self.bn_01(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2, padding="same")(x)
        
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
