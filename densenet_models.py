# -*- coding: utf-8 -*-
# DenseNet Neural Network Models
import tensorflow as tf
import math

# Dense blocks where each the output of every convolution layer is concatenated
class DenseBlock(tf.keras.layers.Layer):
    # Note the added `**kwargs`, as Keras supports many arguments
    def __init__(self, layers, bottleneck, growth, **kwargs):
        super().__init__(**kwargs)
        
        # Layers variable will be needed during forward call
        self.layers = layers
        
        # Lists for storing variable number of convolution/batch norm layers
        self.conv_1x1 = []
        self.conv_3x3 = []
        self.bn_1x1 = []
        self.bn_3x3 = []
        
        # Add convolution/batch norm layers
        # NOTE: range() is used instead of tf.range() to avoid error
        for i in range(self.layers):
            self.conv_1x1.append(tf.keras.layers.Conv2D(filters=bottleneck, kernel_size=1, strides=1, padding="same"))
            self.conv_3x3.append(tf.keras.layers.Conv2D(filters=growth, kernel_size=3, strides=1, padding="same"))
            self.bn_1x1.append(tf.keras.layers.BatchNormalization(axis=-1))
            self.bn_3x3.append(tf.keras.layers.BatchNormalization(axis=-1))

    def call(self, x):
        # Initialize concatenated feature maps
        feature_maps = x
        
        # For each pair of 1x1 and 3x3 convolution layers in the dense block,
        # generate k feature maps and concatenate with all previously generated
        # feature maps
        for i in range(self.layers):
            layer_output = self.bn_1x1[i](feature_maps)
            layer_output = tf.keras.layers.ReLU()(layer_output)
            layer_output = self.conv_1x1[i](layer_output)
            
            layer_output = self.bn_3x3[i](layer_output)
            layer_output = tf.keras.layers.ReLU()(layer_output)
            layer_output = self.conv_3x3[i](layer_output)
            
            feature_maps = tf.keras.layers.Concatenate(axis=-1)([feature_maps, layer_output])

        return feature_maps

# Transition layer between dense blocks to decrease feature map size
class TransitionLayer(tf.keras.layers.Layer):
    # Note the added `**kwargs`, as Keras supports many arguments
    def __init__(self, compression, **kwargs):
        super().__init__(**kwargs)
        
        # Compression factor will be needed for the build call
        self.compression = compression
        
        # Downsampling average pooling layer does not need to know input shape
        self.avg_pool = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2, padding="same")
        
        # Batch normalization layer preceding convolution layer
        self.bn = tf.keras.layers.BatchNormalization(axis=-1)
        
    # To properly implement DenseNet compression, the number of input feature
    # maps must be known
    def build(self, input_shape):
        self.conv = tf.keras.layers.Conv2D(filters=math.floor(input_shape[-1]*self.compression), kernel_size=1, strides=1, padding="same")

    def call(self, x):
        x = self.bn(x)
        x = tf.keras.layers.ReLU()(x)
        x = self.conv(x)
        x = self.avg_pool(x)
        
        return x
    
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

# Full DenseNet network
# param classes:            number of dataset classes
# param initial_features:   number of features created by first convolution
# param blocks:             the number of dense blocks in the model
# param layers:             list indicating the number of convolution layers 
#                           within each dense block
# param bottleneck:         number of features each bottleneck layer produces
# param growth:             number of features each layer within each dense
#                           block produces (aka "k")
# param compression:        compression factor for decreasing the number of 
#                           features between dense blocks (aka "theta")
# Parameter configurations that will be tested:
# (35, 16, 4, [2, 4, 8, 6], 32, 8, 0.5)
# (35, 16, 4, [2, 4, 8, 5], 32, 8, 0.5)
# (35, 16, 4, [2, 4, 8, 4], 32, 8, 0.5)
# (35, 16, 4, [2, 4, 8, 3], 32, 8, 0.5)
# (35, 16, 4, [2, 4, 8, 2], 32, 8, 0.5)
# (35, 16, 4, [2, 4, 8, 1], 32, 8, 0.5)
# (35, 16, 4, [2, 4, 7, 1], 32, 8, 0.5)
# (35, 16, 4, [2, 4, 6, 1], 32, 8, 0.5)
# (35, 16, 4, [2, 4, 5, 1], 32, 8, 0.5)
# (35, 16, 4, [2, 4, 4, 1], 32, 8, 0.5)
# (35, 16, 4, [2, 4, 3, 1], 32, 8, 0.5)
# (35, 16, 4, [2, 4, 2, 1], 32, 8, 0.5)
# (35, 16, 4, [2, 4, 1, 1], 32, 8, 0.5)
# (35, 16, 4, [2, 3, 1, 1], 32, 8, 0.5)
# (35, 16, 4, [2, 2, 1, 1], 32, 8, 0.5)
# (35, 16, 4, [2, 1, 1, 1], 32, 8, 0.5)
# (35, 16, 4, [1, 1, 1, 1], 32, 8, 0.5)
# (35, 16, 3, [1, 1, 1], 32, 8, 0.5)
# (35, 16, 2, [1, 1], 32, 8, 0.5)
# (35, 16, 1, [1], 32, 8, 0.5)
class DenseNet(tf.keras.Model):
    def __init__(self, classes, initial_features, blocks, layers, bottleneck, growth, compression, **kwargs):
        super().__init__(**kwargs)
        
        # Save number of dense blocks
        self.blocks = blocks
        
        # Initial convolution layer (produces intial_features)
        self.conv_01 = tf.keras.layers.Conv2D(filters=initial_features, kernel_size=7, strides=2, padding="same")
        self.bn_01 = tf.keras.layers.BatchNormalization(axis=-1)
        
        # Add dense blocks specified by blocks, layers, bottleneck, and growth params
        # Add transition layers specified by compression param
        self.dense_blocks = []
        self.transition_layers = []
        
        for i in range(self.blocks):
            self.dense_blocks.append(DenseBlock(layers=layers[i], bottleneck=bottleneck, growth=growth))
            
        for i in range(self.blocks-1):
            self.transition_layers.append(TransitionLayer(compression=compression))
         
        # Global average pooling layer (reduces to 1x1xC)
        self.avg_pool = GlobalPoolLayer()  
        
        # Fully connected layer with classes param outputs
        self.full_con = tf.keras.layers.Dense(units=classes, activation="softmax")

    def call(self, x):
        x = self.bn_01(x)
        x = tf.keras.layers.ReLU()(x)
        x = self.conv_01(x)

        for i in range(self.blocks-1):
            x = self.dense_blocks[i](x)
            x = self.transition_layers[i](x)
            
        x = self.dense_blocks[-1](x)
        x = self.avg_pool(x)
        x = tf.keras.layers.Flatten()(x)
        x = self.full_con(x)
        
        return x
    
    # To get model summary, use DenseNet(params).model(input_shape).summary()
    # NOTE: Exclude batch dimension from input shape
    def model(self, input_shape):
        x = tf.keras.layers.Input(shape=input_shape)
        return tf.keras.Model(inputs=x, outputs=self.call(x))
