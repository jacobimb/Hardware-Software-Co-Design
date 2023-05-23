# -*- coding: utf-8 -*-
# ResNet Neural Network Models
import tensorflow as tf

# Regular residual layer with two convolution layers and a single shortcut
# conection that also reduces the size of the input to match the output size
class ResidualConvReduce(tf.keras.layers.Layer):
    # Note the added `**kwargs`, as Keras supports many arguments
    def __init__(self, out_features, **kwargs):
        super().__init__(**kwargs)
        
        # Two regular convolution layers that reduce the size by half
        self.conv1 = tf.keras.layers.Conv2D(filters=out_features, kernel_size=3, strides=2, padding="same")
        self.conv2 = tf.keras.layers.Conv2D(filters=out_features, kernel_size=3, strides=1, padding="same")
        self.bn_01 = tf.keras.layers.BatchNormalization(axis=-1)
        self.bn_02 = tf.keras.layers.BatchNormalization(axis=-1)
        
        # Convolution layer for shortcut connection that also reduces the size
        self.short = tf.keras.layers.Conv2D(filters=out_features, kernel_size=1, strides=2)
        self.bn_short = tf.keras.layers.BatchNormalization(axis=-1)

    def call(self, inputs):
        # Forward path
        forward = self.conv1(inputs)
        forward = self.bn_01(forward)
        forward = tf.keras.layers.ReLU()(forward)
        forward = self.conv2(forward)
        forward = self.bn_02(forward)
        forward = tf.keras.layers.ReLU()(forward)
        
        # Shortcut connection
        shortcut = self.short(inputs)
        shortcut = self.bn_short(shortcut)
        shortcut = tf.keras.layers.ReLU()(shortcut)
        
        # Add forward and shortcut to complete the residual layer
        return tf.keras.layers.add(inputs=[forward, shortcut])

# Regular residual layer with two convolution layers and a single one-to-one
# shortcut conection
class ResidualConv(tf.keras.layers.Layer):
    # Note the added `**kwargs`, as Keras supports many arguments
    def __init__(self, out_features, **kwargs):
        super().__init__(**kwargs)
        
        # Two regular convolution layers
        self.conv1 = tf.keras.layers.Conv2D(filters=out_features, kernel_size=3, strides=1, padding="same")
        self.conv2 = tf.keras.layers.Conv2D(filters=out_features, kernel_size=3, strides=1, padding="same")
        self.bn_01 = tf.keras.layers.BatchNormalization(axis=-1)
        self.bn_02 = tf.keras.layers.BatchNormalization(axis=-1)

    def call(self, inputs):
        # Forward path
        forward = self.conv1(inputs)
        forward = self.bn_01(forward)
        forward = tf.keras.layers.ReLU()(forward)
        forward = self.conv2(forward)
        forward = self.bn_02(forward)
        forward = tf.keras.layers.ReLU()(forward)
        
        # Add forward and original input to complete the residual layer
        return tf.keras.layers.add(inputs=[forward, inputs])
    
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
# param features:           list indicating the number of features in each set 
#                           of residual layers
# param layers:             list indicating the number of consecutive residual 
#                           layers before a transition occurs
# Parameter configurations that will be tested:
# (35, 16, [16, 24, 32], [2, 2, 2])
# (35, 16, [16, 24, 32], [2, 2, 1])
# (35, 16, [16, 24, 32], [2, 1, 1])
# (35, 16, [16, 24, 32], [1, 1, 1])
# (35, 16, [16, 24], [1, 1])
# (35, 16, [16], [1])
class ResNet(tf.keras.Model):
    def __init__(self, classes, initial_features, features, layers, **kwargs):
        super().__init__(**kwargs)
        
        # Save total number of residual layers
        self.layer_count = sum(layers)
        
        # Initial layer
        self.conv_01 = tf.keras.layers.Conv2D(filters=initial_features, kernel_size=7, strides=2, padding="same")
        self.bn_01 = tf.keras.layers.BatchNormalization(axis=-1)
        
        # Lists to hold variable number of residual layers
        self.conv_residual = []
        
        # Initial chain of residual layers do not reduce the feature size
        for i in range(layers[0]):
                self.conv_residual.append(ResidualConv(out_features=features[0]))
        
        # Add remaining residual layers
        for i in range(1, len(features)):
            self.conv_residual.append(ResidualConvReduce(out_features=features[i]))
            for j in range(layers[i]-1):
                self.conv_residual.append(ResidualConv(out_features=features[i]))
        
        # Global average pooling layer (reduces to 1x1xC)
        self.avg_pool = GlobalPoolLayer()
        
        # Fully connected layer with classes param outputs
        self.full_con = tf.keras.layers.Dense(units=classes, activation="softmax")

    def call(self, x):
        x = self.conv_01(x)
        x = self.bn_01(x)
        x = tf.keras.layers.ReLU()(x)
        
        for i in range(self.layer_count):
            x = self.conv_residual[i](x)
        
        x = self.avg_pool(x)
        x = tf.keras.layers.Flatten()(x)
        x = self.full_con(x)
        return x
    
    # To get model summary, use ResNet(params).model(input_shape).summary()
    # NOTE: Exclude batch dimension from input shape
    def model(self, input_shape):
        x = tf.keras.layers.Input(shape=input_shape)
        return tf.keras.Model(inputs=x, outputs=self.call(x))
