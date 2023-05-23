# -*- coding: utf-8 -*-
# DS-CNN Neural Network Models
import tensorflow as tf

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

# Full DS-CNN network 
# param classes:            number of dataset classes
# param initial_features:   number of features created by first convolution
# param features:           list indicating the number of features in each set 
#                           of depthwise separable convolutions
# param layers:             number of consecutive depthwise separable 
#                           convolutions before a transition occurs
# Parameter configurations that will be tested:
# (35, 16, [32, 64, 128], [1, 6, 2])
# (35, 16, [32, 64, 128], [1, 6, 1])
# (35, 16, [32, 64, 128], [1, 5, 1])
# (35, 16, [32, 64, 128], [1, 4, 1])
# (35, 16, [32, 64, 128], [1, 3, 1])
# (35, 16, [32, 64, 128], [1, 2, 1])
# (35, 16, [32, 64, 128], [1, 1, 1])
# (35, 16, [32, 64], [1, 1])
# (35, 16, [32], [1])
class DSCNN(tf.keras.Model):
    def __init__(self, classes, initial_features, features, layers, **kwargs):
        super().__init__(**kwargs)
        
        # Save total number of depthwise separable convolution layers
        self.layer_count = sum(layers)
        
        # Initial layer
        self.conv_01 = tf.keras.layers.Conv2D(filters=initial_features, kernel_size=3, strides=2, padding="same")
        self.bn_01 = tf.keras.layers.BatchNormalization(axis=-1)
        
        # Lists to hold variable number of convolution layers
        self.conv_dws = []
        self.conv_1x1 = []
        self.bn_dws = []
        self.bn_1x1 = []
        
        # Initial chain of convolutions does not have initial stride=2 layer
        for i in range(layers[0]):
                self.conv_dws.append(tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, padding="same"))
                self.conv_1x1.append(tf.keras.layers.Conv2D(filters=features[0], kernel_size=1, strides=1, padding="same"))
                self.bn_dws.append(tf.keras.layers.BatchNormalization(axis=-1))
                self.bn_1x1.append(tf.keras.layers.BatchNormalization(axis=-1))
        
        # Add remaining convolution layers
        for i in range(1, len(features)):
            self.conv_dws.append(tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=2, padding="same"))
            self.conv_1x1.append(tf.keras.layers.Conv2D(filters=features[i], kernel_size=1, strides=1, padding="same"))
            self.bn_dws.append(tf.keras.layers.BatchNormalization(axis=-1))
            self.bn_1x1.append(tf.keras.layers.BatchNormalization(axis=-1))
            for j in range(layers[i]-1):
                self.conv_dws.append(tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, padding="same"))
                self.conv_1x1.append(tf.keras.layers.Conv2D(filters=features[i], kernel_size=1, strides=1, padding="same"))
                self.bn_dws.append(tf.keras.layers.BatchNormalization(axis=-1))
                self.bn_1x1.append(tf.keras.layers.BatchNormalization(axis=-1))
        
        # Global average pooling layer (reduces to 1x1xC)
        self.avg_pool = GlobalPoolLayer()
        
        # Fully connected layer with classes param outputs
        self.full_con = tf.keras.layers.Dense(units=classes, activation="softmax")

    def call(self, x):
        x = self.conv_01(x)
        x = self.bn_01(x)
        x = tf.keras.layers.ReLU()(x)

        for i in range(self.layer_count):
            x = self.conv_dws[i](x)
            x = self.bn_dws[i](x)
            x = tf.keras.layers.ReLU()(x)
            
            x = self.conv_1x1[i](x)
            x = self.bn_1x1[i](x)
            x = tf.keras.layers.ReLU()(x)
            
        x = self.avg_pool(x)
        x = tf.keras.layers.Flatten()(x)
        x = self.full_con(x)
        
        return x
    
    # To get model summary, use DSCNN(params).model(input_shape).summary()
    # NOTE: Exclude batch dimension from input shape
    def model(self, input_shape):
        x = tf.keras.layers.Input(shape=input_shape)
        return tf.keras.Model(inputs=x, outputs=self.call(x))
