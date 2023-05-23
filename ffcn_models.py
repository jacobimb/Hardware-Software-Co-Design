# -*- coding: utf-8 -*-
# FFCN Neural Network Models
import tensorflow as tf

# Full ResNet network
# param classes:    number of dataset classes
# param units:      list indicating the number of neurons in each dense layer
# Parameter configurations that will be tested:
# (35, [48, 48, 48, 48, 48])
# (35, [48, 48, 48, 48])
# (35, [48, 48, 48])
# (35, [48, 48])
# (35, [48])
class FFCN(tf.keras.Model):
    def __init__(self, classes, units, **kwargs):
        super().__init__(**kwargs)
        
        # Record number of dense layers
        self.layer_count = len(units)
        
        # List to hold dense layers
        self.dense = []
        
        # Add each dense layer with number of units specified in units param
        for i in range(self.layer_count):
            self.dense.append(tf.keras.layers.Dense(units=units[i]))
            
        # Final classification layer
        self.classify = tf.keras.layers.Dense(units=classes, activation="softmax")

    def call(self, x):
        x = tf.keras.layers.Flatten()(x)
        
        for i in range(self.layer_count):
            x = self.dense[i](x)
            x = tf.keras.layers.ReLU()(x)
        
        x = self.classify(x)
        
        return x
    
    # To get model summary, use FFCN(params).model(input_shape).summary()
    # NOTE: Exclude batch dimension from input shape
    def model(self, input_shape):
        x = tf.keras.layers.Input(shape=input_shape)
        return tf.keras.Model(inputs=x, outputs=self.call(x))
