# -*- coding: utf-8 -*-
# Training Script
# Main script to train neural networks
# The following networks are currently supported:
# 1. Fully Connected Feedforward
# 2. MobileNet/DS-CNN
# 3. DesneNet
# 4. ResNet
# 5. Inception v1
# 6. CENet
# Networks are trained using Google's Speech Commands dataset and multiple 
# configurations of each network may be trained and tested.
import tensorflow as tf
import constants
import dataset
import dscnn_models
import densenet_models
import resnet_models
import ffcn_models
import inception_models
import cenet_models
import math
import os
import sys
import getopt

# Main model training function
# param category:   network architecture type
# param number:     architecture configuration to train
# param resume:     if training resumes from the last completed epoch
def train_model(category, number, resume=False):
    # Create datasets
    train_ds, test_ds, label_names = dataset.create_datasets(constants.DATASET_PATH, 
                                                             constants.BATCH_SIZE, 
                                                             constants.VALIDATION_SPLIT, 
                                                             constants.SAMPLE_RATE,
                                                             constants.AUDIO_LENGTH_SEC, 
                                                             constants.NFFT,
                                                             constants.STEP,
                                                             constants.MEL_BANKS,
                                                             constants.MEL_DB_MAX,
                                                             False,
                                                             False)

    # Calculate spectrogram shape for model creation
    spectrogram_shape = (math.ceil(constants.SAMPLE_RATE*constants.AUDIO_LENGTH_SEC/constants.STEP), constants.MEL_BANKS, 1)   
    
    # FFCN models
    if (category == 0):
        if (number == 0):
            if (resume):
                path = os.path.join("./checkpoints/ffcn/", str(number))
                epoch_start = len(os.listdir(path))-1
                
                path = os.path.join(path, str(epoch_start), "checkpoint")
                model = tf.keras.models.load_model(path)
            else:
                epoch_start = 0
                
                model = ffcn_models.FFCN(35, [48, 48, 48, 48, 48]).model(input_shape=spectrogram_shape)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                              metrics=["accuracy"])
            
            model.fit(x=train_ds,
                      epochs=constants.EPOCHS,
                      verbose=2,
                      callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoints/ffcn/0/{epoch}/checkpoint"),
                                 tf.keras.callbacks.CSVLogger(filename="./checkpoints/ffcn/0/log.log", append=True)],
                      validation_data=test_ds,
                      initial_epoch=epoch_start)
        elif (number == 1): 
            if (resume):
                path = os.path.join("./checkpoints/ffcn/", str(number))
                epoch_start = len(os.listdir(path))-1
                
                path = os.path.join(path, str(epoch_start), "checkpoint")
                model = tf.keras.models.load_model(path)
            else:
                epoch_start = 0
                
                model = ffcn_models.FFCN(35, [48, 48, 48, 48]).model(input_shape=spectrogram_shape)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                              metrics=["accuracy"])
            
            model.fit(x=train_ds,
                      epochs=constants.EPOCHS,
                      verbose=2,
                      callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoints/ffcn/1/{epoch}/checkpoint"),
                                 tf.keras.callbacks.CSVLogger(filename="./checkpoints/ffcn/1/log.log", append=True)],
                      validation_data=test_ds,
                      initial_epoch=epoch_start)
        elif (number == 2):
            if (resume):
                path = os.path.join("./checkpoints/ffcn/", str(number))
                epoch_start = len(os.listdir(path))-1
                
                path = os.path.join(path, str(epoch_start), "checkpoint")
                model = tf.keras.models.load_model(path)
            else:
                epoch_start = 0
                
                model = ffcn_models.FFCN(35, [48, 48, 48]).model(input_shape=spectrogram_shape)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                              metrics=["accuracy"])
            
            model.fit(x=train_ds,
                      epochs=constants.EPOCHS,
                      verbose=2,
                      callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoints/ffcn/2/{epoch}/checkpoint"),
                                 tf.keras.callbacks.CSVLogger(filename="./checkpoints/ffcn/2/log.log", append=True)],
                      validation_data=test_ds,
                      initial_epoch=epoch_start)
        elif (number == 3): 
            if (resume):
                path = os.path.join("./checkpoints/ffcn/", str(number))
                epoch_start = len(os.listdir(path))-1
                
                path = os.path.join(path, str(epoch_start), "checkpoint")
                model = tf.keras.models.load_model(path)
            else:
                epoch_start = 0
                
                model = ffcn_models.FFCN(35, [48, 48]).model(input_shape=spectrogram_shape)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                              metrics=["accuracy"])
            
            model.fit(x=train_ds,
                      epochs=constants.EPOCHS,
                      verbose=2,
                      callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoints/ffcn/3/{epoch}/checkpoint"),
                                 tf.keras.callbacks.CSVLogger(filename="./checkpoints/ffcn/3/log.log", append=True)],
                      validation_data=test_ds,
                      initial_epoch=epoch_start)
        elif (number == 4):
            if (resume):
                path = os.path.join("./checkpoints/ffcn/", str(number))
                epoch_start = len(os.listdir(path))-1
                
                path = os.path.join(path, str(epoch_start), "checkpoint")
                model = tf.keras.models.load_model(path)
            else:
                epoch_start = 0
                
                model = ffcn_models.FFCN(35, [48]).model(input_shape=spectrogram_shape)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                              metrics=["accuracy"])
            
            model.fit(x=train_ds,
                      epochs=constants.EPOCHS,
                      verbose=2,
                      callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoints/ffcn/4/{epoch}/checkpoint"),
                                 tf.keras.callbacks.CSVLogger(filename="./checkpoints/ffcn/4/log.log", append=True)],
                      validation_data=test_ds,
                      initial_epoch=epoch_start)
        else: 
            print("ERROR: INVALID MODEL SELECTION!")
            return
    
    # DSCNN models
    elif (category == 1):            
        if (number == 0):
            if (resume):
                path = os.path.join("./checkpoints/dscnn/", str(number))
                epoch_start = len(os.listdir(path))-1
                
                path = os.path.join(path, str(epoch_start), "checkpoint")
                model = tf.keras.models.load_model(path)
            else:
                epoch_start = 0
                
                model = dscnn_models.DSCNN(35, 16, [32, 64, 128], [1, 6, 2]).model(input_shape=spectrogram_shape)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                              metrics=["accuracy"])
            
            model.fit(x=train_ds,
                      epochs=constants.EPOCHS,
                      verbose=2,
                      callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoints/dscnn/0/{epoch}/checkpoint"),
                                 tf.keras.callbacks.CSVLogger(filename="./checkpoints/dscnn/0/log.log", append=True)],
                      validation_data=test_ds,
                      initial_epoch=epoch_start)
        elif (number == 1):
            if (resume):
                path = os.path.join("./checkpoints/dscnn/", str(number))
                epoch_start = len(os.listdir(path))-1
                
                path = os.path.join(path, str(epoch_start), "checkpoint")
                model = tf.keras.models.load_model(path)
            else:
                epoch_start = 0
                
                model = dscnn_models.DSCNN(35, 16, [32, 64, 128], [1, 6, 1]).model(input_shape=spectrogram_shape)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                              metrics=["accuracy"])
            
            model.fit(x=train_ds,
                      epochs=constants.EPOCHS,
                      verbose=2,
                      callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoints/dscnn/1/{epoch}/checkpoint"),
                                 tf.keras.callbacks.CSVLogger(filename="./checkpoints/dscnn/1/log.log", append=True)],
                      validation_data=test_ds,
                      initial_epoch=epoch_start)
        elif (number == 2): 
            if (resume):
                path = os.path.join("./checkpoints/dscnn/", str(number))
                epoch_start = len(os.listdir(path))-1
                
                path = os.path.join(path, str(epoch_start), "checkpoint")
                model = tf.keras.models.load_model(path)
            else:
                epoch_start = 0
                
                model = dscnn_models.DSCNN(35, 16, [32, 64, 128], [1, 5, 1]).model(input_shape=spectrogram_shape)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                              metrics=["accuracy"])
            
            model.fit(x=train_ds,
                      epochs=constants.EPOCHS,
                      verbose=2,
                      callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoints/dscnn/2/{epoch}/checkpoint"),
                                 tf.keras.callbacks.CSVLogger(filename="./checkpoints/dscnn/2/log.log", append=True)],
                      validation_data=test_ds,
                      initial_epoch=epoch_start)
        elif (number == 3): 
            if (resume):
                path = os.path.join("./checkpoints/dscnn/", str(number))
                epoch_start = len(os.listdir(path))-1
                
                path = os.path.join(path, str(epoch_start), "checkpoint")
                model = tf.keras.models.load_model(path)
            else:
                epoch_start = 0
                
                model = dscnn_models.DSCNN(35, 16, [32, 64, 128], [1, 4, 1]).model(input_shape=spectrogram_shape)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                              metrics=["accuracy"])
            
            model.fit(x=train_ds,
                      epochs=constants.EPOCHS,
                      verbose=2,
                      callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoints/dscnn/3/{epoch}/checkpoint"),
                                 tf.keras.callbacks.CSVLogger(filename="./checkpoints/dscnn/3/log.log", append=True)],
                      validation_data=test_ds,
                      initial_epoch=epoch_start)
        elif (number == 4): 
            if (resume):
                path = os.path.join("./checkpoints/dscnn/", str(number))
                epoch_start = len(os.listdir(path))-1
                
                path = os.path.join(path, str(epoch_start), "checkpoint")
                model = tf.keras.models.load_model(path)
            else:
                epoch_start = 0
                
                model = dscnn_models.DSCNN(35, 16, [32, 64, 128], [1, 3, 1]).model(input_shape=spectrogram_shape)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                              metrics=["accuracy"])
            
            model.fit(x=train_ds,
                      epochs=constants.EPOCHS,
                      verbose=2,
                      callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoints/dscnn/4/{epoch}/checkpoint"),
                                 tf.keras.callbacks.CSVLogger(filename="./checkpoints/dscnn/4/log.log", append=True)],
                      validation_data=test_ds,
                      initial_epoch=epoch_start)
        elif (number == 5): 
            if (resume):
                path = os.path.join("./checkpoints/dscnn/", str(number))
                epoch_start = len(os.listdir(path))-1
                
                path = os.path.join(path, str(epoch_start), "checkpoint")
                model = tf.keras.models.load_model(path)
            else:
                epoch_start = 0
                
                model = dscnn_models.DSCNN(35, 16, [32, 64, 128], [1, 2, 1]).model(input_shape=spectrogram_shape)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                              metrics=["accuracy"])
            
            model.fit(x=train_ds,
                      epochs=constants.EPOCHS,
                      verbose=2,
                      callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoints/dscnn/5/{epoch}/checkpoint"),
                                 tf.keras.callbacks.CSVLogger(filename="./checkpoints/dscnn/5/log.log", append=True)],
                      validation_data=test_ds,
                      initial_epoch=epoch_start)
        elif (number == 6):
            if (resume):
                path = os.path.join("./checkpoints/dscnn/", str(number))
                epoch_start = len(os.listdir(path))-1
                
                path = os.path.join(path, str(epoch_start), "checkpoint")
                model = tf.keras.models.load_model(path)
            else:
                epoch_start = 0
                
                model = dscnn_models.DSCNN(35, 16, [32, 64, 128], [1, 1, 1]).model(input_shape=spectrogram_shape)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                              metrics=["accuracy"])
            
            model.fit(x=train_ds,
                      epochs=constants.EPOCHS,
                      verbose=2,
                      callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoints/dscnn/6/{epoch}/checkpoint"),
                                 tf.keras.callbacks.CSVLogger(filename="./checkpoints/dscnn/6/log.log", append=True)],
                      validation_data=test_ds,
                      initial_epoch=epoch_start)
        elif (number == 7): 
            if (resume):
                path = os.path.join("./checkpoints/dscnn/", str(number))
                epoch_start = len(os.listdir(path))-1
                
                path = os.path.join(path, str(epoch_start), "checkpoint")
                model = tf.keras.models.load_model(path)
            else:
                epoch_start = 0
                
            model = dscnn_models.DSCNN(35, 16, [32, 64], [1, 1]).model(input_shape=spectrogram_shape)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                          metrics=["accuracy"])
            
            model.fit(x=train_ds,
                      epochs=constants.EPOCHS,
                      verbose=2,
                      callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoints/dscnn/7/{epoch}/checkpoint"),
                                 tf.keras.callbacks.CSVLogger(filename="./checkpoints/dscnn/7/log.log", append=True)],
                      validation_data=test_ds,
                      initial_epoch=epoch_start)
        elif (number == 8): 
            if (resume):
                path = os.path.join("./checkpoints/dscnn/", str(number))
                epoch_start = len(os.listdir(path))-1
                
                path = os.path.join(path, str(epoch_start), "checkpoint")
                model = tf.keras.models.load_model(path)
            else:
                epoch_start = 0
                
                model = dscnn_models.DSCNN(35, 16, [32], [1]).model(input_shape=spectrogram_shape)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                              metrics=["accuracy"])
            
            model.fit(x=train_ds,
                      epochs=constants.EPOCHS,
                      verbose=2,
                      callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoints/dscnn/8/{epoch}/checkpoint"),
                                 tf.keras.callbacks.CSVLogger(filename="./checkpoints/dscnn/8/log.log", append=True)],
                      validation_data=test_ds,
                      initial_epoch=epoch_start)
        else: 
            print("ERROR: INVALID MODEL SELECTION!")
            return
        
    # DenseNet models
    elif (category == 2):
        if (number == 0): 
            if (resume):
                path = os.path.join("./checkpoints/densenet/", str(number))
                epoch_start = len(os.listdir(path))-1
                
                path = os.path.join(path, str(epoch_start), "checkpoint")
                model = tf.keras.models.load_model(path)
            else:
                epoch_start = 0
                
                model = densenet_models.DenseNet(35, 16, 4, [2, 4, 8, 6], 32, 8, 0.5).model(input_shape=spectrogram_shape)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                              metrics=["accuracy"])
            
            model.fit(x=train_ds,
                      epochs=constants.EPOCHS,
                      verbose=2,
                      callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoints/densenet/0/{epoch}/checkpoint"),
                                 tf.keras.callbacks.CSVLogger(filename="./checkpoints/densenet/0/log.log", append=True)],
                      validation_data=test_ds,
                      initial_epoch=epoch_start)
        elif (number == 1): 
            if (resume):
                path = os.path.join("./checkpoints/densenet/", str(number))
                epoch_start = len(os.listdir(path))-1
                
                path = os.path.join(path, str(epoch_start), "checkpoint")
                model = tf.keras.models.load_model(path)
            else:
                epoch_start = 0
                
                model = densenet_models.DenseNet(35, 16, 4, [2, 4, 8, 5], 32, 8, 0.5).model(input_shape=spectrogram_shape)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                              metrics=["accuracy"])
                
            model.fit(x=train_ds,
                      epochs=constants.EPOCHS,
                      verbose=2,
                      callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoints/densenet/1/{epoch}/checkpoint"),
                                 tf.keras.callbacks.CSVLogger(filename="./checkpoints/densenet/1/log.log", append=True)],
                      validation_data=test_ds,
                      initial_epoch=epoch_start)
        elif (number == 2): 
            if (resume):
                path = os.path.join("./checkpoints/densenet/", str(number))
                epoch_start = len(os.listdir(path))-1
                
                path = os.path.join(path, str(epoch_start), "checkpoint")
                model = tf.keras.models.load_model(path)
            else:
                epoch_start = 0
                
                model = densenet_models.DenseNet(35, 16, 4, [2, 4, 8, 4], 32, 8, 0.5).model(input_shape=spectrogram_shape)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                              metrics=["accuracy"])
            
            model.fit(x=train_ds,
                      epochs=constants.EPOCHS,
                      verbose=2,
                      callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoints/densenet/2/{epoch}/checkpoint"),
                                 tf.keras.callbacks.CSVLogger(filename="./checkpoints/densenet/2/log.log", append=True)],
                      validation_data=test_ds,
                      initial_epoch=epoch_start)
        elif (number == 3): 
            if (resume):
                path = os.path.join("./checkpoints/densenet/", str(number))
                epoch_start = len(os.listdir(path))-1
                
                path = os.path.join(path, str(epoch_start), "checkpoint")
                model = tf.keras.models.load_model(path)
            else:
                epoch_start = 0
                
                model = densenet_models.DenseNet(35, 16, 4, [2, 4, 8, 3], 32, 8, 0.5).model(input_shape=spectrogram_shape)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                              metrics=["accuracy"])
            
            model.fit(x=train_ds,
                      epochs=constants.EPOCHS,
                      verbose=2,
                      callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoints/densenet/3/{epoch}/checkpoint"),
                                 tf.keras.callbacks.CSVLogger(filename="./checkpoints/densenet/3/log.log", append=True)],
                      validation_data=test_ds,
                      initial_epoch=epoch_start)
        elif (number == 4): 
            if (resume):
                path = os.path.join("./checkpoints/densenet/", str(number))
                epoch_start = len(os.listdir(path))-1
                
                path = os.path.join(path, str(epoch_start), "checkpoint")
                model = tf.keras.models.load_model(path)
            else:
                epoch_start = 0
                
                model = densenet_models.DenseNet(35, 16, 4, [2, 4, 8, 2], 32, 8, 0.5).model(input_shape=spectrogram_shape)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                              metrics=["accuracy"])
            
            model.fit(x=train_ds,
                      epochs=constants.EPOCHS,
                      verbose=2,
                      callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoints/densenet/4/{epoch}/checkpoint"),
                                 tf.keras.callbacks.CSVLogger(filename="./checkpoints/densenet/4/log.log", append=True)],
                      validation_data=test_ds,
                      initial_epoch=epoch_start)
        elif (number == 5):
            if (resume):
                path = os.path.join("./checkpoints/densenet/", str(number))
                epoch_start = len(os.listdir(path))-1
                
                path = os.path.join(path, str(epoch_start), "checkpoint")
                model = tf.keras.models.load_model(path)
            else:
                epoch_start = 0
                
                model = densenet_models.DenseNet(35, 16, 4, [2, 4, 8, 1], 32, 8, 0.5).model(input_shape=spectrogram_shape)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                              metrics=["accuracy"])
            
            model.fit(x=train_ds,
                      epochs=constants.EPOCHS,
                      verbose=2,
                      callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoints/densenet/5/{epoch}/checkpoint"),
                                 tf.keras.callbacks.CSVLogger(filename="./checkpoints/densenet/5/log.log", append=True)],
                      validation_data=test_ds,
                      initial_epoch=epoch_start)
        elif (number == 6):
            if (resume):
                path = os.path.join("./checkpoints/densenet/", str(number))
                epoch_start = len(os.listdir(path))-1
                
                path = os.path.join(path, str(epoch_start), "checkpoint")
                model = tf.keras.models.load_model(path)
            else:
                epoch_start = 0
                
                model = densenet_models.DenseNet(35, 16, 4, [2, 4, 7, 1], 32, 8, 0.5).model(input_shape=spectrogram_shape)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                              metrics=["accuracy"])
            
            model.fit(x=train_ds,
                      epochs=constants.EPOCHS,
                      verbose=2,
                      callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoints/densenet/6/{epoch}/checkpoint"),
                                 tf.keras.callbacks.CSVLogger(filename="./checkpoints/densenet/6/log.log", append=True)],
                      validation_data=test_ds,
                      initial_epoch=epoch_start)
        elif (number == 7): 
            if (resume):
                path = os.path.join("./checkpoints/densenet/", str(number))
                epoch_start = len(os.listdir(path))-1
                
                path = os.path.join(path, str(epoch_start), "checkpoint")
                model = tf.keras.models.load_model(path)
            else:
                epoch_start = 0
                
                model = densenet_models.DenseNet(35, 16, 4, [2, 4, 6, 1], 32, 8, 0.5).model(input_shape=spectrogram_shape)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                              metrics=["accuracy"])
            
            model.fit(x=train_ds,
                      epochs=constants.EPOCHS,
                      verbose=2,
                      callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoints/densenet/7/{epoch}/checkpoint"),
                                 tf.keras.callbacks.CSVLogger(filename="./checkpoints/densenet/7/log.log", append=True)],
                      validation_data=test_ds,
                      initial_epoch=epoch_start)
        elif (number == 8): 
            if (resume):
                path = os.path.join("./checkpoints/densenet/", str(number))
                epoch_start = len(os.listdir(path))-1
                
                path = os.path.join(path, str(epoch_start), "checkpoint")
                model = tf.keras.models.load_model(path)
            else:
                epoch_start = 0
                
                model = densenet_models.DenseNet(35, 16, 4, [2, 4, 5, 1], 32, 8, 0.5).model(input_shape=spectrogram_shape)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                              metrics=["accuracy"])
            
            model.fit(x=train_ds,
                      epochs=constants.EPOCHS,
                      verbose=2,
                      callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoints/densenet/8/{epoch}/checkpoint"),
                                 tf.keras.callbacks.CSVLogger(filename="./checkpoints/densenet/8/log.log", append=True)],
                      validation_data=test_ds,
                      initial_epoch=epoch_start)
        elif (number == 9):
            if (resume):
                path = os.path.join("./checkpoints/densenet/", str(number))
                epoch_start = len(os.listdir(path))-1
                
                path = os.path.join(path, str(epoch_start), "checkpoint")
                model = tf.keras.models.load_model(path)
            else:
                epoch_start = 0
                
                model = densenet_models.DenseNet(35, 16, 4, [2, 4, 4, 1], 32, 8, 0.5).model(input_shape=spectrogram_shape)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                              metrics=["accuracy"])
            
            model.fit(x=train_ds,
                      epochs=constants.EPOCHS,
                      verbose=2,
                      callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoints/densenet/9/{epoch}/checkpoint"),
                                 tf.keras.callbacks.CSVLogger(filename="./checkpoints/densenet/9/log.log", append=True)],
                      validation_data=test_ds,
                      initial_epoch=epoch_start)
        elif (number == 10):
            if (resume):
                path = os.path.join("./checkpoints/densenet/", str(number))
                epoch_start = len(os.listdir(path))-1
                
                path = os.path.join(path, str(epoch_start), "checkpoint")
                model = tf.keras.models.load_model(path)
            else:
                epoch_start = 0
                
                model = densenet_models.DenseNet(35, 16, 4, [2, 4, 3, 1], 32, 8, 0.5).model(input_shape=spectrogram_shape)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                              metrics=["accuracy"])
            
            model.fit(x=train_ds,
                      epochs=constants.EPOCHS,
                      verbose=2,
                      callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoints/densenet/10/{epoch}/checkpoint"),
                                 tf.keras.callbacks.CSVLogger(filename="./checkpoints/densenet/10/log.log", append=True)],
                      validation_data=test_ds,
                      initial_epoch=epoch_start)
        elif (number == 11):
            if (resume):
                path = os.path.join("./checkpoints/densenet/", str(number))
                epoch_start = len(os.listdir(path))-1
                
                path = os.path.join(path, str(epoch_start), "checkpoint")
                model = tf.keras.models.load_model(path)
            else:
                epoch_start = 0
                
                model = densenet_models.DenseNet(35, 16, 4, [2, 4, 2, 1], 32, 8, 0.5).model(input_shape=spectrogram_shape)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                              metrics=["accuracy"])
                
            model.fit(x=train_ds,
                      epochs=constants.EPOCHS,
                      verbose=2,
                      callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoints/densenet/11/{epoch}/checkpoint"),
                                 tf.keras.callbacks.CSVLogger(filename="./checkpoints/densenet/11/log.log", append=True)],
                      validation_data=test_ds,
                      initial_epoch=epoch_start)
        elif (number == 12): 
            if (resume):
                path = os.path.join("./checkpoints/densenet/", str(number))
                epoch_start = len(os.listdir(path))-1
                
                path = os.path.join(path, str(epoch_start), "checkpoint")
                model = tf.keras.models.load_model(path)
            else:
                epoch_start = 0
                
                model = densenet_models.DenseNet(35, 16, 4, [2, 4, 1, 1], 32, 8, 0.5).model(input_shape=spectrogram_shape)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                              metrics=["accuracy"])
            
            model.fit(x=train_ds,
                      epochs=constants.EPOCHS,
                      verbose=2,
                      callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoints/densenet/12/{epoch}/checkpoint"),
                                 tf.keras.callbacks.CSVLogger(filename="./checkpoints/densenet/12/log.log", append=True)],
                      validation_data=test_ds,
                      initial_epoch=epoch_start)
        elif (number == 13):
            if (resume):
                path = os.path.join("./checkpoints/densenet/", str(number))
                epoch_start = len(os.listdir(path))-1
                
                path = os.path.join(path, str(epoch_start), "checkpoint")
                model = tf.keras.models.load_model(path)
            else:
                epoch_start = 0
                
                model = densenet_models.DenseNet(35, 16, 4, [2, 3, 1, 1], 32, 8, 0.5).model(input_shape=spectrogram_shape)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                              metrics=["accuracy"])
            
            model.fit(x=train_ds,
                      epochs=constants.EPOCHS,
                      verbose=2,
                      callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoints/densenet/13/{epoch}/checkpoint"),
                                 tf.keras.callbacks.CSVLogger(filename="./checkpoints/densenet/13/log.log", append=True)],
                      validation_data=test_ds,
                      initial_epoch=epoch_start)
        elif (number == 14): 
            if (resume):
                path = os.path.join("./checkpoints/densenet/", str(number))
                epoch_start = len(os.listdir(path))-1
                
                path = os.path.join(path, str(epoch_start), "checkpoint")
                model = tf.keras.models.load_model(path)
            else:
                epoch_start = 0
                
                model = densenet_models.DenseNet(35, 16, 4, [2, 2, 1, 1], 32, 8, 0.5).model(input_shape=spectrogram_shape)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                          metrics=["accuracy"])
            
            model.fit(x=train_ds,
                      epochs=constants.EPOCHS,
                      verbose=2,
                      callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoints/densenet/14/{epoch}/checkpoint"),
                                 tf.keras.callbacks.CSVLogger(filename="./checkpoints/densenet/14/log.log", append=True)],
                      validation_data=test_ds,
                      initial_epoch=epoch_start)
        elif (number == 15): 
            if (resume):
                path = os.path.join("./checkpoints/densenet/", str(number))
                epoch_start = len(os.listdir(path))-1
                
                path = os.path.join(path, str(epoch_start), "checkpoint")
                model = tf.keras.models.load_model(path)
            else:
                epoch_start = 0
                
                model = densenet_models.DenseNet(35, 16, 4, [2, 1, 1, 1], 32, 8, 0.5).model(input_shape=spectrogram_shape)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                          metrics=["accuracy"])
            
            model.fit(x=train_ds,
                      epochs=constants.EPOCHS,
                      verbose=2,
                      callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoints/densenet/15/{epoch}/checkpoint"),
                                 tf.keras.callbacks.CSVLogger(filename="./checkpoints/densenet/15/log.log", append=True)],
                      validation_data=test_ds,
                      initial_epoch=epoch_start)
        elif (number == 16): 
            if (resume):
                path = os.path.join("./checkpoints/densenet/", str(number))
                epoch_start = len(os.listdir(path))-1
                
                path = os.path.join(path, str(epoch_start), "checkpoint")
                model = tf.keras.models.load_model(path)
            else:
                epoch_start = 0
                
                model = densenet_models.DenseNet(35, 16, 4, [1, 1, 1, 1], 32, 8, 0.5).model(input_shape=spectrogram_shape)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                          metrics=["accuracy"])
            
            model.fit(x=train_ds,
                      epochs=constants.EPOCHS,
                      verbose=2,
                      callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoints/densenet/16/{epoch}/checkpoint"),
                                 tf.keras.callbacks.CSVLogger(filename="./checkpoints/densenet/16/log.log", append=True)],
                      validation_data=test_ds,
                      initial_epoch=epoch_start)
        elif (number == 17): 
            if (resume):
                path = os.path.join("./checkpoints/densenet/", str(number))
                epoch_start = len(os.listdir(path))-1
                
                path = os.path.join(path, str(epoch_start), "checkpoint")
                model = tf.keras.models.load_model(path)
            else:
                epoch_start = 0
                
                model = densenet_models.DenseNet(35, 16, 3, [1, 1, 1], 32, 8, 0.5).model(input_shape=spectrogram_shape)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                          metrics=["accuracy"])
            
            model.fit(x=train_ds,
                      epochs=constants.EPOCHS,
                      verbose=2,
                      callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoints/densenet/17/{epoch}/checkpoint"),
                                 tf.keras.callbacks.CSVLogger(filename="./checkpoints/densenet/17/log.log", append=True)],
                      validation_data=test_ds,
                      initial_epoch=epoch_start)
        elif (number == 18): 
            if (resume):
                path = os.path.join("./checkpoints/densenet/", str(number))
                epoch_start = len(os.listdir(path))-1
                
                path = os.path.join(path, str(epoch_start), "checkpoint")
                model = tf.keras.models.load_model(path)
            else:
                epoch_start = 0
                
                model = densenet_models.DenseNet(35, 16, 2, [1, 1], 32, 8, 0.5).model(input_shape=spectrogram_shape)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                              metrics=["accuracy"])
            
            model.fit(x=train_ds,
                      epochs=constants.EPOCHS,
                      verbose=2,
                      callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoints/densenet/18/{epoch}/checkpoint"),
                                 tf.keras.callbacks.CSVLogger(filename="./checkpoints/densenet/18/log.log", append=True)],
                      validation_data=test_ds,
                      initial_epoch=epoch_start)
        elif (number == 19):
            if (resume):
                path = os.path.join("./checkpoints/densenet/", str(number))
                epoch_start = len(os.listdir(path))-1
                
                path = os.path.join(path, str(epoch_start), "checkpoint")
                model = tf.keras.models.load_model(path)
            else:
                epoch_start = 0
                
                model = densenet_models.DenseNet(35, 16, 1, [1], 32, 8, 0.5).model(input_shape=spectrogram_shape)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                              metrics=["accuracy"])
            
            model.fit(x=train_ds,
                      epochs=constants.EPOCHS,
                      verbose=2,
                      callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoints/densenet/19/{epoch}/checkpoint"),
                                 tf.keras.callbacks.CSVLogger(filename="./checkpoints/densenet/19/log.log", append=True)],
                      validation_data=test_ds,
                      initial_epoch=epoch_start)
        else: 
            print("ERROR: INVALID MODEL SELECTION!")
            return
        
    # ResNet models
    elif (category == 3):
        if (number == 0): 
            if (resume):
                path = os.path.join("./checkpoints/resnet/", str(number))
                epoch_start = len(os.listdir(path))-1
                
                path = os.path.join(path, str(epoch_start), "checkpoint")
                model = tf.keras.models.load_model(path)
            else:
                epoch_start = 0
                
                model = resnet_models.ResNet(35, 16, [16, 24, 32], [2, 2, 2]).model(input_shape=spectrogram_shape)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                              metrics=["accuracy"])
            
            model.fit(x=train_ds,
                      epochs=constants.EPOCHS,
                      verbose=2,
                      callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoints/resnet/0/{epoch}/checkpoint"),
                                 tf.keras.callbacks.CSVLogger(filename="./checkpoints/resnet/0/log.log", append=True)],
                      validation_data=test_ds,
                      initial_epoch=epoch_start)
        elif (number == 1): 
            if (resume):
                path = os.path.join("./checkpoints/resnet/", str(number))
                epoch_start = len(os.listdir(path))-1
                
                path = os.path.join(path, str(epoch_start), "checkpoint")
                model = tf.keras.models.load_model(path)
            else:
                epoch_start = 0
                
                model = resnet_models.ResNet(35, 16, [16, 24, 32], [2, 2, 1]).model(input_shape=spectrogram_shape)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                              metrics=["accuracy"])
            
            model.fit(x=train_ds,
                      epochs=constants.EPOCHS,
                      verbose=2,
                      callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoints/resnet/1/{epoch}/checkpoint"),
                                 tf.keras.callbacks.CSVLogger(filename="./checkpoints/resnet/1/log.log", append=True)],
                      validation_data=test_ds,
                      initial_epoch=epoch_start)
        elif (number == 2): 
            if (resume):
                path = os.path.join("./checkpoints/resnet/", str(number))
                epoch_start = len(os.listdir(path))-1
                
                path = os.path.join(path, str(epoch_start), "checkpoint")
                model = tf.keras.models.load_model(path)
            else:
                epoch_start = 0
                
                model = resnet_models.ResNet(35, 16, [16, 24, 32], [2, 1, 1]).model(input_shape=spectrogram_shape)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                              metrics=["accuracy"])
                
            model.fit(x=train_ds,
                      epochs=constants.EPOCHS,
                      verbose=2,
                      callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoints/resnet/2/{epoch}/checkpoint"),
                                 tf.keras.callbacks.CSVLogger(filename="./checkpoints/resnet/2/log.log", append=True)],
                      validation_data=test_ds,
                      initial_epoch=epoch_start)
        elif (number == 3): 
            if (resume):
                path = os.path.join("./checkpoints/resnet/", str(number))
                epoch_start = len(os.listdir(path))-1
                
                path = os.path.join(path, str(epoch_start), "checkpoint")
                model = tf.keras.models.load_model(path)
            else:
                epoch_start = 0
                
                model = resnet_models.ResNet(35, 16, [16, 24, 32], [1, 1, 1]).model(input_shape=spectrogram_shape)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                              metrics=["accuracy"])
                
            model.fit(x=train_ds,
                      epochs=constants.EPOCHS,
                      verbose=2,
                      callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoints/resnet/3/{epoch}/checkpoint"),
                                 tf.keras.callbacks.CSVLogger(filename="./checkpoints/resnet/3/log.log", append=True)],
                      validation_data=test_ds,
                      initial_epoch=epoch_start)
        elif (number == 4): 
            if (resume):
                path = os.path.join("./checkpoints/resnet/", str(number))
                epoch_start = len(os.listdir(path))-1
                
                path = os.path.join(path, str(epoch_start), "checkpoint")
                model = tf.keras.models.load_model(path)
            else:
                epoch_start = 0
                
                model = resnet_models.ResNet(35, 16, [16, 24], [1, 1]).model(input_shape=spectrogram_shape)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                              metrics=["accuracy"])
            
            model.fit(x=train_ds,
                      epochs=constants.EPOCHS,
                      verbose=2,
                      callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoints/resnet/4/{epoch}/checkpoint"),
                                 tf.keras.callbacks.CSVLogger(filename="./checkpoints/resnet/4/log.log", append=True)],
                      validation_data=test_ds,
                      initial_epoch=epoch_start)
        elif (number == 5): 
            if (resume):
                path = os.path.join("./checkpoints/resnet/", str(number))
                epoch_start = len(os.listdir(path))-1
                
                path = os.path.join(path, str(epoch_start), "checkpoint")
                model = tf.keras.models.load_model(path)
            else:
                epoch_start = 0
                
                model = resnet_models.ResNet(35, 16, [16], [1]).model(input_shape=spectrogram_shape)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                              metrics=["accuracy"])
            
            model.fit(x=train_ds,
                      epochs=constants.EPOCHS,
                      verbose=2,
                      callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoints/resnet/5/{epoch}/checkpoint"),
                                 tf.keras.callbacks.CSVLogger(filename="./checkpoints/resnet/5/log.log", append=True)],
                      validation_data=test_ds,
                      initial_epoch=epoch_start)
        else: 
            print("ERROR: INVALID MODEL SELECTION!")
            return
        
    # Inception v1 models
    elif (category == 4):    
        if (number == 0): 
            if (resume):
                path = os.path.join("./checkpoints/inception/", str(number))
                epoch_start = len(os.listdir(path))-1
                
                path = os.path.join(path, str(epoch_start), "checkpoint")
                model = tf.keras.models.load_model(path)
            else:
                epoch_start = 0
                
                model = inception_models.InceptionV1(35, 16, [2, 4, 2], [[8, 12, 16, 2, 4, 4], [24, 12, 26, 2, 6, 8], [48, 24, 48, 6, 16, 16]]).model(input_shape=spectrogram_shape)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                              metrics=["accuracy"])
            
            model.fit(x=train_ds,
                      epochs=constants.EPOCHS,
                      verbose=2,
                      callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoints/inception/0/{epoch}/checkpoint"),
                                 tf.keras.callbacks.CSVLogger(filename="./checkpoints/inception/0/log.log", append=True)],
                      validation_data=test_ds,
                      initial_epoch=epoch_start)
        elif (number == 1): 
            if (resume):
                path = os.path.join("./checkpoints/inception/", str(number))
                epoch_start = len(os.listdir(path))-1
                
                path = os.path.join(path, str(epoch_start), "checkpoint")
                model = tf.keras.models.load_model(path)
            else:
                epoch_start = 0
                
                model = inception_models.InceptionV1(35, 16, [2, 4, 1], [[8, 12, 16, 2, 4, 4], [24, 12, 26, 2, 6, 8], [48, 24, 48, 6, 16, 16]]).model(input_shape=spectrogram_shape)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                              metrics=["accuracy"])
            
            model.fit(x=train_ds,
                      epochs=constants.EPOCHS,
                      verbose=2,
                      callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoints/inception/1/{epoch}/checkpoint"),
                                 tf.keras.callbacks.CSVLogger(filename="./checkpoints/inception/1/log.log", append=True)],
                      validation_data=test_ds,
                      initial_epoch=epoch_start)
        elif (number == 2): 
            if (resume):
                path = os.path.join("./checkpoints/inception/", str(number))
                epoch_start = len(os.listdir(path))-1
                
                path = os.path.join(path, str(epoch_start), "checkpoint")
                model = tf.keras.models.load_model(path)
            else:
                epoch_start = 0
                
                model = inception_models.InceptionV1(35, 16, [2, 3, 1], [[8, 12, 16, 2, 4, 4], [24, 12, 26, 2, 6, 8], [48, 24, 48, 6, 16, 16]]).model(input_shape=spectrogram_shape)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                              metrics=["accuracy"])
            
            model.fit(x=train_ds,
                      epochs=constants.EPOCHS,
                      verbose=2,
                      callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoints/inception/2/{epoch}/checkpoint"),
                                 tf.keras.callbacks.CSVLogger(filename="./checkpoints/inception/2/log.log", append=True)],
                      validation_data=test_ds,
                      initial_epoch=epoch_start)
        elif (number == 3): 
            if (resume):
                path = os.path.join("./checkpoints/inception/", str(number))
                epoch_start = len(os.listdir(path))-1
                
                path = os.path.join(path, str(epoch_start), "checkpoint")
                model = tf.keras.models.load_model(path)
            else:
                epoch_start = 0
                
                model = inception_models.InceptionV1(35, 16, [2, 2, 1], [[8, 12, 16, 2, 4, 4], [24, 12, 26, 2, 6, 8], [48, 24, 48, 6, 16, 16]]).model(input_shape=spectrogram_shape)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                              metrics=["accuracy"])
                
            model.fit(x=train_ds,
                      epochs=constants.EPOCHS,
                      verbose=2,
                      callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoints/inception/3/{epoch}/checkpoint"),
                                 tf.keras.callbacks.CSVLogger(filename="./checkpoints/inception/3/log.log", append=True)],
                      validation_data=test_ds,
                      initial_epoch=epoch_start)
        elif (number == 4): 
            if (resume):
                path = os.path.join("./checkpoints/inception/", str(number))
                epoch_start = len(os.listdir(path))-1
                
                path = os.path.join(path, str(epoch_start), "checkpoint")
                model = tf.keras.models.load_model(path)
            else:
                epoch_start = 0
                
                model = inception_models.InceptionV1(35, 16, [2, 1, 1], [[8, 12, 16, 2, 4, 4], [24, 12, 26, 2, 6, 8], [48, 24, 48, 6, 16, 16]]).model(input_shape=spectrogram_shape)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                              metrics=["accuracy"])
            
            model.fit(x=train_ds,
                      epochs=constants.EPOCHS,
                      verbose=2,
                      callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoints/inception/4/{epoch}/checkpoint"),
                                 tf.keras.callbacks.CSVLogger(filename="./checkpoints/inception/4/log.log", append=True)],
                      validation_data=test_ds,
                      initial_epoch=epoch_start)
        elif (number == 5): 
            if (resume):
                path = os.path.join("./checkpoints/inception/", str(number))
                epoch_start = len(os.listdir(path))-1
                
                path = os.path.join(path, str(epoch_start), "checkpoint")
                model = tf.keras.models.load_model(path)
            else:
                epoch_start = 0
                
                model = inception_models.InceptionV1(35, 16, [1, 1, 1], [[8, 12, 16, 2, 4, 4], [24, 12, 26, 2, 6, 8], [48, 24, 48, 6, 16, 16]]).model(input_shape=spectrogram_shape)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                              metrics=["accuracy"])
            
            model.fit(x=train_ds,
                      epochs=constants.EPOCHS,
                      verbose=2,
                      callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoints/inception/5/{epoch}/checkpoint"),
                                 tf.keras.callbacks.CSVLogger(filename="./checkpoints/inception/5/log.log", append=True)],
                      validation_data=test_ds,
                      initial_epoch=epoch_start)
        elif (number == 6): 
            if (resume):
                path = os.path.join("./checkpoints/inception/", str(number))
                epoch_start = len(os.listdir(path))-1
                
                path = os.path.join(path, str(epoch_start), "checkpoint")
                model = tf.keras.models.load_model(path)
            else:
                epoch_start = 0
                
                model = inception_models.InceptionV1(35, 16, [1, 1], [[8, 12, 16, 2, 4, 4], [24, 12, 26, 2, 6, 8]]).model(input_shape=spectrogram_shape)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                              metrics=["accuracy"])
            
            model.fit(x=train_ds,
                      epochs=constants.EPOCHS,
                      verbose=2,
                      callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoints/inception/6/{epoch}/checkpoint"),
                                 tf.keras.callbacks.CSVLogger(filename="./checkpoints/inception/6/log.log", append=True)],
                      validation_data=test_ds,
                      initial_epoch=epoch_start)
        elif (number == 7): 
            if (resume):
                path = os.path.join("./checkpoints/inception/", str(number))
                epoch_start = len(os.listdir(path))-1
                
                path = os.path.join(path, str(epoch_start), "checkpoint")
                model = tf.keras.models.load_model(path)
            else:
                epoch_start = 0
                
                model = inception_models.InceptionV1(35, 16, [1], [[8, 12, 16, 2, 4, 4]]).model(input_shape=spectrogram_shape)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                              metrics=["accuracy"])
            
            model.fit(x=train_ds,
                      epochs=constants.EPOCHS,
                      verbose=2,
                      callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoints/inception/7/{epoch}/checkpoint"),
                                 tf.keras.callbacks.CSVLogger(filename="./checkpoints/inception/7/log.log", append=True)],
                      validation_data=test_ds,
                      initial_epoch=epoch_start)
        else: 
            print("ERROR: INVALID MODEL SELECTION!")
            return
        
    # CENet models
    elif (category == 5):
        if (number == 0): 
            if (resume):
                path = os.path.join("./checkpoints/cenet/", str(number))
                epoch_start = len(os.listdir(path))-1
                
                path = os.path.join(path, str(epoch_start), "checkpoint")
                model = tf.keras.models.load_model(path)
            else:
                epoch_start = 0
                
                model = cenet_models.CENet(35, 16, [15, 15, 7], [[0.5, 0.5, 32], [0.25, 0.25, 48], [0.25, 0.25, 64]]).model(input_shape=spectrogram_shape)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                              metrics=["accuracy"])
                
            model.fit(x=train_ds,
                      epochs=constants.EPOCHS,
                      verbose=2,
                      callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoints/cenet/0/{epoch}/checkpoint"),
                                 tf.keras.callbacks.CSVLogger(filename="./checkpoints/cenet/0/log.log", append=True)],
                      validation_data=test_ds,
                      initial_epoch=epoch_start)
        elif (number == 1): 
            if (resume):
                path = os.path.join("./checkpoints/cenet/", str(number))
                epoch_start = len(os.listdir(path))-1
                
                path = os.path.join(path, str(epoch_start), "checkpoint")
                model = tf.keras.models.load_model(path)
            else:
                epoch_start = 0
                
                model = cenet_models.CENet(35, 16, [7, 7, 7], [[0.5, 0.5, 32], [0.25, 0.25, 48], [0.25, 0.25, 64]]).model(input_shape=spectrogram_shape)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                              metrics=["accuracy"])
            
            model.fit(x=train_ds,
                      epochs=constants.EPOCHS,
                      verbose=2,
                      callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoints/cenet/1/{epoch}/checkpoint"),
                                 tf.keras.callbacks.CSVLogger(filename="./checkpoints/cenet/1/log.log", append=True)],
                      validation_data=test_ds,
                      initial_epoch=epoch_start)
        elif (number == 2): 
            if (resume):
                path = os.path.join("./checkpoints/cenet/", str(number))
                epoch_start = len(os.listdir(path))-1
                
                path = os.path.join(path, str(epoch_start), "checkpoint")
                model = tf.keras.models.load_model(path)
            else:
                epoch_start = 0
                
                model = cenet_models.CENet(35, 16, [7, 7, 6], [[0.5, 0.5, 32], [0.25, 0.25, 48], [0.25, 0.25, 64]]).model(input_shape=spectrogram_shape)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                              metrics=["accuracy"])
            
            model.fit(x=train_ds,
                      epochs=constants.EPOCHS,
                      verbose=2,
                      callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoints/cenet/2/{epoch}/checkpoint"),
                                 tf.keras.callbacks.CSVLogger(filename="./checkpoints/cenet/2/log.log", append=True)],
                      validation_data=test_ds,
                      initial_epoch=epoch_start)
        elif (number == 3): 
            if (resume):
                path = os.path.join("./checkpoints/cenet/", str(number))
                epoch_start = len(os.listdir(path))-1
                
                path = os.path.join(path, str(epoch_start), "checkpoint")
                model = tf.keras.models.load_model(path)
            else:
                epoch_start = 0
                
                model = cenet_models.CENet(35, 16, [7, 7, 5], [[0.5, 0.5, 32], [0.25, 0.25, 48], [0.25, 0.25, 64]]).model(input_shape=spectrogram_shape)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                              metrics=["accuracy"])
            
            model.fit(x=train_ds,
                      epochs=constants.EPOCHS,
                      verbose=2,
                      callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoints/cenet/3/{epoch}/checkpoint"),
                                 tf.keras.callbacks.CSVLogger(filename="./checkpoints/cenet/3/log.log", append=True)],
                      validation_data=test_ds,
                      initial_epoch=epoch_start)
        elif (number == 4): 
            if (resume):
                path = os.path.join("./checkpoints/cenet/", str(number))
                epoch_start = len(os.listdir(path))-1
                
                path = os.path.join(path, str(epoch_start), "checkpoint")
                model = tf.keras.models.load_model(path)
            else:
                epoch_start = 0
                
                model = cenet_models.CENet(35, 16, [7, 7, 4], [[0.5, 0.5, 32], [0.25, 0.25, 48], [0.25, 0.25, 64]]).model(input_shape=spectrogram_shape)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                              metrics=["accuracy"])
            
            model.fit(x=train_ds,
                      epochs=constants.EPOCHS,
                      verbose=2,
                      callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoints/cenet/4/{epoch}/checkpoint"),
                                 tf.keras.callbacks.CSVLogger(filename="./checkpoints/cenet/4/log.log", append=True)],
                      validation_data=test_ds,
                      initial_epoch=epoch_start)
        elif (number == 5): 
            if (resume):
                path = os.path.join("./checkpoints/cenet/", str(number))
                epoch_start = len(os.listdir(path))-1
                
                path = os.path.join(path, str(epoch_start), "checkpoint")
                model = tf.keras.models.load_model(path)
            else:
                epoch_start = 0
                
                model = cenet_models.CENet(35, 16, [7, 7, 3], [[0.5, 0.5, 32], [0.25, 0.25, 48], [0.25, 0.25, 64]]).model(input_shape=spectrogram_shape)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                              metrics=["accuracy"])
            
            model.fit(x=train_ds,
                      epochs=constants.EPOCHS,
                      verbose=2,
                      callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoints/cenet/5/{epoch}/checkpoint"),
                                 tf.keras.callbacks.CSVLogger(filename="./checkpoints/cenet/5/log.log", append=True)],
                      validation_data=test_ds,
                      initial_epoch=epoch_start)
        elif (number == 6): 
            if (resume):
                path = os.path.join("./checkpoints/cenet/", str(number))
                epoch_start = len(os.listdir(path))-1
                
                path = os.path.join(path, str(epoch_start), "checkpoint")
                model = tf.keras.models.load_model(path)
            else:
                epoch_start = 0
                
                model = cenet_models.CENet(35, 16, [7, 7, 2], [[0.5, 0.5, 32], [0.25, 0.25, 48], [0.25, 0.25, 64]]).model(input_shape=spectrogram_shape)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                              metrics=["accuracy"])
            
            model.fit(x=train_ds,
                      epochs=constants.EPOCHS,
                      verbose=2,
                      callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoints/cenet/6/{epoch}/checkpoint"),
                                 tf.keras.callbacks.CSVLogger(filename="./checkpoints/cenet/6/log.log", append=True)],
                      validation_data=test_ds,
                      initial_epoch=epoch_start)
        elif (number == 7):
            if (resume):
                path = os.path.join("./checkpoints/cenet/", str(number))
                epoch_start = len(os.listdir(path))-1
                
                path = os.path.join(path, str(epoch_start), "checkpoint")
                model = tf.keras.models.load_model(path)
            else:
                epoch_start = 0
                
                model = cenet_models.CENet(35, 16, [7, 7, 1], [[0.5, 0.5, 32], [0.25, 0.25, 48], [0.25, 0.25, 64]]).model(input_shape=spectrogram_shape)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                              metrics=["accuracy"])
            
            model.fit(x=train_ds,
                      epochs=constants.EPOCHS,
                      verbose=2,
                      callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoints/cenet/7/{epoch}/checkpoint"),
                                 tf.keras.callbacks.CSVLogger(filename="./checkpoints/cenet/7/log.log", append=True)],
                      validation_data=test_ds,
                      initial_epoch=epoch_start)
        elif (number == 8): 
            if (resume):
                path = os.path.join("./checkpoints/cenet/", str(number))
                epoch_start = len(os.listdir(path))-1
                
                path = os.path.join(path, str(epoch_start), "checkpoint")
                model = tf.keras.models.load_model(path)
            else:
                epoch_start = 0
                
                model = cenet_models.CENet(35, 16, [7, 6, 1], [[0.5, 0.5, 32], [0.25, 0.25, 48], [0.25, 0.25, 64]]).model(input_shape=spectrogram_shape)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                              metrics=["accuracy"])
            
            model.fit(x=train_ds,
                      epochs=constants.EPOCHS,
                      verbose=2,
                      callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoints/cenet/8/{epoch}/checkpoint"),
                                 tf.keras.callbacks.CSVLogger(filename="./checkpoints/cenet/8/log.log", append=True)],
                      validation_data=test_ds,
                      initial_epoch=epoch_start)
        elif (number == 9): 
            if (resume):
                path = os.path.join("./checkpoints/cenet/", str(number))
                epoch_start = len(os.listdir(path))-1
                
                path = os.path.join(path, str(epoch_start), "checkpoint")
                model = tf.keras.models.load_model(path)
            else:
                epoch_start = 0
                
                model = cenet_models.CENet(35, 16, [7, 5, 1], [[0.5, 0.5, 32], [0.25, 0.25, 48], [0.25, 0.25, 64]]).model(input_shape=spectrogram_shape)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                              metrics=["accuracy"])
            
            model.fit(x=train_ds,
                      epochs=constants.EPOCHS,
                      verbose=2,
                      callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoints/cenet/9/{epoch}/checkpoint"),
                                 tf.keras.callbacks.CSVLogger(filename="./checkpoints/cenet/9/log.log", append=True)],
                      validation_data=test_ds,
                      initial_epoch=epoch_start)
        elif (number == 10): 
            if (resume):
                path = os.path.join("./checkpoints/cenet/", str(number))
                epoch_start = len(os.listdir(path))-1
                
                path = os.path.join(path, str(epoch_start), "checkpoint")
                model = tf.keras.models.load_model(path)
            else:
                epoch_start = 0
                
                model = cenet_models.CENet(35, 16, [7, 4, 1], [[0.5, 0.5, 32], [0.25, 0.25, 48], [0.25, 0.25, 64]]).model(input_shape=spectrogram_shape)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                              metrics=["accuracy"])
            
            model.fit(x=train_ds,
                      epochs=constants.EPOCHS,
                      verbose=2,
                      callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoints/cenet/10/{epoch}/checkpoint"),
                                 tf.keras.callbacks.CSVLogger(filename="./checkpoints/cenet/10/log.log", append=True)],
                      validation_data=test_ds,
                      initial_epoch=epoch_start)
        elif (number == 11): 
            if (resume):
                path = os.path.join("./checkpoints/cenet/", str(number))
                epoch_start = len(os.listdir(path))-1
                
                path = os.path.join(path, str(epoch_start), "checkpoint")
                model = tf.keras.models.load_model(path)
            else:
                epoch_start = 0
                
                model = cenet_models.CENet(35, 16, [7, 3, 1], [[0.5, 0.5, 32], [0.25, 0.25, 48], [0.25, 0.25, 64]]).model(input_shape=spectrogram_shape)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                              metrics=["accuracy"])
                
            model.fit(x=train_ds,
                      epochs=constants.EPOCHS,
                      verbose=2,
                      callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoints/cenet/11/{epoch}/checkpoint"),
                                 tf.keras.callbacks.CSVLogger(filename="./checkpoints/cenet/11/log.log", append=True)],
                      validation_data=test_ds,
                      initial_epoch=epoch_start)
        elif (number == 12): 
            if (resume):
                path = os.path.join("./checkpoints/cenet/", str(number))
                epoch_start = len(os.listdir(path))-1
                
                path = os.path.join(path, str(epoch_start), "checkpoint")
                model = tf.keras.models.load_model(path)
            else:
                epoch_start = 0
                
                model = cenet_models.CENet(35, 16, [7, 2, 1], [[0.5, 0.5, 32], [0.25, 0.25, 48], [0.25, 0.25, 64]]).model(input_shape=spectrogram_shape)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                              metrics=["accuracy"])
            
            model.fit(x=train_ds,
                      epochs=constants.EPOCHS,
                      verbose=2,
                      callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoints/cenet/12/{epoch}/checkpoint"),
                                 tf.keras.callbacks.CSVLogger(filename="./checkpoints/cenet/12/log.log", append=True)],
                      validation_data=test_ds,
                      initial_epoch=epoch_start)
        elif (number == 13): 
            if (resume):
                path = os.path.join("./checkpoints/cenet/", str(number))
                epoch_start = len(os.listdir(path))-1
                
                path = os.path.join(path, str(epoch_start), "checkpoint")
                model = tf.keras.models.load_model(path)
            else:
                epoch_start = 0
                
                model = cenet_models.CENet(35, 16, [7, 1, 1], [[0.5, 0.5, 32], [0.25, 0.25, 48], [0.25, 0.25, 64]]).model(input_shape=spectrogram_shape)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                              metrics=["accuracy"])
            
            model.fit(x=train_ds,
                      epochs=constants.EPOCHS,
                      verbose=2,
                      callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoints/cenet/13/{epoch}/checkpoint"),
                                 tf.keras.callbacks.CSVLogger(filename="./checkpoints/cenet/13/log.log", append=True)],
                      validation_data=test_ds,
                      initial_epoch=epoch_start)
        elif (number == 14): 
            if (resume):
                path = os.path.join("./checkpoints/cenet/", str(number))
                epoch_start = len(os.listdir(path))-1
                
                path = os.path.join(path, str(epoch_start), "checkpoint")
                model = tf.keras.models.load_model(path)
            else:
                epoch_start = 0
                
                model = cenet_models.CENet(35, 16, [6, 1, 1], [[0.5, 0.5, 32], [0.25, 0.25, 48], [0.25, 0.25, 64]]).model(input_shape=spectrogram_shape)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                              metrics=["accuracy"])
            
            model.fit(x=train_ds,
                      epochs=constants.EPOCHS,
                      verbose=2,
                      callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoints/cenet/14/{epoch}/checkpoint"),
                                 tf.keras.callbacks.CSVLogger(filename="./checkpoints/cenet/14/log.log", append=True)],
                      validation_data=test_ds,
                      initial_epoch=epoch_start)
        elif (number == 15): 
            if (resume):
                path = os.path.join("./checkpoints/cenet/", str(number))
                epoch_start = len(os.listdir(path))-1
                
                path = os.path.join(path, str(epoch_start), "checkpoint")
                model = tf.keras.models.load_model(path)
            else:
                epoch_start = 0
                
                model = cenet_models.CENet(35, 16, [5, 1, 1], [[0.5, 0.5, 32], [0.25, 0.25, 48], [0.25, 0.25, 64]]).model(input_shape=spectrogram_shape)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                              metrics=["accuracy"])
            
            model.fit(x=train_ds,
                      epochs=constants.EPOCHS,
                      verbose=2,
                      callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoints/cenet/15/{epoch}/checkpoint"),
                                 tf.keras.callbacks.CSVLogger(filename="./checkpoints/cenet/15/log.log", append=True)],
                      validation_data=test_ds,
                      initial_epoch=epoch_start)
        elif (number == 16): 
            if (resume):
                path = os.path.join("./checkpoints/cenet/", str(number))
                epoch_start = len(os.listdir(path))-1
                
                path = os.path.join(path, str(epoch_start), "checkpoint")
                model = tf.keras.models.load_model(path)
            else:
                epoch_start = 0
                
                model = cenet_models.CENet(35, 16, [4, 1, 1], [[0.5, 0.5, 32], [0.25, 0.25, 48], [0.25, 0.25, 64]]).model(input_shape=spectrogram_shape)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                              metrics=["accuracy"])
            
            model.fit(x=train_ds,
                      epochs=constants.EPOCHS,
                      verbose=2,
                      callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoints/cenet/16/{epoch}/checkpoint"),
                                 tf.keras.callbacks.CSVLogger(filename="./checkpoints/cenet/16/log.log", append=True)],
                      validation_data=test_ds,
                      initial_epoch=epoch_start)
        elif (number == 17): 
            if (resume):
                path = os.path.join("./checkpoints/cenet/", str(number))
                epoch_start = len(os.listdir(path))-1
                
                path = os.path.join(path, str(epoch_start), "checkpoint")
                model = tf.keras.models.load_model(path)
            else:
                epoch_start = 0
                
                model = cenet_models.CENet(35, 16, [3, 1, 1], [[0.5, 0.5, 32], [0.25, 0.25, 48], [0.25, 0.25, 64]]).model(input_shape=spectrogram_shape)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                              metrics=["accuracy"])
            
            model.fit(x=train_ds,
                      epochs=constants.EPOCHS,
                      verbose=2,
                      callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoints/cenet/17/{epoch}/checkpoint"),
                                 tf.keras.callbacks.CSVLogger(filename="./checkpoints/cenet/17/log.log", append=True)],
                      validation_data=test_ds,
                      initial_epoch=epoch_start)
        elif (number == 18): 
            if (resume):
                path = os.path.join("./checkpoints/cenet/", str(number))
                epoch_start = len(os.listdir(path))-1
                
                path = os.path.join(path, str(epoch_start), "checkpoint")
                model = tf.keras.models.load_model(path)
            else:
                epoch_start = 0
                
                model = cenet_models.CENet(35, 16, [2, 1, 1], [[0.5, 0.5, 32], [0.25, 0.25, 48], [0.25, 0.25, 64]]).model(input_shape=spectrogram_shape)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                              metrics=["accuracy"])
            
            model.fit(x=train_ds,
                      epochs=constants.EPOCHS,
                      verbose=2,
                      callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoints/cenet/18/{epoch}/checkpoint"),
                                 tf.keras.callbacks.CSVLogger(filename="./checkpoints/cenet/18/log.log", append=True)],
                      validation_data=test_ds,
                      initial_epoch=epoch_start)
        elif (number == 19):
            if (resume):
                path = os.path.join("./checkpoints/cenet/", str(number))
                epoch_start = len(os.listdir(path))-1
                
                path = os.path.join(path, str(epoch_start), "checkpoint")
                model = tf.keras.models.load_model(path)
            else:
                epoch_start = 0
                
                model = cenet_models.CENet(35, 16, [1, 1, 1], [[0.5, 0.5, 32], [0.25, 0.25, 48], [0.25, 0.25, 64]]).model(input_shape=spectrogram_shape)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                              metrics=["accuracy"])
            
            model.fit(x=train_ds,
                      epochs=constants.EPOCHS,
                      verbose=2,
                      callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoints/cenet/19/{epoch}/checkpoint"),
                                 tf.keras.callbacks.CSVLogger(filename="./checkpoints/cenet/19/log.log", append=True)],
                      validation_data=test_ds,
                      initial_epoch=epoch_start)
        elif (number == 20): 
            if (resume):
                path = os.path.join("./checkpoints/cenet/", str(number))
                epoch_start = len(os.listdir(path))-1
                
                path = os.path.join(path, str(epoch_start), "checkpoint")
                model = tf.keras.models.load_model(path)
            else:
                epoch_start = 0
                
                model = cenet_models.CENet(35, 16, [1, 1], [[0.5, 0.5, 32], [0.25, 0.25, 48]]).model(input_shape=spectrogram_shape)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                              metrics=["accuracy"])
            
            model.fit(x=train_ds,
                      epochs=constants.EPOCHS,
                      verbose=2,
                      callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoints/cenet/20/{epoch}/checkpoint"),
                                 tf.keras.callbacks.CSVLogger(filename="./checkpoints/cenet/20/log.log", append=True)],
                      validation_data=test_ds,
                      initial_epoch=epoch_start)
        elif (number == 21):
            if (resume):
                path = os.path.join("./checkpoints/cenet/", str(number))
                epoch_start = len(os.listdir(path))-1
                
                path = os.path.join(path, str(epoch_start), "checkpoint")
                model = tf.keras.models.load_model(path)
            else:
                epoch_start = 0
                
                model = cenet_models.CENet(35, 16, [1], [[0.5, 0.5, 32]]).model(input_shape=spectrogram_shape)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                              metrics=["accuracy"])
            
            model.fit(x=train_ds,
                      epochs=constants.EPOCHS,
                      verbose=2,
                      callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoints/cenet/21/{epoch}/checkpoint"),
                                 tf.keras.callbacks.CSVLogger(filename="./checkpoints/cenet/21/log.log", append=True)],
                      validation_data=test_ds,
                      initial_epoch=epoch_start)
        else: 
            print("ERROR: INVALID MODEL SELECTION!")
            return
    else: 
        print("ERROR: INVALID MODEL CATEGORY!")
        return
    
    return

def main(argv):
    # Variables to hold CLI arguments
    architecture = None
    config = None
    resume = False
    
    # Parse CLI input
    opts, args = getopt.getopt(argv, "ha:c:r", ["help", "arch=", "config=", "resume"])
    for opt, arg in opts:
        if opt in ("-h", "--help"): 
            print ("main.py -a <architecture> -c <configuration> -r")
            sys.exit()
        elif opt in ("-a", "--arch"):
            architecture = int(arg)
        elif opt in ("-c", "--config"):
            config = int(arg)
        elif opt in ("-r", "--resume"):
            resume = True
    
    # Train selected architecture and configuration
    train_model(category=architecture, number=config, resume=resume)

if __name__ == "__main__":
   main(sys.argv[1:])
