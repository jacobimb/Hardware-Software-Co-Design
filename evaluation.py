# -*- coding: utf-8 -*-
# Evaluation Script
# Script iterates through Tensorflow saved models and collects accuracy,
# parameter, latency, and size information
import tensorflow as tf
import constants
import dataset
import dscnn_models
import densenet_models
import resnet_models
import ffcn_models
import inception_models
import cenet_models
import os
import sys
import pandas as pd
import numpy as np
import subprocess
import getopt

# Hardcode specific epochs to evaluate to save time on data collection
epochs_to_test = ['1', '5', '10', '15', '20', '25', '30', '35', '40', '45', '50', 
                  '55', '60', '65', '70', '75', '80', '85', '90', '95', '100']

# RepresentativeDataset class for TensorFlow quantization
class RepresentativeDataset():
    def __init__(self, dataset, samples):
        # Take input dataset, change batch size to one, and take random samples
        self.dataset = dataset.unbatch().batch(1).take(samples)
        
    def __call__(self):
        # Generator function
        for data, _ in self.dataset:
            yield [tf.dtypes.cast(data, tf.float32)]
            
# Function for finding the number parameters in the model
# param model_path: path to TensorFlow model (NOT Tensorflow Lite)
def get_params(model_path, output_file):
    # Load TensorFlow model
    model = tf.keras.models.load_model(model_path)
    
    # Save model.summary()
    with open(output_file, "a") as f: 
        model.summary(line_length=500, print_fn=f.write)
        
    # Read model.summary()
    with open(output_file, "r") as f: 
        summary_string = f.read()
        
    # Extract parameter count from model.summary()
    substrings_1 = summary_string.split("Total params: ")
    substrings_2 = substrings_1[1].split("Trainable params: ")
    substrings_3 = substrings_2[1].split("Non-trainable params: ")
    substrings_4 = substrings_3[1].split("_")
    
    # Convert paramter counts into integers
    total = int(substrings_2[0].replace(",", ""))
    trainable = int(substrings_3[0].replace(",", ""))
    nontrainable = int(substrings_4[0].replace(",", ""))
    
    return [total, trainable, nontrainable]

# Function for running the TensorFlow Lite model with dataset
# param model_path:     path to TFLite model
# param quantization:   if the model requires quantization
def run_model(model_path, quantization=False):
    # Create original datasets used during training stage
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
    
    # Take validation dataset, adjust batch size, and convert to numpy array
    actual_test_dataset = test_ds.unbatch().batch(1).as_numpy_iterator()
    
    # Load the TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Get input and output tensor information
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Arrays for saving true vs predicted values
    actual = []
    predicted = []
    
    # Loop through validation dataset
    for data, label in actual_test_dataset:
        # Save correct label
        actual.append(label[0])
    
        # Apply quantization to input if necessary
        if (quantization):
            data = (data/input_details[0]['quantization'][0]) + input_details[0]['quantization'][1]
            data = np.array(data, dtype=np.int8)
        
        # Give input data to model
        interpreter.set_tensor(input_details[0]['index'], data)
        
        # Run the model
        interpreter.invoke()
        
        # Get result tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Reverse output quantization if necessary
        if (quantization):
            output_data = (output_data/output_details[0]['quantization'][0]) + output_details[0]['quantization'][1]

        # Save predicted label
        predicted.append(output_data.argmax())
    
    return [actual, predicted]

# Function for calculating model accuracy and creating cross confusion matrices
# param model_path:     path to TFLite model
# param quantization:   if the model requires quantization
# 
# Note on the arrangement of the cross confuion matrix:
# tf.math.confusion_matrix([1, 2, 4], [2, 2, 4]) ==>
#        Predicted Label
# True  [[0 0 0 0 0]
# Label  [0 0 1 0 0]
#        [0 0 1 0 0]
#        [0 0 0 0 0]
#        [0 0 0 0 1]]
def get_accuracy(model_path, quantization=False):
    # Run the model on the validation dataset
    [actual, predicted] = run_model(model_path, quantization)
    
    # Create cross confusion matrix
    cross_confusion_matrix = tf.math.confusion_matrix(actual, predicted).numpy()
    
    # Calculate accuracy by computing correct_labels/total_predictions
    accuracy = cross_confusion_matrix.trace()/cross_confusion_matrix.sum()
    
    return [accuracy, cross_confusion_matrix]

# Function for running TensorFlow Lite benchmark tool
# param model_path:     path to TFLite model
# param output_file:    file to use to save benchmark results
def run_benchmark(model_path, output_file):
    # Prepare benchmark args
    model_arg = "--graph=\""+model_path+"\""
    benchmark_args = ["./linux_x86-64_benchmark_model", 
                      model_arg, 
                      "--num_threads=1", 
                      "--enable_op_profiling=true",
                      "--report_peak_memory_footprint=true",
                      "--print_preinvoke_state=true",
                      "--print_postinvoke_state=true"]
    cmd = " ".join(benchmark_args)
    
    # Run the benchmark and redirect stdout results to file
    with open(output_file, "a") as f:  
        subprocess.run(cmd, shell=True, stdout=f)
    
    return 

# Function for collecting model latency and memory statistics
# param model_path:     path to TFLite model
# param output_file:    file to use to save benchmark results
def get_runtime_stats(model_path, output_file):
    # Run the benchmark
    run_benchmark(model_path, output_file)

    # Open benchmark results
    with open(output_file, "r") as f: 
        report_string= f.read()
        
    # Extract latency and memory statistics
    substrings_1 = report_string.split("Inference timings in us: Init: ")
    substrings_2 = substrings_1[1].split(", First inference: ")
    substrings_3 = substrings_2[1].split(", Warmup (avg): ")
    substrings_4 = substrings_3[1].split(", Inference (avg): ")
    substrings_5 = substrings_4[1].split("Note: ")
    substrings_6 = substrings_5[1].split("Memory footprint delta from the start of the tool (MB): init=")
    substrings_7 = substrings_6[1].split(" overall=")
    substrings_8 = substrings_7[1].split("Overall peak memory footprint (MB) via periodic monitoring: ")
    substrings_9 = substrings_8[1].split("Memory status at the end of exeution:")
    
    time_init = float(substrings_2[0])
    time_first_inference = float(substrings_3[0])
    time_warmup_avg = float(substrings_4[0])
    time_inference_avg = float(substrings_5[0])
    
    memory_init = float(substrings_7[0])
    memory_overall = float(substrings_8[0])
    memory_peak = float(substrings_9[0])
    
    latency_stats = [time_init, time_first_inference, time_warmup_avg, time_inference_avg]
    memory_stats = [memory_init, memory_overall, memory_peak]
    
    return [latency_stats, memory_stats]

# Function for saving all collected results
# param model_arch:             list of model architectures
# param model_config:           list of architecture configurations
# param model_quantization:     list of model quantization methods
# param model_size:             list of model file size in bytes
# param model_params:           list of lists of parameter counts
# param model_latency:          list of lists of latency statistics
# param model_memory:           list of lists of memory statistics
# param model_ccm:              list of lists of constants.EPOCHS 
#                               constants.NUM_CLASSES by NUM_CLASSES numpy 
#                               cross confusion matrices
# param model_accuracy:         list of lists of constants.EPOCHS accuracy 
#                               measurements
# Each parameter should be equal to TOTAL_NUMBER_NETWORK_CONFIGS*5
def create_dataframe(model_arch,
                     model_config,
                     model_quantization,
                     model_size,
                     model_params,
                     model_latency,
                     model_memory,
                     model_ccm,
                     model_accuracy):    
    
    # Array for saving results
    data = []
    
    # For each architecture/config/quantization combination
    for i in range(len(model_arch)):
        # Combine data for model into single list of values
        data_row = []
        
        data_row.append(model_arch[i])
        data_row.append(model_config[i])
        data_row.append(model_quantization[i])
        data_row.append(model_size[i])
        data_row.extend(model_params[i])
        data_row.extend(model_memory[i])
        data_row.extend(model_latency[i])
        data_row.extend(model_accuracy[i])
        
        data.append(data_row)
        
        # Array for model's cross confusion matrix data
        ccm_data = []
        
        # For each epoch trained
        for epoch in list(model_ccm[i]):
            # Flatten cross confusion matrix and append to larger data array
            ccm_row = []
            
            for array in list(epoch):
                ccm_row.extend(list(array))
                
            ccm_data.append(ccm_row)
        
        # Create pandas dataframe from data array 
        ccm_dataframe = pd.DataFrame(columns=["Label: backward Predicted: backward",
                                            "Label: backward Predicted: bed",
                                            "Label: backward Predicted: bird",
                                            "Label: backward Predicted: cat",
                                            "Label: backward Predicted: dog",
                                            "Label: backward Predicted: down",
                                            "Label: backward Predicted: eight",
                                            "Label: backward Predicted: five",
                                            "Label: backward Predicted: follow",
                                            "Label: backward Predicted: forward",
                                            "Label: backward Predicted: four",
                                            "Label: backward Predicted: go",
                                            "Label: backward Predicted: happy",
                                            "Label: backward Predicted: house",
                                            "Label: backward Predicted: learn",
                                            "Label: backward Predicted: left",
                                            "Label: backward Predicted: marvin",
                                            "Label: backward Predicted: nine",
                                            "Label: backward Predicted: no",
                                            "Label: backward Predicted: off",
                                            "Label: backward Predicted: on",
                                            "Label: backward Predicted: one",
                                            "Label: backward Predicted: right",
                                            "Label: backward Predicted: seven",
                                            "Label: backward Predicted: sheila",
                                            "Label: backward Predicted: six",
                                            "Label: backward Predicted: stop",
                                            "Label: backward Predicted: three",
                                            "Label: backward Predicted: tree",
                                            "Label: backward Predicted: two",
                                            "Label: backward Predicted: up",
                                            "Label: backward Predicted: visual",
                                            "Label: backward Predicted: wow",
                                            "Label: backward Predicted: yes",
                                            "Label: backward Predicted: zero",
                                            "Label: bed Predicted: backward",
                                            "Label: bed Predicted: bed",
                                            "Label: bed Predicted: bird",
                                            "Label: bed Predicted: cat",
                                            "Label: bed Predicted: dog",
                                            "Label: bed Predicted: down",
                                            "Label: bed Predicted: eight",
                                            "Label: bed Predicted: five",
                                            "Label: bed Predicted: follow",
                                            "Label: bed Predicted: forward",
                                            "Label: bed Predicted: four",
                                            "Label: bed Predicted: go",
                                            "Label: bed Predicted: happy",
                                            "Label: bed Predicted: house",
                                            "Label: bed Predicted: learn",
                                            "Label: bed Predicted: left",
                                            "Label: bed Predicted: marvin",
                                            "Label: bed Predicted: nine",
                                            "Label: bed Predicted: no",
                                            "Label: bed Predicted: off",
                                            "Label: bed Predicted: on",
                                            "Label: bed Predicted: one",
                                            "Label: bed Predicted: right",
                                            "Label: bed Predicted: seven",
                                            "Label: bed Predicted: sheila",
                                            "Label: bed Predicted: six",
                                            "Label: bed Predicted: stop",
                                            "Label: bed Predicted: three",
                                            "Label: bed Predicted: tree",
                                            "Label: bed Predicted: two",
                                            "Label: bed Predicted: up",
                                            "Label: bed Predicted: visual",
                                            "Label: bed Predicted: wow",
                                            "Label: bed Predicted: yes",
                                            "Label: bed Predicted: zero",
                                            "Label: bird Predicted: backward",
                                            "Label: bird Predicted: bed",
                                            "Label: bird Predicted: bird",
                                            "Label: bird Predicted: cat",
                                            "Label: bird Predicted: dog",
                                            "Label: bird Predicted: down",
                                            "Label: bird Predicted: eight",
                                            "Label: bird Predicted: five",
                                            "Label: bird Predicted: follow",
                                            "Label: bird Predicted: forward",
                                            "Label: bird Predicted: four",
                                            "Label: bird Predicted: go",
                                            "Label: bird Predicted: happy",
                                            "Label: bird Predicted: house",
                                            "Label: bird Predicted: learn",
                                            "Label: bird Predicted: left",
                                            "Label: bird Predicted: marvin",
                                            "Label: bird Predicted: nine",
                                            "Label: bird Predicted: no",
                                            "Label: bird Predicted: off",
                                            "Label: bird Predicted: on",
                                            "Label: bird Predicted: one",
                                            "Label: bird Predicted: right",
                                            "Label: bird Predicted: seven",
                                            "Label: bird Predicted: sheila",
                                            "Label: bird Predicted: six",
                                            "Label: bird Predicted: stop",
                                            "Label: bird Predicted: three",
                                            "Label: bird Predicted: tree",
                                            "Label: bird Predicted: two",
                                            "Label: bird Predicted: up",
                                            "Label: bird Predicted: visual",
                                            "Label: bird Predicted: wow",
                                            "Label: bird Predicted: yes",
                                            "Label: bird Predicted: zero",
                                            "Label: cat Predicted: backward",
                                            "Label: cat Predicted: bed",
                                            "Label: cat Predicted: bird",
                                            "Label: cat Predicted: cat",
                                            "Label: cat Predicted: dog",
                                            "Label: cat Predicted: down",
                                            "Label: cat Predicted: eight",
                                            "Label: cat Predicted: five",
                                            "Label: cat Predicted: follow",
                                            "Label: cat Predicted: forward",
                                            "Label: cat Predicted: four",
                                            "Label: cat Predicted: go",
                                            "Label: cat Predicted: happy",
                                            "Label: cat Predicted: house",
                                            "Label: cat Predicted: learn",
                                            "Label: cat Predicted: left",
                                            "Label: cat Predicted: marvin",
                                            "Label: cat Predicted: nine",
                                            "Label: cat Predicted: no",
                                            "Label: cat Predicted: off",
                                            "Label: cat Predicted: on",
                                            "Label: cat Predicted: one",
                                            "Label: cat Predicted: right",
                                            "Label: cat Predicted: seven",
                                            "Label: cat Predicted: sheila",
                                            "Label: cat Predicted: six",
                                            "Label: cat Predicted: stop",
                                            "Label: cat Predicted: three",
                                            "Label: cat Predicted: tree",
                                            "Label: cat Predicted: two",
                                            "Label: cat Predicted: up",
                                            "Label: cat Predicted: visual",
                                            "Label: cat Predicted: wow",
                                            "Label: cat Predicted: yes",
                                            "Label: cat Predicted: zero",
                                            "Label: dog Predicted: backward",
                                            "Label: dog Predicted: bed",
                                            "Label: dog Predicted: bird",
                                            "Label: dog Predicted: cat",
                                            "Label: dog Predicted: dog",
                                            "Label: dog Predicted: down",
                                            "Label: dog Predicted: eight",
                                            "Label: dog Predicted: five",
                                            "Label: dog Predicted: follow",
                                            "Label: dog Predicted: forward",
                                            "Label: dog Predicted: four",
                                            "Label: dog Predicted: go",
                                            "Label: dog Predicted: happy",
                                            "Label: dog Predicted: house",
                                            "Label: dog Predicted: learn",
                                            "Label: dog Predicted: left",
                                            "Label: dog Predicted: marvin",
                                            "Label: dog Predicted: nine",
                                            "Label: dog Predicted: no",
                                            "Label: dog Predicted: off",
                                            "Label: dog Predicted: on",
                                            "Label: dog Predicted: one",
                                            "Label: dog Predicted: right",
                                            "Label: dog Predicted: seven",
                                            "Label: dog Predicted: sheila",
                                            "Label: dog Predicted: six",
                                            "Label: dog Predicted: stop",
                                            "Label: dog Predicted: three",
                                            "Label: dog Predicted: tree",
                                            "Label: dog Predicted: two",
                                            "Label: dog Predicted: up",
                                            "Label: dog Predicted: visual",
                                            "Label: dog Predicted: wow",
                                            "Label: dog Predicted: yes",
                                            "Label: dog Predicted: zero",
                                            "Label: down Predicted: backward",
                                            "Label: down Predicted: bed",
                                            "Label: down Predicted: bird",
                                            "Label: down Predicted: cat",
                                            "Label: down Predicted: dog",
                                            "Label: down Predicted: down",
                                            "Label: down Predicted: eight",
                                            "Label: down Predicted: five",
                                            "Label: down Predicted: follow",
                                            "Label: down Predicted: forward",
                                            "Label: down Predicted: four",
                                            "Label: down Predicted: go",
                                            "Label: down Predicted: happy",
                                            "Label: down Predicted: house",
                                            "Label: down Predicted: learn",
                                            "Label: down Predicted: left",
                                            "Label: down Predicted: marvin",
                                            "Label: down Predicted: nine",
                                            "Label: down Predicted: no",
                                            "Label: down Predicted: off",
                                            "Label: down Predicted: on",
                                            "Label: down Predicted: one",
                                            "Label: down Predicted: right",
                                            "Label: down Predicted: seven",
                                            "Label: down Predicted: sheila",
                                            "Label: down Predicted: six",
                                            "Label: down Predicted: stop",
                                            "Label: down Predicted: three",
                                            "Label: down Predicted: tree",
                                            "Label: down Predicted: two",
                                            "Label: down Predicted: up",
                                            "Label: down Predicted: visual",
                                            "Label: down Predicted: wow",
                                            "Label: down Predicted: yes",
                                            "Label: down Predicted: zero",
                                            "Label: eight Predicted: backward",
                                            "Label: eight Predicted: bed",
                                            "Label: eight Predicted: bird",
                                            "Label: eight Predicted: cat",
                                            "Label: eight Predicted: dog",
                                            "Label: eight Predicted: down",
                                            "Label: eight Predicted: eight",
                                            "Label: eight Predicted: five",
                                            "Label: eight Predicted: follow",
                                            "Label: eight Predicted: forward",
                                            "Label: eight Predicted: four",
                                            "Label: eight Predicted: go",
                                            "Label: eight Predicted: happy",
                                            "Label: eight Predicted: house",
                                            "Label: eight Predicted: learn",
                                            "Label: eight Predicted: left",
                                            "Label: eight Predicted: marvin",
                                            "Label: eight Predicted: nine",
                                            "Label: eight Predicted: no",
                                            "Label: eight Predicted: off",
                                            "Label: eight Predicted: on",
                                            "Label: eight Predicted: one",
                                            "Label: eight Predicted: right",
                                            "Label: eight Predicted: seven",
                                            "Label: eight Predicted: sheila",
                                            "Label: eight Predicted: six",
                                            "Label: eight Predicted: stop",
                                            "Label: eight Predicted: three",
                                            "Label: eight Predicted: tree",
                                            "Label: eight Predicted: two",
                                            "Label: eight Predicted: up",
                                            "Label: eight Predicted: visual",
                                            "Label: eight Predicted: wow",
                                            "Label: eight Predicted: yes",
                                            "Label: eight Predicted: zero",
                                            "Label: five Predicted: backward",
                                            "Label: five Predicted: bed",
                                            "Label: five Predicted: bird",
                                            "Label: five Predicted: cat",
                                            "Label: five Predicted: dog",
                                            "Label: five Predicted: down",
                                            "Label: five Predicted: eight",
                                            "Label: five Predicted: five",
                                            "Label: five Predicted: follow",
                                            "Label: five Predicted: forward",
                                            "Label: five Predicted: four",
                                            "Label: five Predicted: go",
                                            "Label: five Predicted: happy",
                                            "Label: five Predicted: house",
                                            "Label: five Predicted: learn",
                                            "Label: five Predicted: left",
                                            "Label: five Predicted: marvin",
                                            "Label: five Predicted: nine",
                                            "Label: five Predicted: no",
                                            "Label: five Predicted: off",
                                            "Label: five Predicted: on",
                                            "Label: five Predicted: one",
                                            "Label: five Predicted: right",
                                            "Label: five Predicted: seven",
                                            "Label: five Predicted: sheila",
                                            "Label: five Predicted: six",
                                            "Label: five Predicted: stop",
                                            "Label: five Predicted: three",
                                            "Label: five Predicted: tree",
                                            "Label: five Predicted: two",
                                            "Label: five Predicted: up",
                                            "Label: five Predicted: visual",
                                            "Label: five Predicted: wow",
                                            "Label: five Predicted: yes",
                                            "Label: five Predicted: zero",
                                            "Label: follow Predicted: backward",
                                            "Label: follow Predicted: bed",
                                            "Label: follow Predicted: bird",
                                            "Label: follow Predicted: cat",
                                            "Label: follow Predicted: dog",
                                            "Label: follow Predicted: down",
                                            "Label: follow Predicted: eight",
                                            "Label: follow Predicted: five",
                                            "Label: follow Predicted: follow",
                                            "Label: follow Predicted: forward",
                                            "Label: follow Predicted: four",
                                            "Label: follow Predicted: go",
                                            "Label: follow Predicted: happy",
                                            "Label: follow Predicted: house",
                                            "Label: follow Predicted: learn",
                                            "Label: follow Predicted: left",
                                            "Label: follow Predicted: marvin",
                                            "Label: follow Predicted: nine",
                                            "Label: follow Predicted: no",
                                            "Label: follow Predicted: off",
                                            "Label: follow Predicted: on",
                                            "Label: follow Predicted: one",
                                            "Label: follow Predicted: right",
                                            "Label: follow Predicted: seven",
                                            "Label: follow Predicted: sheila",
                                            "Label: follow Predicted: six",
                                            "Label: follow Predicted: stop",
                                            "Label: follow Predicted: three",
                                            "Label: follow Predicted: tree",
                                            "Label: follow Predicted: two",
                                            "Label: follow Predicted: up",
                                            "Label: follow Predicted: visual",
                                            "Label: follow Predicted: wow",
                                            "Label: follow Predicted: yes",
                                            "Label: follow Predicted: zero",
                                            "Label: forward Predicted: backward",
                                            "Label: forward Predicted: bed",
                                            "Label: forward Predicted: bird",
                                            "Label: forward Predicted: cat",
                                            "Label: forward Predicted: dog",
                                            "Label: forward Predicted: down",
                                            "Label: forward Predicted: eight",
                                            "Label: forward Predicted: five",
                                            "Label: forward Predicted: follow",
                                            "Label: forward Predicted: forward",
                                            "Label: forward Predicted: four",
                                            "Label: forward Predicted: go",
                                            "Label: forward Predicted: happy",
                                            "Label: forward Predicted: house",
                                            "Label: forward Predicted: learn",
                                            "Label: forward Predicted: left",
                                            "Label: forward Predicted: marvin",
                                            "Label: forward Predicted: nine",
                                            "Label: forward Predicted: no",
                                            "Label: forward Predicted: off",
                                            "Label: forward Predicted: on",
                                            "Label: forward Predicted: one",
                                            "Label: forward Predicted: right",
                                            "Label: forward Predicted: seven",
                                            "Label: forward Predicted: sheila",
                                            "Label: forward Predicted: six",
                                            "Label: forward Predicted: stop",
                                            "Label: forward Predicted: three",
                                            "Label: forward Predicted: tree",
                                            "Label: forward Predicted: two",
                                            "Label: forward Predicted: up",
                                            "Label: forward Predicted: visual",
                                            "Label: forward Predicted: wow",
                                            "Label: forward Predicted: yes",
                                            "Label: forward Predicted: zero",
                                            "Label: four Predicted: backward",
                                            "Label: four Predicted: bed",
                                            "Label: four Predicted: bird",
                                            "Label: four Predicted: cat",
                                            "Label: four Predicted: dog",
                                            "Label: four Predicted: down",
                                            "Label: four Predicted: eight",
                                            "Label: four Predicted: five",
                                            "Label: four Predicted: follow",
                                            "Label: four Predicted: forward",
                                            "Label: four Predicted: four",
                                            "Label: four Predicted: go",
                                            "Label: four Predicted: happy",
                                            "Label: four Predicted: house",
                                            "Label: four Predicted: learn",
                                            "Label: four Predicted: left",
                                            "Label: four Predicted: marvin",
                                            "Label: four Predicted: nine",
                                            "Label: four Predicted: no",
                                            "Label: four Predicted: off",
                                            "Label: four Predicted: on",
                                            "Label: four Predicted: one",
                                            "Label: four Predicted: right",
                                            "Label: four Predicted: seven",
                                            "Label: four Predicted: sheila",
                                            "Label: four Predicted: six",
                                            "Label: four Predicted: stop",
                                            "Label: four Predicted: three",
                                            "Label: four Predicted: tree",
                                            "Label: four Predicted: two",
                                            "Label: four Predicted: up",
                                            "Label: four Predicted: visual",
                                            "Label: four Predicted: wow",
                                            "Label: four Predicted: yes",
                                            "Label: four Predicted: zero",
                                            "Label: go Predicted: backward",
                                            "Label: go Predicted: bed",
                                            "Label: go Predicted: bird",
                                            "Label: go Predicted: cat",
                                            "Label: go Predicted: dog",
                                            "Label: go Predicted: down",
                                            "Label: go Predicted: eight",
                                            "Label: go Predicted: five",
                                            "Label: go Predicted: follow",
                                            "Label: go Predicted: forward",
                                            "Label: go Predicted: four",
                                            "Label: go Predicted: go",
                                            "Label: go Predicted: happy",
                                            "Label: go Predicted: house",
                                            "Label: go Predicted: learn",
                                            "Label: go Predicted: left",
                                            "Label: go Predicted: marvin",
                                            "Label: go Predicted: nine",
                                            "Label: go Predicted: no",
                                            "Label: go Predicted: off",
                                            "Label: go Predicted: on",
                                            "Label: go Predicted: one",
                                            "Label: go Predicted: right",
                                            "Label: go Predicted: seven",
                                            "Label: go Predicted: sheila",
                                            "Label: go Predicted: six",
                                            "Label: go Predicted: stop",
                                            "Label: go Predicted: three",
                                            "Label: go Predicted: tree",
                                            "Label: go Predicted: two",
                                            "Label: go Predicted: up",
                                            "Label: go Predicted: visual",
                                            "Label: go Predicted: wow",
                                            "Label: go Predicted: yes",
                                            "Label: go Predicted: zero",
                                            "Label: happy Predicted: backward",
                                            "Label: happy Predicted: bed",
                                            "Label: happy Predicted: bird",
                                            "Label: happy Predicted: cat",
                                            "Label: happy Predicted: dog",
                                            "Label: happy Predicted: down",
                                            "Label: happy Predicted: eight",
                                            "Label: happy Predicted: five",
                                            "Label: happy Predicted: follow",
                                            "Label: happy Predicted: forward",
                                            "Label: happy Predicted: four",
                                            "Label: happy Predicted: go",
                                            "Label: happy Predicted: happy",
                                            "Label: happy Predicted: house",
                                            "Label: happy Predicted: learn",
                                            "Label: happy Predicted: left",
                                            "Label: happy Predicted: marvin",
                                            "Label: happy Predicted: nine",
                                            "Label: happy Predicted: no",
                                            "Label: happy Predicted: off",
                                            "Label: happy Predicted: on",
                                            "Label: happy Predicted: one",
                                            "Label: happy Predicted: right",
                                            "Label: happy Predicted: seven",
                                            "Label: happy Predicted: sheila",
                                            "Label: happy Predicted: six",
                                            "Label: happy Predicted: stop",
                                            "Label: happy Predicted: three",
                                            "Label: happy Predicted: tree",
                                            "Label: happy Predicted: two",
                                            "Label: happy Predicted: up",
                                            "Label: happy Predicted: visual",
                                            "Label: happy Predicted: wow",
                                            "Label: happy Predicted: yes",
                                            "Label: happy Predicted: zero",
                                            "Label: house Predicted: backward",
                                            "Label: house Predicted: bed",
                                            "Label: house Predicted: bird",
                                            "Label: house Predicted: cat",
                                            "Label: house Predicted: dog",
                                            "Label: house Predicted: down",
                                            "Label: house Predicted: eight",
                                            "Label: house Predicted: five",
                                            "Label: house Predicted: follow",
                                            "Label: house Predicted: forward",
                                            "Label: house Predicted: four",
                                            "Label: house Predicted: go",
                                            "Label: house Predicted: happy",
                                            "Label: house Predicted: house",
                                            "Label: house Predicted: learn",
                                            "Label: house Predicted: left",
                                            "Label: house Predicted: marvin",
                                            "Label: house Predicted: nine",
                                            "Label: house Predicted: no",
                                            "Label: house Predicted: off",
                                            "Label: house Predicted: on",
                                            "Label: house Predicted: one",
                                            "Label: house Predicted: right",
                                            "Label: house Predicted: seven",
                                            "Label: house Predicted: sheila",
                                            "Label: house Predicted: six",
                                            "Label: house Predicted: stop",
                                            "Label: house Predicted: three",
                                            "Label: house Predicted: tree",
                                            "Label: house Predicted: two",
                                            "Label: house Predicted: up",
                                            "Label: house Predicted: visual",
                                            "Label: house Predicted: wow",
                                            "Label: house Predicted: yes",
                                            "Label: house Predicted: zero",
                                            "Label: learn Predicted: backward",
                                            "Label: learn Predicted: bed",
                                            "Label: learn Predicted: bird",
                                            "Label: learn Predicted: cat",
                                            "Label: learn Predicted: dog",
                                            "Label: learn Predicted: down",
                                            "Label: learn Predicted: eight",
                                            "Label: learn Predicted: five",
                                            "Label: learn Predicted: follow",
                                            "Label: learn Predicted: forward",
                                            "Label: learn Predicted: four",
                                            "Label: learn Predicted: go",
                                            "Label: learn Predicted: happy",
                                            "Label: learn Predicted: house",
                                            "Label: learn Predicted: learn",
                                            "Label: learn Predicted: left",
                                            "Label: learn Predicted: marvin",
                                            "Label: learn Predicted: nine",
                                            "Label: learn Predicted: no",
                                            "Label: learn Predicted: off",
                                            "Label: learn Predicted: on",
                                            "Label: learn Predicted: one",
                                            "Label: learn Predicted: right",
                                            "Label: learn Predicted: seven",
                                            "Label: learn Predicted: sheila",
                                            "Label: learn Predicted: six",
                                            "Label: learn Predicted: stop",
                                            "Label: learn Predicted: three",
                                            "Label: learn Predicted: tree",
                                            "Label: learn Predicted: two",
                                            "Label: learn Predicted: up",
                                            "Label: learn Predicted: visual",
                                            "Label: learn Predicted: wow",
                                            "Label: learn Predicted: yes",
                                            "Label: learn Predicted: zero",
                                            "Label: left Predicted: backward",
                                            "Label: left Predicted: bed",
                                            "Label: left Predicted: bird",
                                            "Label: left Predicted: cat",
                                            "Label: left Predicted: dog",
                                            "Label: left Predicted: down",
                                            "Label: left Predicted: eight",
                                            "Label: left Predicted: five",
                                            "Label: left Predicted: follow",
                                            "Label: left Predicted: forward",
                                            "Label: left Predicted: four",
                                            "Label: left Predicted: go",
                                            "Label: left Predicted: happy",
                                            "Label: left Predicted: house",
                                            "Label: left Predicted: learn",
                                            "Label: left Predicted: left",
                                            "Label: left Predicted: marvin",
                                            "Label: left Predicted: nine",
                                            "Label: left Predicted: no",
                                            "Label: left Predicted: off",
                                            "Label: left Predicted: on",
                                            "Label: left Predicted: one",
                                            "Label: left Predicted: right",
                                            "Label: left Predicted: seven",
                                            "Label: left Predicted: sheila",
                                            "Label: left Predicted: six",
                                            "Label: left Predicted: stop",
                                            "Label: left Predicted: three",
                                            "Label: left Predicted: tree",
                                            "Label: left Predicted: two",
                                            "Label: left Predicted: up",
                                            "Label: left Predicted: visual",
                                            "Label: left Predicted: wow",
                                            "Label: left Predicted: yes",
                                            "Label: left Predicted: zero",
                                            "Label: marvin Predicted: backward",
                                            "Label: marvin Predicted: bed",
                                            "Label: marvin Predicted: bird",
                                            "Label: marvin Predicted: cat",
                                            "Label: marvin Predicted: dog",
                                            "Label: marvin Predicted: down",
                                            "Label: marvin Predicted: eight",
                                            "Label: marvin Predicted: five",
                                            "Label: marvin Predicted: follow",
                                            "Label: marvin Predicted: forward",
                                            "Label: marvin Predicted: four",
                                            "Label: marvin Predicted: go",
                                            "Label: marvin Predicted: happy",
                                            "Label: marvin Predicted: house",
                                            "Label: marvin Predicted: learn",
                                            "Label: marvin Predicted: left",
                                            "Label: marvin Predicted: marvin",
                                            "Label: marvin Predicted: nine",
                                            "Label: marvin Predicted: no",
                                            "Label: marvin Predicted: off",
                                            "Label: marvin Predicted: on",
                                            "Label: marvin Predicted: one",
                                            "Label: marvin Predicted: right",
                                            "Label: marvin Predicted: seven",
                                            "Label: marvin Predicted: sheila",
                                            "Label: marvin Predicted: six",
                                            "Label: marvin Predicted: stop",
                                            "Label: marvin Predicted: three",
                                            "Label: marvin Predicted: tree",
                                            "Label: marvin Predicted: two",
                                            "Label: marvin Predicted: up",
                                            "Label: marvin Predicted: visual",
                                            "Label: marvin Predicted: wow",
                                            "Label: marvin Predicted: yes",
                                            "Label: marvin Predicted: zero",
                                            "Label: nine Predicted: backward",
                                            "Label: nine Predicted: bed",
                                            "Label: nine Predicted: bird",
                                            "Label: nine Predicted: cat",
                                            "Label: nine Predicted: dog",
                                            "Label: nine Predicted: down",
                                            "Label: nine Predicted: eight",
                                            "Label: nine Predicted: five",
                                            "Label: nine Predicted: follow",
                                            "Label: nine Predicted: forward",
                                            "Label: nine Predicted: four",
                                            "Label: nine Predicted: go",
                                            "Label: nine Predicted: happy",
                                            "Label: nine Predicted: house",
                                            "Label: nine Predicted: learn",
                                            "Label: nine Predicted: left",
                                            "Label: nine Predicted: marvin",
                                            "Label: nine Predicted: nine",
                                            "Label: nine Predicted: no",
                                            "Label: nine Predicted: off",
                                            "Label: nine Predicted: on",
                                            "Label: nine Predicted: one",
                                            "Label: nine Predicted: right",
                                            "Label: nine Predicted: seven",
                                            "Label: nine Predicted: sheila",
                                            "Label: nine Predicted: six",
                                            "Label: nine Predicted: stop",
                                            "Label: nine Predicted: three",
                                            "Label: nine Predicted: tree",
                                            "Label: nine Predicted: two",
                                            "Label: nine Predicted: up",
                                            "Label: nine Predicted: visual",
                                            "Label: nine Predicted: wow",
                                            "Label: nine Predicted: yes",
                                            "Label: nine Predicted: zero",
                                            "Label: no Predicted: backward",
                                            "Label: no Predicted: bed",
                                            "Label: no Predicted: bird",
                                            "Label: no Predicted: cat",
                                            "Label: no Predicted: dog",
                                            "Label: no Predicted: down",
                                            "Label: no Predicted: eight",
                                            "Label: no Predicted: five",
                                            "Label: no Predicted: follow",
                                            "Label: no Predicted: forward",
                                            "Label: no Predicted: four",
                                            "Label: no Predicted: go",
                                            "Label: no Predicted: happy",
                                            "Label: no Predicted: house",
                                            "Label: no Predicted: learn",
                                            "Label: no Predicted: left",
                                            "Label: no Predicted: marvin",
                                            "Label: no Predicted: nine",
                                            "Label: no Predicted: no",
                                            "Label: no Predicted: off",
                                            "Label: no Predicted: on",
                                            "Label: no Predicted: one",
                                            "Label: no Predicted: right",
                                            "Label: no Predicted: seven",
                                            "Label: no Predicted: sheila",
                                            "Label: no Predicted: six",
                                            "Label: no Predicted: stop",
                                            "Label: no Predicted: three",
                                            "Label: no Predicted: tree",
                                            "Label: no Predicted: two",
                                            "Label: no Predicted: up",
                                            "Label: no Predicted: visual",
                                            "Label: no Predicted: wow",
                                            "Label: no Predicted: yes",
                                            "Label: no Predicted: zero",
                                            "Label: off Predicted: backward",
                                            "Label: off Predicted: bed",
                                            "Label: off Predicted: bird",
                                            "Label: off Predicted: cat",
                                            "Label: off Predicted: dog",
                                            "Label: off Predicted: down",
                                            "Label: off Predicted: eight",
                                            "Label: off Predicted: five",
                                            "Label: off Predicted: follow",
                                            "Label: off Predicted: forward",
                                            "Label: off Predicted: four",
                                            "Label: off Predicted: go",
                                            "Label: off Predicted: happy",
                                            "Label: off Predicted: house",
                                            "Label: off Predicted: learn",
                                            "Label: off Predicted: left",
                                            "Label: off Predicted: marvin",
                                            "Label: off Predicted: nine",
                                            "Label: off Predicted: no",
                                            "Label: off Predicted: off",
                                            "Label: off Predicted: on",
                                            "Label: off Predicted: one",
                                            "Label: off Predicted: right",
                                            "Label: off Predicted: seven",
                                            "Label: off Predicted: sheila",
                                            "Label: off Predicted: six",
                                            "Label: off Predicted: stop",
                                            "Label: off Predicted: three",
                                            "Label: off Predicted: tree",
                                            "Label: off Predicted: two",
                                            "Label: off Predicted: up",
                                            "Label: off Predicted: visual",
                                            "Label: off Predicted: wow",
                                            "Label: off Predicted: yes",
                                            "Label: off Predicted: zero",
                                            "Label: on Predicted: backward",
                                            "Label: on Predicted: bed",
                                            "Label: on Predicted: bird",
                                            "Label: on Predicted: cat",
                                            "Label: on Predicted: dog",
                                            "Label: on Predicted: down",
                                            "Label: on Predicted: eight",
                                            "Label: on Predicted: five",
                                            "Label: on Predicted: follow",
                                            "Label: on Predicted: forward",
                                            "Label: on Predicted: four",
                                            "Label: on Predicted: go",
                                            "Label: on Predicted: happy",
                                            "Label: on Predicted: house",
                                            "Label: on Predicted: learn",
                                            "Label: on Predicted: left",
                                            "Label: on Predicted: marvin",
                                            "Label: on Predicted: nine",
                                            "Label: on Predicted: no",
                                            "Label: on Predicted: off",
                                            "Label: on Predicted: on",
                                            "Label: on Predicted: one",
                                            "Label: on Predicted: right",
                                            "Label: on Predicted: seven",
                                            "Label: on Predicted: sheila",
                                            "Label: on Predicted: six",
                                            "Label: on Predicted: stop",
                                            "Label: on Predicted: three",
                                            "Label: on Predicted: tree",
                                            "Label: on Predicted: two",
                                            "Label: on Predicted: up",
                                            "Label: on Predicted: visual",
                                            "Label: on Predicted: wow",
                                            "Label: on Predicted: yes",
                                            "Label: on Predicted: zero",
                                            "Label: one Predicted: backward",
                                            "Label: one Predicted: bed",
                                            "Label: one Predicted: bird",
                                            "Label: one Predicted: cat",
                                            "Label: one Predicted: dog",
                                            "Label: one Predicted: down",
                                            "Label: one Predicted: eight",
                                            "Label: one Predicted: five",
                                            "Label: one Predicted: follow",
                                            "Label: one Predicted: forward",
                                            "Label: one Predicted: four",
                                            "Label: one Predicted: go",
                                            "Label: one Predicted: happy",
                                            "Label: one Predicted: house",
                                            "Label: one Predicted: learn",
                                            "Label: one Predicted: left",
                                            "Label: one Predicted: marvin",
                                            "Label: one Predicted: nine",
                                            "Label: one Predicted: no",
                                            "Label: one Predicted: off",
                                            "Label: one Predicted: on",
                                            "Label: one Predicted: one",
                                            "Label: one Predicted: right",
                                            "Label: one Predicted: seven",
                                            "Label: one Predicted: sheila",
                                            "Label: one Predicted: six",
                                            "Label: one Predicted: stop",
                                            "Label: one Predicted: three",
                                            "Label: one Predicted: tree",
                                            "Label: one Predicted: two",
                                            "Label: one Predicted: up",
                                            "Label: one Predicted: visual",
                                            "Label: one Predicted: wow",
                                            "Label: one Predicted: yes",
                                            "Label: one Predicted: zero",
                                            "Label: right Predicted: backward",
                                            "Label: right Predicted: bed",
                                            "Label: right Predicted: bird",
                                            "Label: right Predicted: cat",
                                            "Label: right Predicted: dog",
                                            "Label: right Predicted: down",
                                            "Label: right Predicted: eight",
                                            "Label: right Predicted: five",
                                            "Label: right Predicted: follow",
                                            "Label: right Predicted: forward",
                                            "Label: right Predicted: four",
                                            "Label: right Predicted: go",
                                            "Label: right Predicted: happy",
                                            "Label: right Predicted: house",
                                            "Label: right Predicted: learn",
                                            "Label: right Predicted: left",
                                            "Label: right Predicted: marvin",
                                            "Label: right Predicted: nine",
                                            "Label: right Predicted: no",
                                            "Label: right Predicted: off",
                                            "Label: right Predicted: on",
                                            "Label: right Predicted: one",
                                            "Label: right Predicted: right",
                                            "Label: right Predicted: seven",
                                            "Label: right Predicted: sheila",
                                            "Label: right Predicted: six",
                                            "Label: right Predicted: stop",
                                            "Label: right Predicted: three",
                                            "Label: right Predicted: tree",
                                            "Label: right Predicted: two",
                                            "Label: right Predicted: up",
                                            "Label: right Predicted: visual",
                                            "Label: right Predicted: wow",
                                            "Label: right Predicted: yes",
                                            "Label: right Predicted: zero",
                                            "Label: seven Predicted: backward",
                                            "Label: seven Predicted: bed",
                                            "Label: seven Predicted: bird",
                                            "Label: seven Predicted: cat",
                                            "Label: seven Predicted: dog",
                                            "Label: seven Predicted: down",
                                            "Label: seven Predicted: eight",
                                            "Label: seven Predicted: five",
                                            "Label: seven Predicted: follow",
                                            "Label: seven Predicted: forward",
                                            "Label: seven Predicted: four",
                                            "Label: seven Predicted: go",
                                            "Label: seven Predicted: happy",
                                            "Label: seven Predicted: house",
                                            "Label: seven Predicted: learn",
                                            "Label: seven Predicted: left",
                                            "Label: seven Predicted: marvin",
                                            "Label: seven Predicted: nine",
                                            "Label: seven Predicted: no",
                                            "Label: seven Predicted: off",
                                            "Label: seven Predicted: on",
                                            "Label: seven Predicted: one",
                                            "Label: seven Predicted: right",
                                            "Label: seven Predicted: seven",
                                            "Label: seven Predicted: sheila",
                                            "Label: seven Predicted: six",
                                            "Label: seven Predicted: stop",
                                            "Label: seven Predicted: three",
                                            "Label: seven Predicted: tree",
                                            "Label: seven Predicted: two",
                                            "Label: seven Predicted: up",
                                            "Label: seven Predicted: visual",
                                            "Label: seven Predicted: wow",
                                            "Label: seven Predicted: yes",
                                            "Label: seven Predicted: zero",
                                            "Label: sheila Predicted: backward",
                                            "Label: sheila Predicted: bed",
                                            "Label: sheila Predicted: bird",
                                            "Label: sheila Predicted: cat",
                                            "Label: sheila Predicted: dog",
                                            "Label: sheila Predicted: down",
                                            "Label: sheila Predicted: eight",
                                            "Label: sheila Predicted: five",
                                            "Label: sheila Predicted: follow",
                                            "Label: sheila Predicted: forward",
                                            "Label: sheila Predicted: four",
                                            "Label: sheila Predicted: go",
                                            "Label: sheila Predicted: happy",
                                            "Label: sheila Predicted: house",
                                            "Label: sheila Predicted: learn",
                                            "Label: sheila Predicted: left",
                                            "Label: sheila Predicted: marvin",
                                            "Label: sheila Predicted: nine",
                                            "Label: sheila Predicted: no",
                                            "Label: sheila Predicted: off",
                                            "Label: sheila Predicted: on",
                                            "Label: sheila Predicted: one",
                                            "Label: sheila Predicted: right",
                                            "Label: sheila Predicted: seven",
                                            "Label: sheila Predicted: sheila",
                                            "Label: sheila Predicted: six",
                                            "Label: sheila Predicted: stop",
                                            "Label: sheila Predicted: three",
                                            "Label: sheila Predicted: tree",
                                            "Label: sheila Predicted: two",
                                            "Label: sheila Predicted: up",
                                            "Label: sheila Predicted: visual",
                                            "Label: sheila Predicted: wow",
                                            "Label: sheila Predicted: yes",
                                            "Label: sheila Predicted: zero",
                                            "Label: six Predicted: backward",
                                            "Label: six Predicted: bed",
                                            "Label: six Predicted: bird",
                                            "Label: six Predicted: cat",
                                            "Label: six Predicted: dog",
                                            "Label: six Predicted: down",
                                            "Label: six Predicted: eight",
                                            "Label: six Predicted: five",
                                            "Label: six Predicted: follow",
                                            "Label: six Predicted: forward",
                                            "Label: six Predicted: four",
                                            "Label: six Predicted: go",
                                            "Label: six Predicted: happy",
                                            "Label: six Predicted: house",
                                            "Label: six Predicted: learn",
                                            "Label: six Predicted: left",
                                            "Label: six Predicted: marvin",
                                            "Label: six Predicted: nine",
                                            "Label: six Predicted: no",
                                            "Label: six Predicted: off",
                                            "Label: six Predicted: on",
                                            "Label: six Predicted: one",
                                            "Label: six Predicted: right",
                                            "Label: six Predicted: seven",
                                            "Label: six Predicted: sheila",
                                            "Label: six Predicted: six",
                                            "Label: six Predicted: stop",
                                            "Label: six Predicted: three",
                                            "Label: six Predicted: tree",
                                            "Label: six Predicted: two",
                                            "Label: six Predicted: up",
                                            "Label: six Predicted: visual",
                                            "Label: six Predicted: wow",
                                            "Label: six Predicted: yes",
                                            "Label: six Predicted: zero",
                                            "Label: stop Predicted: backward",
                                            "Label: stop Predicted: bed",
                                            "Label: stop Predicted: bird",
                                            "Label: stop Predicted: cat",
                                            "Label: stop Predicted: dog",
                                            "Label: stop Predicted: down",
                                            "Label: stop Predicted: eight",
                                            "Label: stop Predicted: five",
                                            "Label: stop Predicted: follow",
                                            "Label: stop Predicted: forward",
                                            "Label: stop Predicted: four",
                                            "Label: stop Predicted: go",
                                            "Label: stop Predicted: happy",
                                            "Label: stop Predicted: house",
                                            "Label: stop Predicted: learn",
                                            "Label: stop Predicted: left",
                                            "Label: stop Predicted: marvin",
                                            "Label: stop Predicted: nine",
                                            "Label: stop Predicted: no",
                                            "Label: stop Predicted: off",
                                            "Label: stop Predicted: on",
                                            "Label: stop Predicted: one",
                                            "Label: stop Predicted: right",
                                            "Label: stop Predicted: seven",
                                            "Label: stop Predicted: sheila",
                                            "Label: stop Predicted: six",
                                            "Label: stop Predicted: stop",
                                            "Label: stop Predicted: three",
                                            "Label: stop Predicted: tree",
                                            "Label: stop Predicted: two",
                                            "Label: stop Predicted: up",
                                            "Label: stop Predicted: visual",
                                            "Label: stop Predicted: wow",
                                            "Label: stop Predicted: yes",
                                            "Label: stop Predicted: zero",
                                            "Label: three Predicted: backward",
                                            "Label: three Predicted: bed",
                                            "Label: three Predicted: bird",
                                            "Label: three Predicted: cat",
                                            "Label: three Predicted: dog",
                                            "Label: three Predicted: down",
                                            "Label: three Predicted: eight",
                                            "Label: three Predicted: five",
                                            "Label: three Predicted: follow",
                                            "Label: three Predicted: forward",
                                            "Label: three Predicted: four",
                                            "Label: three Predicted: go",
                                            "Label: three Predicted: happy",
                                            "Label: three Predicted: house",
                                            "Label: three Predicted: learn",
                                            "Label: three Predicted: left",
                                            "Label: three Predicted: marvin",
                                            "Label: three Predicted: nine",
                                            "Label: three Predicted: no",
                                            "Label: three Predicted: off",
                                            "Label: three Predicted: on",
                                            "Label: three Predicted: one",
                                            "Label: three Predicted: right",
                                            "Label: three Predicted: seven",
                                            "Label: three Predicted: sheila",
                                            "Label: three Predicted: six",
                                            "Label: three Predicted: stop",
                                            "Label: three Predicted: three",
                                            "Label: three Predicted: tree",
                                            "Label: three Predicted: two",
                                            "Label: three Predicted: up",
                                            "Label: three Predicted: visual",
                                            "Label: three Predicted: wow",
                                            "Label: three Predicted: yes",
                                            "Label: three Predicted: zero",
                                            "Label: tree Predicted: backward",
                                            "Label: tree Predicted: bed",
                                            "Label: tree Predicted: bird",
                                            "Label: tree Predicted: cat",
                                            "Label: tree Predicted: dog",
                                            "Label: tree Predicted: down",
                                            "Label: tree Predicted: eight",
                                            "Label: tree Predicted: five",
                                            "Label: tree Predicted: follow",
                                            "Label: tree Predicted: forward",
                                            "Label: tree Predicted: four",
                                            "Label: tree Predicted: go",
                                            "Label: tree Predicted: happy",
                                            "Label: tree Predicted: house",
                                            "Label: tree Predicted: learn",
                                            "Label: tree Predicted: left",
                                            "Label: tree Predicted: marvin",
                                            "Label: tree Predicted: nine",
                                            "Label: tree Predicted: no",
                                            "Label: tree Predicted: off",
                                            "Label: tree Predicted: on",
                                            "Label: tree Predicted: one",
                                            "Label: tree Predicted: right",
                                            "Label: tree Predicted: seven",
                                            "Label: tree Predicted: sheila",
                                            "Label: tree Predicted: six",
                                            "Label: tree Predicted: stop",
                                            "Label: tree Predicted: three",
                                            "Label: tree Predicted: tree",
                                            "Label: tree Predicted: two",
                                            "Label: tree Predicted: up",
                                            "Label: tree Predicted: visual",
                                            "Label: tree Predicted: wow",
                                            "Label: tree Predicted: yes",
                                            "Label: tree Predicted: zero",
                                            "Label: two Predicted: backward",
                                            "Label: two Predicted: bed",
                                            "Label: two Predicted: bird",
                                            "Label: two Predicted: cat",
                                            "Label: two Predicted: dog",
                                            "Label: two Predicted: down",
                                            "Label: two Predicted: eight",
                                            "Label: two Predicted: five",
                                            "Label: two Predicted: follow",
                                            "Label: two Predicted: forward",
                                            "Label: two Predicted: four",
                                            "Label: two Predicted: go",
                                            "Label: two Predicted: happy",
                                            "Label: two Predicted: house",
                                            "Label: two Predicted: learn",
                                            "Label: two Predicted: left",
                                            "Label: two Predicted: marvin",
                                            "Label: two Predicted: nine",
                                            "Label: two Predicted: no",
                                            "Label: two Predicted: off",
                                            "Label: two Predicted: on",
                                            "Label: two Predicted: one",
                                            "Label: two Predicted: right",
                                            "Label: two Predicted: seven",
                                            "Label: two Predicted: sheila",
                                            "Label: two Predicted: six",
                                            "Label: two Predicted: stop",
                                            "Label: two Predicted: three",
                                            "Label: two Predicted: tree",
                                            "Label: two Predicted: two",
                                            "Label: two Predicted: up",
                                            "Label: two Predicted: visual",
                                            "Label: two Predicted: wow",
                                            "Label: two Predicted: yes",
                                            "Label: two Predicted: zero",
                                            "Label: up Predicted: backward",
                                            "Label: up Predicted: bed",
                                            "Label: up Predicted: bird",
                                            "Label: up Predicted: cat",
                                            "Label: up Predicted: dog",
                                            "Label: up Predicted: down",
                                            "Label: up Predicted: eight",
                                            "Label: up Predicted: five",
                                            "Label: up Predicted: follow",
                                            "Label: up Predicted: forward",
                                            "Label: up Predicted: four",
                                            "Label: up Predicted: go",
                                            "Label: up Predicted: happy",
                                            "Label: up Predicted: house",
                                            "Label: up Predicted: learn",
                                            "Label: up Predicted: left",
                                            "Label: up Predicted: marvin",
                                            "Label: up Predicted: nine",
                                            "Label: up Predicted: no",
                                            "Label: up Predicted: off",
                                            "Label: up Predicted: on",
                                            "Label: up Predicted: one",
                                            "Label: up Predicted: right",
                                            "Label: up Predicted: seven",
                                            "Label: up Predicted: sheila",
                                            "Label: up Predicted: six",
                                            "Label: up Predicted: stop",
                                            "Label: up Predicted: three",
                                            "Label: up Predicted: tree",
                                            "Label: up Predicted: two",
                                            "Label: up Predicted: up",
                                            "Label: up Predicted: visual",
                                            "Label: up Predicted: wow",
                                            "Label: up Predicted: yes",
                                            "Label: up Predicted: zero",
                                            "Label: visual Predicted: backward",
                                            "Label: visual Predicted: bed",
                                            "Label: visual Predicted: bird",
                                            "Label: visual Predicted: cat",
                                            "Label: visual Predicted: dog",
                                            "Label: visual Predicted: down",
                                            "Label: visual Predicted: eight",
                                            "Label: visual Predicted: five",
                                            "Label: visual Predicted: follow",
                                            "Label: visual Predicted: forward",
                                            "Label: visual Predicted: four",
                                            "Label: visual Predicted: go",
                                            "Label: visual Predicted: happy",
                                            "Label: visual Predicted: house",
                                            "Label: visual Predicted: learn",
                                            "Label: visual Predicted: left",
                                            "Label: visual Predicted: marvin",
                                            "Label: visual Predicted: nine",
                                            "Label: visual Predicted: no",
                                            "Label: visual Predicted: off",
                                            "Label: visual Predicted: on",
                                            "Label: visual Predicted: one",
                                            "Label: visual Predicted: right",
                                            "Label: visual Predicted: seven",
                                            "Label: visual Predicted: sheila",
                                            "Label: visual Predicted: six",
                                            "Label: visual Predicted: stop",
                                            "Label: visual Predicted: three",
                                            "Label: visual Predicted: tree",
                                            "Label: visual Predicted: two",
                                            "Label: visual Predicted: up",
                                            "Label: visual Predicted: visual",
                                            "Label: visual Predicted: wow",
                                            "Label: visual Predicted: yes",
                                            "Label: visual Predicted: zero",
                                            "Label: wow Predicted: backward",
                                            "Label: wow Predicted: bed",
                                            "Label: wow Predicted: bird",
                                            "Label: wow Predicted: cat",
                                            "Label: wow Predicted: dog",
                                            "Label: wow Predicted: down",
                                            "Label: wow Predicted: eight",
                                            "Label: wow Predicted: five",
                                            "Label: wow Predicted: follow",
                                            "Label: wow Predicted: forward",
                                            "Label: wow Predicted: four",
                                            "Label: wow Predicted: go",
                                            "Label: wow Predicted: happy",
                                            "Label: wow Predicted: house",
                                            "Label: wow Predicted: learn",
                                            "Label: wow Predicted: left",
                                            "Label: wow Predicted: marvin",
                                            "Label: wow Predicted: nine",
                                            "Label: wow Predicted: no",
                                            "Label: wow Predicted: off",
                                            "Label: wow Predicted: on",
                                            "Label: wow Predicted: one",
                                            "Label: wow Predicted: right",
                                            "Label: wow Predicted: seven",
                                            "Label: wow Predicted: sheila",
                                            "Label: wow Predicted: six",
                                            "Label: wow Predicted: stop",
                                            "Label: wow Predicted: three",
                                            "Label: wow Predicted: tree",
                                            "Label: wow Predicted: two",
                                            "Label: wow Predicted: up",
                                            "Label: wow Predicted: visual",
                                            "Label: wow Predicted: wow",
                                            "Label: wow Predicted: yes",
                                            "Label: wow Predicted: zero",
                                            "Label: yes Predicted: backward",
                                            "Label: yes Predicted: bed",
                                            "Label: yes Predicted: bird",
                                            "Label: yes Predicted: cat",
                                            "Label: yes Predicted: dog",
                                            "Label: yes Predicted: down",
                                            "Label: yes Predicted: eight",
                                            "Label: yes Predicted: five",
                                            "Label: yes Predicted: follow",
                                            "Label: yes Predicted: forward",
                                            "Label: yes Predicted: four",
                                            "Label: yes Predicted: go",
                                            "Label: yes Predicted: happy",
                                            "Label: yes Predicted: house",
                                            "Label: yes Predicted: learn",
                                            "Label: yes Predicted: left",
                                            "Label: yes Predicted: marvin",
                                            "Label: yes Predicted: nine",
                                            "Label: yes Predicted: no",
                                            "Label: yes Predicted: off",
                                            "Label: yes Predicted: on",
                                            "Label: yes Predicted: one",
                                            "Label: yes Predicted: right",
                                            "Label: yes Predicted: seven",
                                            "Label: yes Predicted: sheila",
                                            "Label: yes Predicted: six",
                                            "Label: yes Predicted: stop",
                                            "Label: yes Predicted: three",
                                            "Label: yes Predicted: tree",
                                            "Label: yes Predicted: two",
                                            "Label: yes Predicted: up",
                                            "Label: yes Predicted: visual",
                                            "Label: yes Predicted: wow",
                                            "Label: yes Predicted: yes",
                                            "Label: yes Predicted: zero",
                                            "Label: zero Predicted: backward",
                                            "Label: zero Predicted: bed",
                                            "Label: zero Predicted: bird",
                                            "Label: zero Predicted: cat",
                                            "Label: zero Predicted: dog",
                                            "Label: zero Predicted: down",
                                            "Label: zero Predicted: eight",
                                            "Label: zero Predicted: five",
                                            "Label: zero Predicted: follow",
                                            "Label: zero Predicted: forward",
                                            "Label: zero Predicted: four",
                                            "Label: zero Predicted: go",
                                            "Label: zero Predicted: happy",
                                            "Label: zero Predicted: house",
                                            "Label: zero Predicted: learn",
                                            "Label: zero Predicted: left",
                                            "Label: zero Predicted: marvin",
                                            "Label: zero Predicted: nine",
                                            "Label: zero Predicted: no",
                                            "Label: zero Predicted: off",
                                            "Label: zero Predicted: on",
                                            "Label: zero Predicted: one",
                                            "Label: zero Predicted: right",
                                            "Label: zero Predicted: seven",
                                            "Label: zero Predicted: sheila",
                                            "Label: zero Predicted: six",
                                            "Label: zero Predicted: stop",
                                            "Label: zero Predicted: three",
                                            "Label: zero Predicted: tree",
                                            "Label: zero Predicted: two",
                                            "Label: zero Predicted: up",
                                            "Label: zero Predicted: visual",
                                            "Label: zero Predicted: wow",
                                            "Label: zero Predicted: yes",
                                            "Label: zero Predicted: zero"], data=ccm_data)
        
        # Save model's cross confusion matrix data
        filename = "_".join([model_arch[i], str(model_config[i]), model_quantization[i], "ccm"])+".csv"
        path = os.path.join("./results", filename)
        ccm_dataframe.to_csv(path)
    
    # Create general results pandas dataframe        
    dataframe = pd.DataFrame(columns=["Architecture",
                                      "Config",
                                      "Quantization",
                                      "Size [Bytes]",
                                      "Total Parameters",
                                      "Trainable Parameters",
                                      "Non-trainable Parameters",
                                      "Initialization Memory [MB]",
                                      "Overall Memory [MB]",
                                      "Peak Memory [MB]",
                                      "Initialization Time [us]",
                                      "First Inference Time [us]",
                                      "Average Warmup Time [us]",
                                      "Average Inference Time [us]",
                                      "Accuracy Epoch 1",
                                      "Accuracy Epoch 5",
                                      "Accuracy Epoch 10",
                                      "Accuracy Epoch 15",
                                      "Accuracy Epoch 20",
                                      "Accuracy Epoch 25",
                                      "Accuracy Epoch 30",
                                      "Accuracy Epoch 35",
                                      "Accuracy Epoch 40",
                                      "Accuracy Epoch 45",
                                      "Accuracy Epoch 50",
                                      "Accuracy Epoch 55",
                                      "Accuracy Epoch 60",
                                      "Accuracy Epoch 65",
                                      "Accuracy Epoch 70",
                                      "Accuracy Epoch 75",
                                      "Accuracy Epoch 80",
                                      "Accuracy Epoch 85",
                                      "Accuracy Epoch 90",
                                      "Accuracy Epoch 95",
                                      "Accuracy Epoch 100"], data=data)
    
    # Save general results
    filename = "_".join([model_arch[0], str(model_config[0])])+"_results.csv"
    path = os.path.join("./results", filename)
    dataframe.to_csv(path)
    
    return

# Function for converting the architecture number to the corresponding name
# param arch_number:    architecture model given in evaluation script command 
#                       line call
def architecture_lookup(arch_number):
    # Map architecture number to architecture name
    if (arch_number == 0):
        architecture = "ffcn"
    elif (arch_number == 1):
        architecture = "dscnn"
    elif (arch_number == 2):
        architecture = "densenet"
    elif (arch_number == 3):
        architecture = "resnet"
    elif (arch_number == 4):
        architecture = "inception"
    elif (arch_number == 5):
        architecture = "cenet"
    else:
        print("Invalid architecture selected!")
        sys.exit()
                
    return architecture

# Main model evaluation function
# param architecture:               model architecture to evaluate
# param config:                     specific model configuration to evaluate
# param collect_benchmark_data:     if TFLite benchmark data is collected
def evaluate_model(architecture, config, collect_benchmark_data):
    # Arrays for saving all collected data
    model_arch = []
    model_config = []
    model_quantization = []
    model_size = []
    model_params = []
    model_latency = []
    model_memory = []
    model_ccm = []
    model_accuracy = []
    
    # Convert architecture and config parameters to strings    
    arch = architecture_lookup(architecture)
    config = str(config)
    
    # Path to model checkpoint
    config_path = os.path.join("./checkpoints", arch, config)
    
    print("Beginning arch:", arch)
    print("Beginning config:", config)
    
    # Get list of number of epochs trained
    epochs = os.listdir(config_path)
    
    # Remove log file if present
    if "log.log" in epochs:
        epochs.remove("log.log")
        
    # Sort the list of epochs
    for index in range(len(epochs)): 
        epochs[index] = int(epochs[index])
        
    epochs.sort()
    
    for index in range(len(epochs)): 
        epochs[index] = str(epochs[index])
    
    # Arrays for holding data specific to each model config
    # Accuracy of each TFLite model throughout training
    no_quant_accuracy = []
    dynamic_range_accuracy = []
    integer_float_accuracy = []
    integer_only_accuracy = []
    float16_accuracy = []
    
    # Cross confusion matrices of each TFLite model during training
    no_quant_ccm = []
    dynamic_range_ccm = []
    integer_float_ccm = []
    integer_only_ccm = []
    float16_ccm = []
    
    # TFLite model latency
    no_quant_latency = []
    dynamic_range_latency = []
    integer_float_latency = []
    integer_only_latency = []
    float16_latency = []
    
    # Memory requirements of TFLite model
    no_quant_memory = []
    dynamic_range_memory = []
    integer_float_memory = []
    integer_only_memory = []
    float16_memory = []
    
    # Size of each TFLite model (in bytes)
    no_quant_size = 0
    dynamic_range_size = 0
    integer_float_size = 0
    integer_only_size = 0
    float16_size = 0
    
    for epoch in epochs:
        # Only gather data on selected epochs
        if epoch in epochs_to_test:
            print("Beginning epoch:", epoch)
            
            # Get final model path
            epoch_path = os.path.join(config_path, epoch)
            model_path = os.path.join(epoch_path, "checkpoint")
            
            # Only collect parameter count once
            if (epoch == epochs_to_test[-1]):
                total, trainable, nontrainable = get_params(model_path, os.path.join(config_path, "model_summary.txt"))
                
            # Can convert TensorFlow models into 5 different TFLite models
            for i in range(5):
                # No quantization
                if (i == 0):
                    print("Beginning no quantization model")
                    
                    # Create TFLite model
                    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
                    tflite_model = converter.convert()
                    
                    # Save model
                    tflite_model_path = os.path.join(epoch_path, 'tflite_model.tflite')
                    with open(tflite_model_path, 'wb') as f:
                        f.write(tflite_model)
                        
                    # Only measure model size once
                    if (epoch == epochs_to_test[-1]):
                        no_quant_size = os.path.getsize(tflite_model_path)
                        
                        if (collect_benchmark_data == True):
                            [no_quant_latency, no_quant_memory] = get_runtime_stats(tflite_model_path, os.path.join(config_path, "_".join([arch, config, "none.txt"])))
                        else:
                            [no_quant_latency, no_quant_memory] = [[0,0,0,0],[0,0,0]]
                        
                    # Measure accuracy and generate cross confusion matrix
                    [accuracy, cross_confusion_matrix] = get_accuracy(tflite_model_path)
                    no_quant_ccm.append(cross_confusion_matrix)
                    no_quant_accuracy.append(accuracy)
                    
                # Dynamic range quantization
                elif (i == 1):
                    print("Beginning dynamic quantization model")
                    
                    # Create TFLite model
                    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
                    converter.optimizations = [tf.lite.Optimize.DEFAULT]
                    dynamic_range_model = converter.convert()
                    
                    # Save model
                    tflite_model_path = os.path.join(epoch_path, 'dynamic_range_model.tflite')
                    with open(tflite_model_path, 'wb') as f:
                        f.write(dynamic_range_model)
                        
                    # Only measure model size once
                    if (epoch == epochs_to_test[-1]):
                        dynamic_range_size = os.path.getsize(tflite_model_path)
                        
                        if (collect_benchmark_data == True):
                            [dynamic_range_latency, dynamic_range_memory] = get_runtime_stats(tflite_model_path, os.path.join(config_path, "_".join([arch, config, "dynamic_range.txt"])))
                        else:
                            [dynamic_range_latency, dynamic_range_memory] = [[0,0,0,0],[0,0,0]]

                    # Measure accuracy and generate cross confusion matrix                        
                    [accuracy, cross_confusion_matrix] = get_accuracy(tflite_model_path)
                    dynamic_range_ccm.append(cross_confusion_matrix)
                    dynamic_range_accuracy.append(accuracy)
                
                # Integer with float fallback
                elif (i == 2):
                    print("Beginning integer with float fallback quantization model")
                    
                    # Create dataset to use for representative dataset
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
                    
                    # Create TFLite model
                    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
                    converter.optimizations = [tf.lite.Optimize.DEFAULT]
                    converter.representative_dataset = RepresentativeDataset(test_ds, 100)
                    integer_float_model = converter.convert()
                    
                    # Save model
                    tflite_model_path = os.path.join(epoch_path, 'integer_float_model.tflite')
                    with open(tflite_model_path, 'wb') as f:
                        f.write(integer_float_model)
                    
                    # Only measure model size once
                    if (epoch == epochs_to_test[-1]):
                        integer_float_size = os.path.getsize(tflite_model_path)
                        
                        if (collect_benchmark_data == True):
                            [integer_float_latency, integer_float_memory] = get_runtime_stats(tflite_model_path, os.path.join(config_path, "_".join([arch, config, "integer_float.txt"])))
                        else:
                            [integer_float_latency, integer_float_memory] = [[0,0,0,0],[0,0,0]]
                       
                    # Measure accuracy and generate cross confusion matrix
                    [accuracy, cross_confusion_matrix] = get_accuracy(tflite_model_path)
                    integer_float_ccm.append(cross_confusion_matrix)
                    integer_float_accuracy.append(accuracy)
                    
                # Integer only
                elif (i == 3):
                    print("Beginning integer only quantization model")
                    
                    # Create dataset to use for representative dataset
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
                    
                    # Create TFLite model
                    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
                    converter.optimizations = [tf.lite.Optimize.DEFAULT]
                    converter.representative_dataset = RepresentativeDataset(test_ds, 100)
                    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                    converter.inference_input_type = tf.int8  # or tf.uint8
                    converter.inference_output_type = tf.int8  # or tf.uint8
                    integer_only_model = converter.convert()
                    
                    # Save model
                    tflite_model_path = os.path.join(epoch_path, 'integer_only_model.tflite')
                    with open(tflite_model_path, 'wb') as f:
                        f.write(integer_only_model)
                     
                    # Only measure model size once
                    if (epoch == epochs_to_test[-1]):
                        integer_only_size = os.path.getsize(tflite_model_path)
                        
                        if (collect_benchmark_data == True):
                            [integer_only_latency, integer_only_memory] = get_runtime_stats(tflite_model_path, os.path.join(config_path, "_".join([arch, config, "integer_only.txt"])))
                        else:
                            [integer_only_latency, integer_only_memory] = [[0,0,0,0],[0,0,0]]
                       
                    # Measure accuracy and generate cross confusion matrix
                    [accuracy, cross_confusion_matrix] = get_accuracy(tflite_model_path, True)
                    integer_only_ccm.append(cross_confusion_matrix)
                    integer_only_accuracy.append(accuracy)
                     
                # Float16
                elif (i == 4):
                    print("Beginning float16 model")
                    
                    # Create TFLite model
                    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
                    converter.optimizations = [tf.lite.Optimize.DEFAULT]
                    converter.target_spec.supported_types = [tf.float16]
                    float16_model = converter.convert()
                    
                    # Save model
                    tflite_model_path = os.path.join(epoch_path, 'float16_model.tflite')
                    with open(tflite_model_path, 'wb') as f:
                        f.write(float16_model)
                      
                    # Only measure model size once
                    if (epoch == epochs_to_test[-1]):
                        float16_size = os.path.getsize(tflite_model_path)
                        
                        if (collect_benchmark_data == True):
                            [float16_latency, float16_memory] = get_runtime_stats(tflite_model_path, os.path.join(config_path, "_".join([arch, config, "float16.txt"])))
                        else:
                            [float16_latency, float16_memory] = [[0,0,0,0],[0,0,0]]
                      
                    # Measure accuracy and generate cross confusion matrix
                    [accuracy, cross_confusion_matrix] = get_accuracy(tflite_model_path)
                    float16_ccm.append(cross_confusion_matrix)
                    float16_accuracy.append(accuracy)
       
    # Save data for model config (i.e. data for set of 5 TFLite models)
    # No quantization
    model_arch.append(arch)
    model_config.append(int(config))
    model_quantization.append("none")
    model_size.append(no_quant_size)
    model_params.append([total, trainable, nontrainable])
    model_latency.append(no_quant_latency)
    model_memory.append(no_quant_memory)
    model_ccm.append(no_quant_ccm)
    model_accuracy.append(no_quant_accuracy)
    
    # Dynamic range quantization
    model_arch.append(arch)
    model_config.append(int(config))
    model_quantization.append("dynamic_range")
    model_size.append(dynamic_range_size)
    model_params.append([total, trainable, nontrainable])
    model_latency.append(dynamic_range_latency)
    model_memory.append(dynamic_range_memory)
    model_ccm.append(dynamic_range_ccm)
    model_accuracy.append(dynamic_range_accuracy)
    
    # Integer with float fallback
    model_arch.append(arch)
    model_config.append(int(config))
    model_quantization.append("integer_float")
    model_size.append(integer_float_size)
    model_params.append([total, trainable, nontrainable])
    model_latency.append(integer_float_latency)
    model_memory.append(integer_float_memory)
    model_ccm.append(integer_float_ccm)
    model_accuracy.append(integer_float_accuracy)
    
    # Integer only
    model_arch.append(arch)
    model_config.append(int(config))
    model_quantization.append("integer_only")
    model_size.append(integer_only_size)
    model_params.append([total, trainable, nontrainable])
    model_latency.append(integer_only_latency)
    model_memory.append(integer_only_memory)
    model_ccm.append(integer_only_ccm)
    model_accuracy.append(integer_only_accuracy)
    
    # Float16
    model_arch.append(arch)
    model_config.append(int(config))
    model_quantization.append("float16")
    model_size.append(float16_size)
    model_params.append([total, trainable, nontrainable])
    model_latency.append(float16_latency)
    model_memory.append(float16_memory)
    model_ccm.append(float16_ccm)
    model_accuracy.append(float16_accuracy)
            
    # Create final data results file
    create_dataframe(model_arch,
                     model_config,
                     model_quantization,
                     model_size,
                     model_params,
                     model_latency,
                     model_memory,
                     model_ccm,
                     model_accuracy)
    
    return

def main(argv):
    # Variables to hold CLI arguments
    architecture = None
    config = None
    collect_benchmark_data = False
    
    # Parse CLI input
    opts, args = getopt.getopt(argv, "ha:c:b", ["help", "arch=", "config=", "collect_benchmark_data"])
    for opt, arg in opts:
        if opt in ("-h", "--help"): 
            print ("evaluate.py -a <architecture> -c <configuration> -b")
            sys.exit()
        elif opt in ("-a", "--arch"):
            architecture = int(arg)
        elif opt in ("-c", "--config"):
            config = int(arg)
        elif opt in ("-b", "--collect_benchmark_data"):
            collect_benchmark_data = True
    
    # Train selected architecture and configuration
    evaluate_model(architecture=architecture, config=config, collect_benchmark_data=collect_benchmark_data)

if __name__ == "__main__":
   main(sys.argv[1:])
