# -*- coding: utf-8 -*-
# Wrapper script to prepare model for deployment on Arduino Nano 33 BLE
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
import getopt

# RepresentativeDataset class for TensorFlow quantization
class RepresentativeDataset():
    def __init__(self, dataset, samples):
        # Take input dataset, change batch size to one, and take random samples
        self.dataset = dataset.unbatch().batch(1).take(samples)
        
    def __call__(self):
        # Generator function
        for data, _ in self.dataset:
            yield [tf.dtypes.cast(data, tf.float32)]

# Wrap model to use flattened input
# param sub_model:  TensorFlow model
def wrap_model(sub_model):
    inputs = tf.keras.layers.Input(shape=(40*40))
    reshape = tf.keras.layers.Reshape((40,40,1), trainable=False)(inputs)
    outputs = sub_model(reshape)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
      
    return model

# Prepare model for deployment on Arduino Nano 33 BLE 
# param model_input_path:  path to saved model directory
# param output_directory:  directory to save wrapped model
def prepare_model(model_input_path, output_directory):
    # Load TensorFlow model add necessary layers
    sub_model = tf.keras.models.load_model(model_input_path)
    sub_model.trainable = False
    model = wrap_model(sub_model)
    
    # Save modified model
    output_saved_model_path = os.path.join(output_directory, 'saved_model')
    tf.keras.models.save_model(model, output_saved_model_path)
    
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
    converter = tf.lite.TFLiteConverter.from_saved_model(output_saved_model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = RepresentativeDataset(test_ds, 100)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8  # or tf.uint8
    converter.inference_output_type = tf.int8  # or tf.uint8
    integer_only_model = converter.convert()
    
    # Save final TFLite model
    output_model_path = os.path.join(output_directory, 'wrapped_integer_only_model.tflite')
    with open(output_model_path, 'wb') as f:
        f.write(integer_only_model)
    
    return

def main(argv):
    # Variables to hold CLI arguments
    model_input_path = ""
    output_directory = ""
        
    # Parse CLI input
    opts, args = getopt.getopt(argv, "hi:o:", ["help", "input_model_path=", "output_dir="])
    for opt, arg in opts:
        if opt in ("-h", "--help"): 
            print ("prepare_model.py -i <input model path> -o <output directory>")
            sys.exit()
        elif opt in ("-i", "--input_model_path"):
            model_input_path = arg
        elif opt in ("-o", "--output_dir"):
            output_directory = arg
        
    # Prepare model for deployment on Arduino Nano 33 BLE
    prepare_model(model_input_path=model_input_path, output_directory=output_directory)

if __name__ == "__main__":
   main(sys.argv[1:])
