# -*- coding: utf-8 -*-
# Dataset Functions
import tensorflow as tf
import numpy as np
import pathlib

# Function for converting audio waveforms into Mel spectrograms
# Adapted from:
# https://github.com/tensorflow/io/blob/7ce193bd27d728dc34c0d9c0890c23b2e2f0fb03/tensorflow_io/python/ops/audio_ops.py
# https://librosa.org/doc/latest/_modules/librosa/feature/spectral.html
# https://librosa.org/doc/latest/_modules/librosa/core/spectrum.html
# https://towardsdatascience.com/getting-to-know-the-mel-spectrogram-31bca3e2d9d0 (Harvard TinyML Reference)
def get_spectrogram(waveform, sample_rate, nfft, step_size, mel_banks, max_db):
    # Get power spectrogram
    stft = tf.signal.stft(signals=waveform,
                          frame_length=nfft,
                          frame_step=step_size,
                          fft_length=nfft,
                          window_fn=tf.signal.hann_window,
                          pad_end=True)
    spec = tf.math.abs(stft)
    spec = tf.math.square(spec)
    
    # Get equivalent mel spectrogram
    nbins = tf.shape(spec)[-1]
    mel_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_bins=mel_banks,
                                                       num_spectrogram_bins=nbins,
                                                       sample_rate=sample_rate,
                                                       lower_edge_hertz=0,
                                                       upper_edge_hertz=sample_rate/2)
    mel_spec = tf.tensordot(spec, mel_matrix, 1)
    
    # Convert to linear power to dB power
    mel_log_spec = 10.0 * (tf.math.log(mel_spec) / tf.math.log(10.0))
    mel_log_spec = tf.math.maximum(mel_log_spec, tf.math.reduce_max(mel_log_spec) - max_db)

    # Add a `channels` dimension, so that the spectrogram can be used
    # as image-like input data with convolution layers (which expect
    # shape (`batch_size`, `height`, `width`, `channels`).
    mel_log_spec = mel_log_spec[..., tf.newaxis]
    return mel_log_spec

# Function for creating datasets
# Dataset can easily be created using the constants file:
# train_ds, val_ds, test_ds, label_names = create_datasets(const.DATASET_PATH, 
#                                                          const.BATCH_SIZE, 
#                                                          const.VALIDATION_SPLIT, 
#                                                          const.SAMPLE_RATE,
#                                                          const.AUDIO_LENGTH_SEC, 
#                                                          const.NFFT,
#                                                          const.STEP,
#                                                          const.MEL_BANKS,
#                                                          const.MEL_DB_MAX,
#                                                          True,
#                                                          True)
def create_datasets(dataset_path, batch_size, validation_split, sample_rate, audio_length_sec, nfft, step_size, mel_banks, max_db, train_val_test=False, norm=False):
    # Create initial trainign and validation datasets by grabbing files from directory
    data_dir = pathlib.Path(dataset_path)
    train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(directory=data_dir,
                                                                   batch_size=batch_size,
                                                                   validation_split=validation_split,
                                                                   seed=0,
                                                                   output_sequence_length=sample_rate*audio_length_sec,
                                                                   subset='both')
    
    label_names = np.array(train_ds.class_names)
    print("Label names:", label_names)
    
    # Drop extra dimension of audio samples: (batch, samples, None) --> (batch, samples)
    train_ds = train_ds.map(lambda audio, label: (tf.squeeze(audio, axis=-1), label), tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda audio, label: (tf.squeeze(audio, axis=-1), label), tf.data.AUTOTUNE)
    
    # Split validation dataset into testing and validation datasets (if selected)
    if (train_val_test):
        test_ds = val_ds.shard(num_shards=2, index=0)
        val_ds = val_ds.shard(num_shards=2, index=1)

    for example_audio, example_labels in train_ds.take(1):  
        print("Audio data shape:", example_audio.shape)
        print("Audio label shape:", example_labels.shape)
        
    # Convert time series datasets into spectrogram datasets
    train_spec_ds = train_ds.map(lambda audio, label: (get_spectrogram(audio, sample_rate, nfft, step_size, mel_banks, max_db), label), tf.data.AUTOTUNE)
    val_spec_ds = val_ds.map(lambda audio, label: (get_spectrogram(audio, sample_rate, nfft, step_size, mel_banks, max_db), label), tf.data.AUTOTUNE)
    if (train_val_test):
        test_spec_ds = test_ds.map(lambda audio, label: (get_spectrogram(audio, sample_rate, nfft, step_size, mel_banks, max_db), label), tf.data.AUTOTUNE)
    
    # Normalize spectrograms
    if (norm):
        # Find largest spectrogram magnitude from entire dataset
        extrema = []
        
        train_specs = np.concatenate(list(train_spec_ds.map(lambda audio, label: audio)))
        extrema.append(np.abs(np.max(train_specs)))
        extrema.append(np.abs(np.min(train_specs)))
        
        val_specs = np.concatenate(list(val_spec_ds.map(lambda audio, label: audio)))
        extrema.append(np.abs(np.max(val_specs)))
        extrema.append(np.abs(np.min(val_specs)))
        
        if (train_val_test):
            test_specs = np.concatenate(list(test_spec_ds.map(lambda audio, label: audio)))
            extrema.append(np.abs(np.max(test_specs)))
            extrema.append(np.abs(np.min(test_specs)))
            
        max_magnitude = np.max(extrema)
        
        # Divide all spectrograms by largest magnitude; reduce range to [-1, 1]
        train_norm_spec_ds = train_spec_ds.map(lambda audio, label: (tf.cast(audio, tf.float32)/max_magnitude, label), tf.data.AUTOTUNE)
        val_norm_spec_ds = val_spec_ds.map(lambda audio, label: (tf.cast(audio, tf.float32)/max_magnitude, label), tf.data.AUTOTUNE)
        if (train_val_test):
            test_norm_spec_ds = test_spec_ds.map(lambda audio, label: (tf.cast(audio, tf.float32)/max_magnitude, label), tf.data.AUTOTUNE)
    else:
        train_norm_spec_ds = train_spec_ds
        val_norm_spec_ds = val_spec_ds
        if (train_val_test):
            test_norm_spec_ds = test_spec_ds
        

    # Ouptut dataset elements have size [BATCH_SIZE, SAMPLE_RATE*AUDIO_LENGTH_SEC/STEP, MEL_BANKS, 1]
    # (Rows = time, columns = mel feature)
    if (train_val_test):
        return [train_norm_spec_ds, val_norm_spec_ds, test_norm_spec_ds, label_names]
    else:
        return [train_norm_spec_ds, val_norm_spec_ds, label_names]
