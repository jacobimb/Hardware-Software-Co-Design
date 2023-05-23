# -*- coding: utf-8 -*-
# Global constants used in neural network training

# Data location
DATASET_PATH = 'data'

# Data characteristics
NUM_CLASSES = 35
AUDIO_LENGTH_SEC = 1
SAMPLE_RATE = 16000

# Dataset parameters
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 128

# Training parameters
EPOCHS = 100

# Spectrogram parameters
NFFT = 512
STEP = 400 
MEL_BANKS = 40
MEL_DB_MAX = 80
