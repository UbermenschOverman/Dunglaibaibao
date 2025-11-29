# src/models/dnn_dan.py

import tensorflow as tf
from tensorflow.keras.layers import Conv1D, ReLU, MaxPool1D, UpSampling1D, Input
from tensorflow.keras.models import Model

INPUT_SHAPE = (4096, 1) 

def DNN_DAN():
    """
    Kiến trúc DNN-based Denoising Autoencoder (DAN).
    Triển khai như một Convolutional Autoencoder đơn giản (không có skip connections).
    Sử dụng các lớp Conv và Max Pooling/Upsampling thông thường.
    """
    inputs = Input(shape=INPUT_SHAPE)
    
    # --- ENCODER ---
    
    # Layer 1: Conv -> MaxPool
    conv1 = Conv1D(filters=32, kernel_size=16, strides=1, padding='same')(inputs)
    x = ReLU()(conv1)
    x = MaxPool1D(pool_size=4, strides=4)(x) # Output: (B, 1024, 32)

    # Layer 2: Conv -> MaxPool
    x = Conv1D(filters=64, kernel_size=16, strides=1, padding='same')(x)
    x = ReLU()(x)
    latent = MaxPool1D(pool_size=4, strides=4)(x) # Output: (B, 256, 64)
    
    # --- DECODER ---

    # Layer 3: UpSample
    x = UpSampling1D(size=4)(latent) # Output: (B, 1024, 64)
    x = Conv1D(filters=32, kernel_size=16, strides=1, padding='same')(x)
    x = ReLU()(x)

    # Layer 4: UpSample
    x = UpSampling1D(size=4)(x) # Output: (B, 4096, 32)
    
    # Output Layer
    outputs = Conv1D(filters=1, kernel_size=16, strides=1, padding='same', activation='linear')(x) # Output: (B, 4096, 1)

    model = Model(inputs=inputs, outputs=outputs, name='DNN_DAN')
    return model