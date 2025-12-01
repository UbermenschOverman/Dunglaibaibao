# src/models/dnn_dan.py

import tensorflow as tf
from tensorflow.keras.layers import Conv1D, ReLU, MaxPool1D, UpSampling1D, Input
from tensorflow.keras.models import Model

from .layers.upsample_layer import ZeroInsertionUpsampling1D

INPUT_SHAPE = (4096, 1) 

def DNN_DAN(input_shape=(4096, 1)):
    """
    Kiến trúc DNN-based Denoising Autoencoder (DAN).
    Triển khai như một Convolutional Autoencoder đơn giản (không có skip connections).
    Sử dụng các lớp Conv và Max Pooling/Upsampling thông thường.
    """
    inputs = Input(shape=input_shape)
    
    # --- ENCODER ---
    
    # Layer 1: Conv -> MaxPool -> Replaced with Strided Conv
    conv1 = Conv1D(filters=32, kernel_size=16, strides=1, padding='same')(inputs)
    x = ReLU()(conv1)
    # x = MaxPool1D(pool_size=4, strides=4)(x) 
    x = Conv1D(filters=32, kernel_size=16, strides=4, padding='same')(x) # Output: (B, 1024, 32)
    x = ReLU()(x)

    # Layer 2: Conv -> MaxPool -> Replaced with Strided Conv
    x = Conv1D(filters=64, kernel_size=16, strides=1, padding='same')(x)
    x = ReLU()(x)
    # latent = MaxPool1D(pool_size=4, strides=4)(x) 
    latent = Conv1D(filters=64, kernel_size=16, strides=4, padding='same')(x) # Output: (B, 256, 64)
    latent = ReLU()(latent)
    
    # --- DECODER ---

    # Layer 3: UpSample -> Replaced with ConvTranspose
    # x = UpSampling1D(size=4)(latent) 
    # x = tf.keras.layers.Conv1DTranspose(filters=64, kernel_size=16, strides=4, padding='same')(latent) 
    # x = upsample_conv(latent, filters=64, kernel_size=16, strides=4) 
    x = ZeroInsertionUpsampling1D(strides=4)(latent)
    x = Conv1D(filters=64, kernel_size=16, strides=1, padding='same')(x) # Output: (B, 1024, 64)
    x = Conv1D(filters=32, kernel_size=16, strides=1, padding='same')(x)
    x = ReLU()(x)

    # Layer 4: UpSample -> Replaced with ConvTranspose
    # x = UpSampling1D(size=4)(x) 
    # x = tf.keras.layers.Conv1DTranspose(filters=32, kernel_size=16, strides=4, padding='same')(x) 
    # x = upsample_conv(x, filters=32, kernel_size=16, strides=4) 
    x = ZeroInsertionUpsampling1D(strides=4)(x)
    x = Conv1D(filters=32, kernel_size=16, strides=1, padding='same')(x) # Output: (B, 4096, 32)
    
    # Output Layer
    outputs = Conv1D(filters=1, kernel_size=16, strides=1, padding='same', activation='linear')(x) # Output: (B, 4096, 1)

    model = Model(inputs=inputs, outputs=outputs, name='DNN_DAN')
    return model