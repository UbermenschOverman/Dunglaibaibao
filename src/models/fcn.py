# src/models/fcn.py

import tensorflow as tf
from tensorflow.keras.layers import Conv1D, ReLU, MaxPool1D, UpSampling1D, Input
from tensorflow.keras.models import Model

from .layers.upsample_layer import ZeroInsertionUpsampling1D

INPUT_SHAPE = (4096, 1) 

def FCN(input_shape=(4096, 1)):
    """
    Kiến trúc Fully Convolutional Network (FCN) Denoising Autoencoder.
    Thường được triển khai theo kiểu U-Net hoặc Denoising AE sử dụng Conv1D.
    Chúng ta sẽ sử dụng cấu trúc tương tự FCN Denoising Autoencoder.
    """
    inputs = Input(shape=input_shape)
    
    # --- ENCODER ---
    
    # Block 1 (Down-sampling)
    conv1 = Conv1D(filters=32, kernel_size=16, strides=1, padding='same')(inputs)
    x = ReLU()(conv1)
    # x = MaxPool1D(pool_size=2, strides=2)(x) 
    x = Conv1D(filters=32, kernel_size=16, strides=2, padding='same')(x) # Output: (B, 2048, 32)
    x = ReLU()(x)
    
    # Block 2
    conv2 = Conv1D(filters=64, kernel_size=16, strides=1, padding='same')(x)
    x = ReLU()(x)
    # x = MaxPool1D(pool_size=2, strides=2)(x) 
    x = Conv1D(filters=64, kernel_size=16, strides=2, padding='same')(x) # Output: (B, 1024, 64)
    x = ReLU()(x)

    # Block 3
    conv3 = Conv1D(filters=128, kernel_size=16, strides=1, padding='same')(x)
    latent = ReLU()(conv3) # Output: (B, 128, 1024)

    # --- DECODER (Sử dụng Conv1DTranspose hoặc UpSampling + Conv) ---
    
    # Block 4
    # x = UpSampling1D(size=2)(latent) 
    # x = tf.keras.layers.Conv1DTranspose(filters=128, kernel_size=16, strides=2, padding='same')(latent) 
    # x = upsample_conv(latent, filters=128, kernel_size=16, strides=2) 
    x = ZeroInsertionUpsampling1D(strides=2)(latent)
    x = Conv1D(filters=128, kernel_size=16, strides=1, padding='same')(x) # Output: (B, 2048, 128)
    x = Conv1D(filters=64, kernel_size=16, strides=1, padding='same')(x)
    x = ReLU()(x)
    
    # Block 5
    # x = UpSampling1D(size=2)(x) 
    # x = tf.keras.layers.Conv1DTranspose(filters=64, kernel_size=16, strides=2, padding='same')(x) 
    # x = upsample_conv(x, filters=64, kernel_size=16, strides=2) 
    x = ZeroInsertionUpsampling1D(strides=2)(x)
    x = Conv1D(filters=64, kernel_size=16, strides=1, padding='same')(x) # Output: (B, 4096, 64)
    x = Conv1D(filters=32, kernel_size=16, strides=1, padding='same')(x)
    x = ReLU()(x)
    
    # Output Layer
    outputs = Conv1D(filters=1, kernel_size=16, strides=1, padding='same', activation='linear')(x) # Output: (B, 4096, 1)

    model = Model(inputs=inputs, outputs=outputs, name='FCN')
    return model