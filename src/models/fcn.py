# src/models/fcn.py

import tensorflow as tf
from tensorflow.keras.layers import Conv1D, ReLU, MaxPool1D, UpSampling1D, Input, Concatenate
from tensorflow.keras.models import Model

INPUT_SHAPE = (1, 4096) 

def FCN():
    """
    Kiến trúc Fully Convolutional Network (FCN) Denoising Autoencoder.
    Thường được triển khai theo kiểu U-Net hoặc Denoising AE sử dụng Conv1D.
    Chúng ta sẽ sử dụng cấu trúc tương tự FCN Denoising Autoencoder.
    """
    inputs = Input(shape=INPUT_SHAPE)
    
    # --- ENCODER ---
    
    # Block 1 (Down-sampling)
    conv1 = Conv1D(filters=32, kernel_size=16, strides=1, padding='same', data_format='channels_first')(inputs)
    x = ReLU()(conv1)
    x = MaxPool1D(pool_size=2, strides=2, data_format='channels_first')(x) # Output: (B, 32, 2048)
    
    # Block 2
    conv2 = Conv1D(filters=64, kernel_size=16, strides=1, padding='same', data_format='channels_first')(x)
    x = ReLU()(conv2)
    x = MaxPool1D(pool_size=2, strides=2, data_format='channels_first')(x) # Output: (B, 64, 1024)

    # Block 3
    conv3 = Conv1D(filters=128, kernel_size=16, strides=1, padding='same', data_format='channels_first')(x)
    latent = ReLU()(conv3) # Output: (B, 128, 1024)

    # --- DECODER (Sử dụng Conv1DTranspose hoặc UpSampling + Conv) ---
    
    # Block 4
    x = UpSampling1D(size=2, data_format='channels_first')(latent) # Output: (B, 128, 2048)
    x = Conv1D(filters=64, kernel_size=16, strides=1, padding='same', data_format='channels_first')(x)
    x = ReLU()(x)
    
    # Block 5
    x = UpSampling1D(size=2, data_format='channels_first')(x) # Output: (B, 64, 4096)
    x = Conv1D(filters=32, kernel_size=16, strides=1, padding='same', data_format='channels_first')(x)
    x = ReLU()(x)
    
    # Output Layer
    outputs = Conv1D(filters=1, kernel_size=16, strides=1, padding='same', data_format='channels_first', activation='linear')(x) # Output: (B, 1, 4096)

    model = Model(inputs=inputs, outputs=outputs, name='FCN')
    return model