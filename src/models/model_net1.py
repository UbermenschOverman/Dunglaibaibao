# src/models/model_net1.py

import tensorflow as tf
from tensorflow.keras.layers import Conv1D, ReLU, MaxPool1D, UpSampling1D, Add, Input
from tensorflow.keras.models import Model

from .layers.upsample_layer import ZeroInsertionUpsampling1D

# Kích thước đầu vào cố định
INPUT_SHAPE = (4096, 1)

def conv_block(input_tensor, filters, kernel_size=3, num_layers=5):
    """Xây dựng khối Convolution (Conv Block) gồm 5 lớp Conv 1D."""
    x = input_tensor
    for _ in range(num_layers):
        # Conv1D: kernel=3, stride=1, padding='same' (để giữ kích thước chiều dài)
        x = Conv1D(filters=filters, kernel_size=kernel_size, strides=1, padding='same')(x)
        x = ReLU()(x)
    return x

def ModelNet1(input_shape=(4096, 1)):
    """
    Kiến trúc ModelNet1: CNN Denoising Autoencoder với Max Pooling thông thường.
    Sử dụng 5 lớp Conv trong mỗi khối, tương tự cấu trúc DW-CNN.
    """
    inputs = Input(shape=input_shape)
    
    # --- ENCODER (Down-sampling) ---
    
    # Block 1 (64 filters)
    conv1 = conv_block(inputs, filters=64, num_layers=5) # Output: (B, 64, 4096)
    
    # Pooling 1 (Max Pooling) -> Replaced with Strided Conv for Graph Mode compatibility
    # pool1 = MaxPool1D(pool_size=2, strides=2)(conv1) 
    pool1 = Conv1D(filters=64, kernel_size=3, strides=2, padding='same')(conv1) # Output: (B, 2048, 64)
    pool1 = ReLU()(pool1)
    
    # Block 2 (64 filters)
    conv2 = conv_block(pool1, filters=64, num_layers=5) # Output: (B, 64, 2048)

    # Pooling 2 (Max Pooling) -> Replaced with Strided Conv
    # pool2 = MaxPool1D(pool_size=2, strides=2)(conv2) 
    pool2 = Conv1D(filters=64, kernel_size=3, strides=2, padding='same')(conv2) # Output: (B, 1024, 64)
    pool2 = ReLU()(pool2)

    # Block 3 (Encoder Latent / Bottleneck)
    latent = conv_block(pool2, filters=64, num_layers=5) # Output: (B, 64, 1024)

    # --- DECODER (Up-sampling) ---
    
    # Up-sampling 1
    # Replace UpSampling1D/Conv1DTranspose with custom ZeroInsertionUpsampling1D + Conv1D
    # up1 = upsample_conv(latent, filters=64, kernel_size=2) 
    up1 = ZeroInsertionUpsampling1D(strides=2)(latent)
    up1 = Conv1D(filters=64, kernel_size=2, strides=1, padding='same')(up1) # Output: (B, 2048, 64)

    # Add 1 (Skip connection từ conv2): Cần căn chỉnh filters nếu khác
    # Cả hai đều là 64 filters
    add1 = Add()([up1, conv2]) 

    # Block 4 (Decoder 1)
    conv4 = conv_block(add1, filters=64, num_layers=5) # Output: (B, 64, 2048)

    # Up-sampling 2
    # up2 = upsample_conv(conv4, filters=64, kernel_size=2) 
    up2 = ZeroInsertionUpsampling1D(strides=2)(conv4)
    up2 = Conv1D(filters=64, kernel_size=2, strides=1, padding='same')(up2) # Output: (B, 4096, 64)
    
    # Add 2 (Skip connection từ conv1): Cần căn chỉnh filters nếu khác
    # Cả hai đều là 64 filters
    add2 = Add()([up2, conv1]) 

    # Block 5 (Output Block)
    # Các lớp Conv cuối cùng (5 lớp Conv)
    conv5 = conv_block(add2, filters=64, num_layers=5) # Output: (B, 64, 4096)

    # Output Layer (1 filter, không ReLU, sigmoid/linear cho Regression)
    # Bài báo sử dụng 1x4096 đầu ra, thường là Linear Activation cho Denoising
    outputs = Conv1D(filters=1, kernel_size=3, strides=1, padding='same', activation='linear')(conv5) # Output: (B, 4096, 1)

    model = Model(inputs=inputs, outputs=outputs, name='ModelNet1_MaxPool')
    return model