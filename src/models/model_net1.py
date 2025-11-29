# src/models/model_net1.py

import tensorflow as tf
from tensorflow.keras.layers import Conv1D, ReLU, MaxPool1D, UpSampling1D, Add, Input
from tensorflow.keras.models import Model

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

def ModelNet1():
    """
    Kiến trúc ModelNet1: CNN Denoising Autoencoder với Max Pooling thông thường.
    Sử dụng 5 lớp Conv trong mỗi khối, tương tự cấu trúc DW-CNN.
    """
    inputs = Input(shape=INPUT_SHAPE)
    
    # --- ENCODER (Down-sampling) ---
    
    # Block 1 (64 filters)
    conv1 = conv_block(inputs, filters=64, num_layers=5) # Output: (B, 64, 4096)
    
    # Pooling 1 (Max Pooling)
    pool1 = MaxPool1D(pool_size=2, strides=2)(conv1) # Output: (B, 2048, 64)
    
    # Block 2 (64 filters)
    conv2 = conv_block(pool1, filters=64, num_layers=5) # Output: (B, 64, 2048)

    # Pooling 2 (Max Pooling)
    pool2 = MaxPool1D(pool_size=2, strides=2)(conv2) # Output: (B, 1024, 64)

    # Block 3 (Encoder Latent / Bottleneck)
    latent = conv_block(pool2, filters=64, num_layers=5) # Output: (B, 64, 1024)

    # --- DECODER (Up-sampling) ---
    
    # Up-sampling 1
    up1 = UpSampling1D(size=2)(latent) # Output: (B, 2048, 64)

    # Add 1 (Skip connection từ conv2): Cần căn chỉnh filters nếu khác
    # Cả hai đều là 64 filters
    add1 = Add()([up1, conv2]) 

    # Block 4 (Decoder 1)
    conv4 = conv_block(add1, filters=64, num_layers=5) # Output: (B, 64, 2048)

    # Up-sampling 2
    up2 = UpSampling1D(size=2)(conv4) # Output: (B, 4096, 64)
    
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