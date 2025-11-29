# src/models/dw_cnn.py

import tensorflow as tf
from tensorflow.keras.layers import Conv1D, ReLU, Add, Input
from tensorflow.keras.models import Model
from .layers.dwt_layer import DWT1D_Pooling, IDWT1D_UpSampling

INPUT_SHAPE = (4096, 1) 

def dw_conv_block(input_tensor, filters, final_filters=None, num_layers=5):
    """
    Xây dựng khối Convolution (Conv Block).

    :param final_filters: Số filters cho lớp Conv cuối cùng trong Block. 
                          Nếu là None, sử dụng 'filters'.
    """
    x = input_tensor
    
    # 5 lớp Conv 1D với filters
    for _ in range(num_layers):
        # Sử dụng padding='same' để giữ kích thước chiều dài/thời gian
        x = Conv1D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
        x = ReLU()(x)
    
    # Lớp Conv cuối cùng (Block 3 và Block 4 có số filters khác biệt ở cuối)
    if final_filters is not None:
        x = Conv1D(filters=final_filters, kernel_size=3, strides=1, padding='same')(x)
        x = ReLU()(x) # Thêm ReLU theo xu hướng chung của các lớp Conv
    
    return x

def DW_CNN():
    """
    Kiến trúc Deep Wavelet Convolutional Neural Network (DW-CNN).
    Dựa trên cấu trúc Encoder-Decoder và DWT/IDWT (Bảng 2).
    """
    inputs = Input(shape=INPUT_SHAPE)

    # --- ENCODER (Down-sampling) ---
    
    # Block 1 (Layers 1-5): 5x Conv 1D, 64 filters
    conv1 = dw_conv_block(inputs, filters=64, num_layers=5) # Output: (B, 4096, 64)
    
    # Lưu kết nối skip 1
    skip1 = conv1
    
    # Pooling 1 (Layer 6-7): DWT-based Pooling
    pool1 = DWT1D_Pooling()(conv1) # Output: (B, 2048, 128) - DWT doubles channels
    
    # Block 2 (Layers 7-11): 5x Conv 1D, 64 filters
    conv2 = dw_conv_block(pool1, filters=64, num_layers=5) # Output: (B, 2048, 64)

    # Lưu kết nối skip 2
    skip2 = conv2

    # Pooling 2 (Layer 12-13): DWT-based Pooling
    pool2 = DWT1D_Pooling()(conv2) # Output: (B, 1024, 128) - DWT doubles channels

    # Block 3 (Layers 13-18): Bottleneck. 5x Conv 1D (64 filters) + 1x Conv 1D (128 filters)
    # Tổng cộng 6 lớp Conv trong Block này.
    latent = dw_conv_block(pool2, filters=64, final_filters=128, num_layers=5) # Output: (B, 1024, 128)

    # --- DECODER (Up-sampling) ---
    
    # Up-sampling 1 (Layer 19-20): IDWT-based Up-sampling
    up1 = IDWT1D_UpSampling()(latent) # Output: (B, 2048, 64) - IDWT halves channels

    # Add 1 (Layer 20-21): Skip connection từ Block 2 (skip2)
    add1 = Add()([up1, skip2])  # Both (B, 2048, 64)

    # Block 4 (Layers 21-26): 5x Conv 1D (64 filters) + 1x Conv 1D (128 filters)
    conv4 = dw_conv_block(add1, filters=64, final_filters=128, num_layers=5) # Output: (B, 2048, 128)

    # Up-sampling 2 (Layer 26-27): IDWT-based Up-sampling
    up2 = IDWT1D_UpSampling()(conv4) # Output: (B, 4096, 64) - IDWT halves channels
    
    # Add 2 (Layer 27-28): Skip connection từ Block 1 (skip1)
    add2 = Add()([up2, skip1])  # Both (B, 4096, 64) 

    # Output Block (Layers 28-31): 2x Conv 1D (64 filters) + 1x Conv 1D (1 filter)
    conv5 = Conv1D(filters=64, kernel_size=3, strides=1, padding='same')(add2)
    conv5 = ReLU()(conv5)
    
    conv6 = Conv1D(filters=64, kernel_size=3, strides=1, padding='same')(conv5)
    conv6 = ReLU()(conv6)

    # Final Output Layer (Layer 30-31/31): 1 filter, linear activation
    outputs = Conv1D(filters=1, kernel_size=3, strides=1, padding='same', activation='linear')(conv6) # Output: (B, 4096, 1)

    model = Model(inputs=inputs, outputs=outputs, name='DW_CNN')
    return model