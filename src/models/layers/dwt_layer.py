# src/models/layers/dwt_layer.py

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import Constant
from .wavelet_utils import get_haar_filters, get_haar_reconstruction_filters
import numpy as np

class DWT1D_Pooling(Layer):
    """
    Lớp Pooling dựa trên Discrete Wavelet Transform (DWT) 1D.
    Thay thế cho lớp Max Pooling để tránh mất thông tin hiệu quả[cite: 232].

    Thực hiện Convolution với stride=2 và ghép nối (concatenate) cA và cD.
    Đầu vào: (Batch, Length, Channels) -> Đầu ra: (Batch, Length/2, 2*Channels)
    """
    def __init__(self, **kwargs):
        super(DWT1D_Pooling, self).__init__(**kwargs)
        
        # Lấy bộ lọc Haar
        f_L_np, f_H_np = get_haar_filters()
        
        # Kernel size là 2 (tương ứng với độ dài của bộ lọc Haar)
        self.kernel_size = 2 
        
        # Khởi tạo bộ lọc DWT (sẽ được định hình lại trong build)
        self.f_L_init = Constant(f_L_np)
        self.f_H_init = Constant(f_H_np)

    def build(self, input_shape):
        # Input shape is now (Batch, Length, Channels) - channels_last format
        C_in = input_shape[2] # Số kênh đầu vào (channels_last: last dimension)

        # Định hình lại bộ lọc để sử dụng trong tf.nn.conv1d: 
        # (Kernel_size, Channels_in, Channels_out)
        # Trong DWT, mỗi kênh đầu vào tạo ra 1 kênh cA và 1 kênh cD.
        # Channels_out = Channels_in
        
        # Tích chập với f_L để tạo cA
        f_L_kernel = np.zeros((self.kernel_size, C_in, C_in), dtype=np.float32)
        for i in range(C_in):
            f_L_kernel[:, i, i] = get_haar_filters()[0] # Bộ lọc f_L
        
        # Tích chập với f_H để tạo cD
        f_H_kernel = np.zeros((self.kernel_size, C_in, C_in), dtype=np.float32)
        for i in range(C_in):
            f_H_kernel[:, i, i] = get_haar_filters()[1] # Bộ lọc f_H

        # Khởi tạo tham số bộ lọc (non-trainable)
        self.f_L_kernel = self.add_weight(
            name='f_L_kernel',
            shape=(self.kernel_size, C_in, C_in),
            initializer=Constant(f_L_kernel),
            trainable=False
        )
        self.f_H_kernel = self.add_weight(
            name='f_H_kernel',
            shape=(self.kernel_size, C_in, C_in),
            initializer=Constant(f_H_kernel),
            trainable=False
        )
        super(DWT1D_Pooling, self).build(input_shape)

    def call(self, inputs):
        # Input is already in (Batch, Length, Channels) format - channels_last
        # No need to transpose
        
        # Convolution cho cA (Low-pass)
        cA_out = tf.nn.conv1d(
            inputs,
            self.f_L_kernel,
            stride=2, # Down-sampling / Pooling effect 
            padding='VALID' # Không padding để down-sample chính xác
        )
        
        # Convolution cho cD (High-pass)
        cD_out = tf.nn.conv1d(
            inputs,
            self.f_H_kernel,
            stride=2,
            padding='VALID'
        )

        # Ghép nối (Concatenation): (Batch, L/2, C_in) + (Batch, L/2, C_in) -> (Batch, L/2, 2*C_in)
        output = tf.concat([cA_out, cD_out], axis=-1) 
        
        # Return in channels_last format: (Batch, Length/2, 2*Channels)
        return output

    def compute_output_shape(self, input_shape):
        # Input shape: (Batch, Length, Channels)
        L_in = input_shape[1]
        C_in = input_shape[2]
        L_out = L_in // 2 # Kích thước giảm một nửa
        C_out = C_in * 2  # Số kênh tăng gấp đôi
        return (input_shape[0], L_out, C_out)


class IDWT1D_UpSampling(Layer):
    """
    Lớp Up-sampling dựa trên Inverse Discrete Wavelet Transform (IDWT) 1D.
    Tái tạo tín hiệu và đảm bảo độ chính xác của dữ liệu được tái tạo[cite: 464].

    Đầu vào: (Batch, Length/2, 2*Channels) -> Đầu ra: (Batch, Length, Channels)
    """
    def __init__(self, **kwargs):
        super(IDWT1D_UpSampling, self).__init__(**kwargs)
        
        self.kernel_size = 2 # Kích thước bộ lọc
        self.g_L_init = get_haar_reconstruction_filters()[0]
        self.g_H_init = get_haar_reconstruction_filters()[1]

    def build(self, input_shape):
        # Input shape is now (Batch, Length, Channels) - channels_last format
        C_in = input_shape[2] # Số kênh đầu vào (đã được nhân đôi) - channels_last: last dimension
        C_out = C_in // 2 # Số kênh đầu ra

        # Bộ lọc tái tạo IDWT (Tích chập chuyển vị/Upsampling)
        
        # Reconstruction Kernel (g_L): (Kernel_size, C_out, C_in)
        g_L_kernel = np.zeros((self.kernel_size, C_out, C_in), dtype=np.float32)
        # cA là nửa đầu (0:C_out)
        for i in range(C_out):
            g_L_kernel[:, i, i] = self.g_L_init # g_L
        
        # Reconstruction Kernel (g_H): (Kernel_size, C_out, C_in)
        g_H_kernel = np.zeros((self.kernel_size, C_out, C_in), dtype=np.float32)
        # cD là nửa sau (C_out:C_in)
        for i in range(C_out):
            g_H_kernel[:, i, i + C_out] = self.g_H_init # g_H
        
        # Ghép hai kernel lại (cần xem xét cách Keras xử lý Transposed Conv)
        # Với IDWT, ta phải sử dụng tích chập chuyển vị (Conv1DTranspose)
        
        # Định nghĩa Kernel cho IDWT: (Kernel_size, C_out, C_in)
        # Ghép g_L và g_H thành một kernel duy nhất để Conv1DTranspose hoạt động chính xác
        reconstruction_kernel = g_L_kernel + g_H_kernel 
        
        self.reconstruction_kernel = self.add_weight(
            name='reconstruction_kernel',
            shape=(self.kernel_size, C_out, C_in),
            initializer=Constant(reconstruction_kernel),
            trainable=False
        )
        
        # Create Conv1DTranspose layer in build method
        self.conv1d_transpose = tf.keras.layers.Conv1DTranspose(
            filters=C_out,
            kernel_size=self.kernel_size,
            strides=2,
            padding='VALID',
            kernel_initializer=Constant(reconstruction_kernel),
            trainable=False,
            use_bias=False
        )
        super(IDWT1D_UpSampling, self).build(input_shape)

    def call(self, inputs):
        # Input is already in (Batch, Length, Channels) format - channels_last
        # No need to transpose

        # Thực hiện tích chập chuyển vị (Conv1DTranspose)
        # Conv1DTranspose có stride=2 sẽ up-sample kích thước chiều dài/thời gian lên gấp đôi
        
        output = self.conv1d_transpose(inputs)
        
        # Return in channels_last format: (Batch, Length, Channels)
        return output

    def compute_output_shape(self, input_shape):
        # Input shape: (Batch, Length, Channels)
        L_in = input_shape[1]
        C_in = input_shape[2]
        L_out = L_in * 2 # Kích thước tăng gấp đôi
        C_out = C_in // 2 # Số kênh giảm một nửa
        return (input_shape[0], L_out, C_out)