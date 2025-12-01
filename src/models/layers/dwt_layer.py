# src/models/layers/dwt_layer.py

import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np

class DWT1D_Pooling(Layer):
    """
    1-level Discrete Wavelet Transform (DWT) Pooling Layer using Haar Wavelet.
    Implemented using pure TensorFlow operations for Graph Mode and MirroredStrategy compatibility.
    
    Input: (Batch, Length, Channels)
    Output: (Batch, Length/2, 2*Channels) -> Concatenation of [Approximation (L), Detail (H)]
    """
    def __init__(self, **kwargs):
        super(DWT1D_Pooling, self).__init__(**kwargs)
        self.kernel_size = 2

    def build(self, input_shape):
        C_in = input_shape[-1]
        
        # Haar Constants
        inv_sqrt2 = 1.0 / np.sqrt(2.0)
        
        # Low-pass Filter: [1/sqrt(2), 1/sqrt(2)]
        L_filter = np.array([inv_sqrt2, inv_sqrt2], dtype=np.float32)
        L_filter = L_filter.reshape((2, 1, 1))
        
        # High-pass Filter: [1/sqrt(2), -1/sqrt(2)]
        H_filter = np.array([inv_sqrt2, -inv_sqrt2], dtype=np.float32)
        H_filter = H_filter.reshape((2, 1, 1))
        
        # Identity for channel-wise operation
        eye = np.eye(C_in, dtype=np.float32)
        eye = eye.reshape((1, C_in, C_in))
        
        # Construct Diagonal Kernels: (2, C_in, C_in)
        L_kernel_val = L_filter * eye
        H_kernel_val = H_filter * eye
        
        # Add as non-trainable weights
        self.L_kernel = self.add_weight(
            name='L_kernel',
            shape=L_kernel_val.shape,
            initializer=tf.constant_initializer(L_kernel_val),
            trainable=False,
            dtype=tf.float32
        )
        
        self.H_kernel = self.add_weight(
            name='H_kernel',
            shape=H_kernel_val.shape,
            initializer=tf.constant_initializer(H_kernel_val),
            trainable=False,
            dtype=tf.float32
        )
        
        super(DWT1D_Pooling, self).build(input_shape)

    def call(self, inputs):
        # inputs: (B, L, C)
        
        # DWT via Convolution with stride 2
        # Approximation (Low-pass)
        L_out = tf.nn.conv1d(inputs, self.L_kernel, stride=2, padding='VALID')
        
        # Detail (High-pass)
        H_out = tf.nn.conv1d(inputs, self.H_kernel, stride=2, padding='VALID')
        
        # Concatenate: (B, L/2, 2*C)
        output = tf.concat([L_out, H_out], axis=-1)
        
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] // 2, input_shape[2] * 2)


class IDWT1D_UpSampling(Layer):
    """
    1-level Inverse Discrete Wavelet Transform (IDWT) UpSampling Layer using Haar Wavelet.
    Implemented using pure TensorFlow operations.
    
    Input: (Batch, Length, 2*Channels) -> [Approximation (L), Detail (H)]
    Output: (Batch, Length*2, Channels)
    """
    def __init__(self, **kwargs):
        super(IDWT1D_UpSampling, self).__init__(**kwargs)
        self.kernel_size = 2

    def build(self, input_shape):
        C_total = input_shape[-1]
        self.C_out = C_total // 2
        
        # Haar Constants
        inv_sqrt2 = 1.0 / np.sqrt(2.0)
        
        # Reconstruction Filters (Synthesis)
        # L_syn = [1/sqrt(2), 1/sqrt(2)]
        # H_syn = [1/sqrt(2), -1/sqrt(2)]
        
        L_filter = np.array([inv_sqrt2, inv_sqrt2], dtype=np.float32)
        L_filter = L_filter.reshape((2, 1, 1))
        
        H_filter = np.array([inv_sqrt2, -inv_sqrt2], dtype=np.float32)
        H_filter = H_filter.reshape((2, 1, 1))
        
        eye = np.eye(self.C_out, dtype=np.float32)
        eye = eye.reshape((1, self.C_out, self.C_out))
        
        # Construct Kernels: (2, C_out, C_out)
        L_kernel_val = L_filter * eye
        H_kernel_val = H_filter * eye
        
        # Pre-reverse kernels for Convolution logic in call()
        # We need to convolve with reversed kernel to simulate the synthesis filter application
        # on the upsampled (zero-inserted) data.
        L_kernel_rev = np.flip(L_kernel_val, axis=0)
        H_kernel_rev = np.flip(H_kernel_val, axis=0)
        
        # Add as non-trainable weights
        self.L_kernel = self.add_weight(
            name='L_kernel_rev',
            shape=L_kernel_rev.shape,
            initializer=tf.constant_initializer(L_kernel_rev),
            trainable=False,
            dtype=tf.float32
        )
        
        self.H_kernel = self.add_weight(
            name='H_kernel_rev',
            shape=H_kernel_rev.shape,
            initializer=tf.constant_initializer(H_kernel_rev),
            trainable=False,
            dtype=tf.float32
        )
        
        super(IDWT1D_UpSampling, self).build(input_shape)

    def call(self, inputs):
        # inputs: (B, L, 2*C_out)
        # Split into L (Approximation) and H (Detail)
        L_in = inputs[:, :, :self.C_out]
        H_in = inputs[:, :, self.C_out:]
        
        # Upsample L (Insert zeros)
        # L_in: (B, L, C) -> (B, 2L, C)
        B = tf.shape(L_in)[0]
        Len = tf.shape(L_in)[1]
        
        zeros = tf.zeros_like(L_in)
        
        # Stack: (B, L, 2, C) -> [L0, 0, L1, 0, ...]
        L_stack = tf.stack([L_in, zeros], axis=2)
        L_up = tf.reshape(L_stack, (B, Len * 2, self.C_out))
        
        # Upsample H (Insert zeros)
        H_stack = tf.stack([H_in, zeros], axis=2)
        H_up = tf.reshape(H_stack, (B, Len * 2, self.C_out))
        
        # Pad inputs (Left padding by 1 to align for causal/correct expansion)
        paddings = [[0, 0], [1, 0], [0, 0]]
        L_up_pad = tf.pad(L_up, paddings)
        H_up_pad = tf.pad(H_up, paddings)
        
        # Convolve with Pre-Reversed Synthesis Filters
        # stride=1, padding='VALID'
        L_recon = tf.nn.conv1d(L_up_pad, self.L_kernel, stride=1, padding='VALID')
        H_recon = tf.nn.conv1d(H_up_pad, self.H_kernel, stride=1, padding='VALID')
        
        # Sum
        output = L_recon + H_recon
        
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] * 2, input_shape[2] // 2)