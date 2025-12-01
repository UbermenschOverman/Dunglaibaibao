import tensorflow as tf
from tensorflow.keras.layers import Layer

class ZeroInsertionUpsampling1D(Layer):
    """
    Custom Layer for Zero-Insertion Upsampling (1D).
    Inserts (strides-1) zeros between each sample.
    Output length = Input length * strides.
    
    This layer wraps pure TensorFlow operations to be compatible with Keras Functional API
    and Graph Mode, avoiding 'KerasTensor' errors.
    """
    def __init__(self, strides=2, **kwargs):
        super(ZeroInsertionUpsampling1D, self).__init__(**kwargs)
        self.strides = strides

    def call(self, inputs):
        # inputs: (B, L, C)
        # Create zeros with same shape as inputs
        zeros = tf.zeros_like(inputs)
        
        # Create list of tensors to stack: [x, 0, 0, ...]
        # For stride 2: [x, 0]
        # For stride 4: [x, 0, 0, 0]
        stack_list = [inputs] + [zeros] * (self.strides - 1)
        
        # Stack along new axis 2: (B, L, strides, C)
        stacked = tf.stack(stack_list, axis=2)
        
        # Reshape to (B, L * strides, C)
        shape = tf.shape(inputs)
        B = shape[0]
        L = shape[1]
        C = shape[2]
        
        # Note: We use symbolic shape for B and L to handle dynamic batch/length
        output = tf.reshape(stacked, (B, L * self.strides, C))
        return output

    def compute_output_shape(self, input_shape):
        # input_shape: (B, L, C)
        return (input_shape[0], input_shape[1] * self.strides, input_shape[2])

    def get_config(self):
        config = super(ZeroInsertionUpsampling1D, self).get_config()
        config.update({'strides': self.strides})
        return config
