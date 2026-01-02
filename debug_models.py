import os
import tensorflow as tf
import numpy as np
from src.models.fcn import FCN
from src.models.dnn_dan import DNN_DAN
from src.main import safe_tensor_to_numpy

# Disable eager to simulate graph mode issues
tf.config.run_functions_eagerly(False)

def test_model(model_builder, name):
    print(f"\nTesting {name}...")
    try:
        model = model_builder(input_shape=(4096, 1))
        model.compile(optimizer='adam', loss='mse')
        
        x = tf.random.normal((2, 4096, 1))
        y = tf.random.normal((2, 4096, 1))
        
        print("  Running fit...")
        model.fit(x, y, epochs=1, verbose=1)
        print(f"  {name} PASSED fit.")
        
        print("  Running predict...")
        pred = model.predict(x)
        print(f"  {name} PASSED predict. Output shape: {pred.shape}")
        
    except Exception as e:
        print(f"  {name} FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model(FCN, "FCN")
    test_model(DNN_DAN, "DNN_DAN")
