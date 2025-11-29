# GPU Setup Guide for DW-CNN ECG Denoising

This guide explains how to configure the codebase to train on GPU instead of CPU.

## Prerequisites

### 1. Check GPU Availability

First, verify that your system has a compatible GPU:

```bash
nvidia-smi
```

This should display your GPU information. If you see "command not found", you need to install NVIDIA drivers.

### 2. Install CUDA and cuDNN

For TensorFlow 2.13+ (recommended):
- **CUDA**: Version 11.8 or 12.0
- **cuDNN**: Version 8.6 or 8.9

Download from:
- CUDA: https://developer.nvidia.com/cuda-downloads
- cuDNN: https://developer.nvidia.com/cudnn

### 3. Install TensorFlow with GPU Support

#### Option A: Using tensorflow[and-cuda] (Recommended for TensorFlow 2.13+)

```bash
pip uninstall tensorflow
pip install tensorflow[and-cuda]
```

This automatically installs the correct CUDA and cuDNN versions.

#### Option B: Manual Installation

```bash
# For TensorFlow 2.13+
pip install tensorflow==2.13.0

# Ensure CUDA and cuDNN are installed separately
```

## Verification

After installation, verify GPU detection:

```python
import tensorflow as tf
print("GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("GPU Details: ", tf.config.list_physical_devices('GPU'))
```

You should see your GPU listed.

## Code Configuration

The codebase has been updated to automatically:
1. **Detect available GPUs** at startup
2. **Enable memory growth** to prevent TensorFlow from allocating all GPU memory
3. **Display GPU status** when training starts

### Automatic GPU Configuration

The `configure_gpu()` function in `src/main.py` will:
- Automatically detect and configure available GPUs
- Enable memory growth (allows other processes to use GPU memory)
- Display warnings if no GPU is found

### Manual GPU Selection (Optional)

If you have multiple GPUs and want to use a specific one, you can modify `src/main.py`:

```python
# Use only GPU 0
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Use only GPU 1
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Use multiple GPUs (GPU 0 and 1)
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
```

Add this **before** importing TensorFlow in `src/main.py`.

## Performance Tips

### 1. Mixed Precision Training (Optional)

For faster training with minimal accuracy loss, you can enable mixed precision. Uncomment these lines in `src/main.py`:

```python
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)
```

**Note**: This requires TensorFlow 2.1+ and a GPU with compute capability 7.0+ (Volta, Turing, Ampere, Ada, Hopper).

### 2. Batch Size Optimization

With GPU, you can typically use larger batch sizes. Update `BATCH_SIZE` in `src/config.py`:

```python
BATCH_SIZE = 1024  # Increase from 512 for GPU
```

### 3. Data Pipeline Optimization

For large datasets, consider using `tf.data.Dataset` for better GPU utilization:

```python
# Convert numpy arrays to tf.data.Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((X_train_t, Y_train_t))
train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
```

## Troubleshooting

### Issue: "No GPU devices found"

**Solutions:**
1. Verify GPU is detected: `nvidia-smi`
2. Check CUDA version: `nvcc --version`
3. Verify TensorFlow GPU installation: `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`
4. Reinstall TensorFlow with GPU support: `pip install tensorflow[and-cuda] --upgrade`

### Issue: "Could not create cuDNN handle"

**Solutions:**
1. Ensure cuDNN is properly installed
2. Check cuDNN version compatibility with TensorFlow
3. Try reinstalling: `pip install tensorflow[and-cuda] --upgrade --force-reinstall`

### Issue: "Out of memory" errors

**Solutions:**
1. Reduce batch size in `src/config.py`
2. Enable memory growth (already done in code)
3. Use gradient accumulation for effective larger batch sizes

### Issue: GPU not being used (CPU is used instead)

**Solutions:**
1. Check that `CUDA_VISIBLE_DEVICES` is not set to empty: `unset CUDA_VISIBLE_DEVICES`
2. Verify TensorFlow can see GPU: `python -c "import tensorflow as tf; print(tf.test.is_gpu_available())"`
3. Check TensorFlow version: `python -c "import tensorflow as tf; print(tf.__version__)"`

## Monitoring GPU Usage

During training, monitor GPU usage:

```bash
# In another terminal
watch -n 1 nvidia-smi
```

This shows real-time GPU memory and utilization.

## Expected Performance Improvement

With GPU training, you should see:
- **10-50x faster** training times (depending on GPU model)
- **Larger batch sizes** possible
- **Better memory efficiency** for large models

Typical speedup:
- CPU: ~10-30 minutes per epoch
- GPU (RTX 3080): ~1-3 minutes per epoch
- GPU (A100): ~30 seconds - 1 minute per epoch

