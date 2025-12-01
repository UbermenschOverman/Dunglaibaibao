# src/main.py

import os
import numpy as np
import tensorflow as tf
# # tf.config.run_functions_eagerly(True) # Disabled for performance and multi-GPU support
import pandas as pd

def safe_tensor_to_numpy(tensor):
    """
    Safely convert a tensor to a numpy array, handling both EagerTensor and symbolic Tensor (Graph mode).
    This is crucial for multi-GPU training where eager execution is disabled.
    """
    if hasattr(tensor, 'numpy'):
        return tensor.numpy()
    else:
        # Fallback for graph mode / symbolic tensors
        return tf.make_ndarray(tf.make_tensor_proto(tensor))
from datetime import datetime

# ==============================
# GPU Configuration
# ==============================
def configure_gpu():
    """
    Configure TensorFlow to use GPU with memory growth to avoid allocating all GPU memory.
    """
    # Check for available GPUs
    gpus = tf.config.list_physical_devices('GPU')
    
    if len(gpus) == 0:
        print("[WARNING] No GPU devices found. Training will use CPU.")
        print("[INFO] To use GPU, ensure:")
        print("  1. CUDA and cuDNN are installed")
        print("  2. TensorFlow GPU version is installed: pip install tensorflow[and-cuda]")
        print("  3. GPU drivers are properly installed")
        return False
    
    print(f"[INFO] Found {len(gpus)} GPU(s):")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu.name}")
    
    # Configure GPU memory growth to prevent TensorFlow from allocating all GPU memory
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("[INFO] GPU memory growth enabled.")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(f"[WARNING] Could not set GPU memory growth: {e}")
    
    # Optional: Set mixed precision for better performance (requires TensorFlow 2.1+)
    # Uncomment the following lines if you want to use mixed precision training
    # policy = tf.keras.mixed_precision.Policy('mixed_float16')
    # tf.keras.mixed_precision.set_global_policy(policy)
    # print("[INFO] Mixed precision training enabled.")
    
    return True

# Configure GPU at module import
GPU_AVAILABLE = configure_gpu()

# Import local modules
from .config import *
from .preprocessing.data_loader import ECGDataLoader
from .models import DW_CNN, ModelNet1, DNN_DAN, FCN
from .utils.metrics import calculate_rmse, calculate_snr
from .utils.plotting import plot_ecg_comparison


# ==============================
# Model builder mapping
# ==============================
MODEL_MAPPING = {
    "DW_CNN": DW_CNN,
    "ModelNet1": ModelNet1,
    "DNN_DAN": DNN_DAN,
    "FCN": FCN,
}


# ==============================
# Data loading
# ==============================
def load_data_for_snr(snr_db):
    """
    Load preprocessed data corresponding to a specific SNR level.
    The data is loaded in (batch, channels, length) format and transposed 
    to (batch, length, channels) to meet Keras's standard Conv1D input (Length, Channel).
    """

    snr_path = os.path.join(PROCESSED_DATA_PATH, f"noisy_{snr_db}dB")

    if not os.path.exists(snr_path):
        # Trả về None nếu thư mục không tồn tại
        print(f"[ERROR] Path does not exist: {snr_path}")
        return None

    try:
        X_train = np.load(os.path.join(snr_path, "X_train.npy"))
        Y_train = np.load(os.path.join(snr_path, "Y_train.npy"))
        X_val = np.load(os.path.join(snr_path, "X_val.npy"))
        Y_val = np.load(os.path.join(snr_path, "Y_val.npy"))
        X_test = np.load(os.path.join(snr_path, "X_test.npy"))
        Y_test = np.load(os.path.join(snr_path, "Y_test.npy"))

        # Dữ liệu được lưu ở (batch, channels, length)
        # CHUYỂN ĐỔI sang định dạng chuẩn của Keras: (batch, length, channels)
        # Các mô hình DW_CNN/ModelNet1/... được thiết kế với Conv1D(data_format='channels_first')
        # Tuy nhiên, TensorFlow/Keras vẫn thường cần shape (Length, Channels)
        
        X_train_t = np.transpose(X_train, (0, 2, 1)) # (B, L, C)
        X_val_t = np.transpose(X_val, (0, 2, 1))
        X_test_t = np.transpose(X_test, (0, 2, 1))

        Y_train_t = np.transpose(Y_train, (0, 2, 1))
        Y_val_t = np.transpose(Y_val, (0, 2, 1))
        Y_test_t = np.transpose(Y_test, (0, 2, 1))
        
        # TRẢ VỀ: (B, L, C) cho fit/predict, và (B, C, L) cho metrics
        # (X, Y) là (B, L, C) và (X_raw, Y_raw) là (B, C, L)
        return X_train_t, Y_train_t, X_val_t, Y_val_t, X_test_t, Y_test_t, X_train, Y_train, X_val, Y_val, X_test, Y_test


    except FileNotFoundError:
        print(f"[ERROR] Missing .npy files for SNR={snr_db} dB")
        return None


# ==============================
# Training & Evaluation
# ==============================
def train_and_evaluate(model_name, snr_db,
                       X_train_t, Y_train_t, X_val_t, Y_val_t, X_test_t, Y_test_t, # (B, L, C) for Keras
                       X_test_raw, Y_test_raw, # (B, C, L) for Metrics/Raw data
                       run_type,
                       skip_train=False,
                       custom_checkpoint_dir=None):
    """
    Train model, save best checkpoint, evaluate on test set.
    If skip_train is True, skip training and load weights from custom_checkpoint_dir (or default).
    """

    print(f"\n========== PROCESSING {model_name} @ {snr_db} dB ==========")

    model_builder = MODEL_MAPPING.get(model_name)
    if model_builder is None:
        raise ValueError(f"Model '{model_name}' not found in MODEL_MAPPING.")
    
    # Khởi tạo mô hình
    # CHÚ Ý QUAN TRỌNG: Các hàm tạo mô hình (DW_CNN, v.v.) phải được sửa 
    # để sử dụng Input(shape=(4096, 1)) thay vì Input(shape=(1, 4096))
    # Hoặc ta phải truyền INPUT_SHAPE=(4096, 1) vào đây.
    model = model_builder(input_shape=(4096, 1))
    # Instantiate fresh optimizer and loss to avoid state leakage across experiments
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.MeanSquaredError()
    
    model.compile(optimizer=optimizer, loss=loss_fn)

    # Checkpoint directory
    if custom_checkpoint_dir:
        checkpoint_dir = custom_checkpoint_dir
    else:
        checkpoint_dir = os.path.join(
            CHECKPOINT_PATH, run_type, f"SNR_{snr_db}dB", model_name
        )
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, "best.weights.h5")

    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=ckpt_path,
        save_weights_only=True,
        monitor="val_loss",
        mode="min",
        save_best_only=True
    )

    # -----------------------
    # Training
    # -----------------------
    history = None
    
    if not skip_train:
        print(f"[INFO] Training {model_name}...")
        # Kiểm tra checkpoint trước khi fit
        initial_epoch = 0
        if os.path.exists(ckpt_path):
            print(f"[INFO] Loading existing checkpoint: {ckpt_path}")
            # Build model with dummy input to ensure all layers (including Conv1DTranspose) are initialized
            # Use hardcoded shape (4096, 1) to match model input
            dummy_input = tf.zeros((1, 4096, 1))
            _ = model(dummy_input)
            model.load_weights(ckpt_path)
            # Nếu muốn có log epoch trước đó, bạn có thể lưu epoch train trước vào file
            # tạm thời set initial_epoch=0, fit thêm EPOCHS vẫn ok

        history = model.fit(
            X_train_t, Y_train_t,
            validation_data=(X_val_t, Y_val_t),
            epochs=EPOCHS,
            initial_epoch=initial_epoch, 
            batch_size=BATCH_SIZE,
            callbacks=[ckpt_callback],
            verbose=1
        )
    else:
        print(f"[INFO] Skipping training for {model_name} (Using existing checkpoint).")
        if not os.path.exists(ckpt_path):
             # Fallback: if user wanted to skip but file missing, maybe warn or error?
             # For now, let's raise error because we expect it to exist.
             raise FileNotFoundError(f"Expected checkpoint not found at {ckpt_path} for skipped training.")

    # Load best weights (whether trained or skipped)
    print(f"[INFO] Loading best weights: {ckpt_path}")
    try:
        # Ensure model is built
        dummy_input = tf.zeros((1, 4096, 1))
        _ = model(dummy_input)
        model.load_weights(ckpt_path)
    except Exception as e:
        print(f"[WARNING] Could not load best weights: {e}")

    # -----------------------
    # Evaluation
    # -----------------------
    Y_pred_t = model.predict(X_test_t, batch_size=BATCH_SIZE)

    # Chuyển Y_pred về định dạng (batch, channels, length) (Raw format) để tính metrics
    Y_pred_raw = np.transpose(Y_pred_t, (0, 2, 1))

    rmse_list = []
    snr_list = []

    # Convert tensors to numpy explicitly nếu cần (graph mode)
    Y_pred_raw_np = safe_tensor_to_numpy(Y_pred_raw)

    for i in range(len(Y_pred_raw_np)):
        # Convert từ (channels, length) to 1D cho metrics calculation
        y_test_flat = Y_test_raw[i].flatten()
        y_pred_flat = Y_pred_raw_np[i].flatten()
        rmse_list.append(calculate_rmse(y_test_flat, y_pred_flat))
        snr_list.append(calculate_snr(y_test_flat, y_pred_flat))

    avg_rmse = np.mean(rmse_list)
    avg_snr = np.mean(snr_list)

    print(f"\n[RESULT] {model_name} @ {snr_db} dB")
    print(f"Avg RMSE: {avg_rmse:.4f}")
    print(f"Avg SNR : {avg_snr:.2f} dB")

    return avg_rmse, avg_snr, history, Y_pred_raw


# ==============================
# Experiment Runner
# ==============================
def run_experiments(model_dict, run_type):
    """
    Run all experiments (ablation/comparative) for all SNR levels.
    """

    results = []

    for snr_db in SNR_LEVELS:

        print("\n====================================================")
        print(f"============== RUNNING @ {snr_db} dB ===============")
        print("====================================================")

        data = load_data_for_snr(snr_db)
        if data is None:
            continue
        
        # Bóc tách dữ liệu: 6 tensor (B, L, C) cho training, 4 tensor (B, C, L) cho metrics
        X_train_t, Y_train_t, X_val_t, Y_val_t, X_test_t, Y_test_t, \
        X_train_raw, Y_train_raw, X_val_raw, Y_val_raw, X_test_raw, Y_test_raw = data

        for model_label, model_name in model_dict.items():
            
            # Clear session to prevent graph state leakage between models
            tf.keras.backend.clear_session()

            # Determine if we should skip training (reuse ablation checkpoint)
            skip_train = False
            custom_ckpt_dir = None
            
            if run_type == "comparative" and model_name == "DW_CNN":
                # Reuse checkpoint from ablation experiment
                skip_train = True
                custom_ckpt_dir = os.path.join(
                    CHECKPOINT_PATH, "ablation", f"SNR_{snr_db}dB", "DW_CNN"
                )
                print(f"[INFO] Configured to reuse Ablation checkpoint for {model_name}")

            # Standard Checkpoint Directory (if not custom)
            if custom_ckpt_dir:
                checkpoint_dir = custom_ckpt_dir
            else:
                checkpoint_dir = os.path.join(
                    CHECKPOINT_PATH, run_type, f"SNR_{snr_db}dB", model_name
                )
            
            # Check if experiment is already completed (only if NOT skipping training logic above)
            # If we are reusing ablation checkpoint, we still want to run evaluation to get metrics for the comparative table.
            # But if it's a standard run, we check for TRAINING_COMPLETE.
            
            done_marker = os.path.join(checkpoint_dir, "TRAINING_COMPLETE")
            
            # If it's a standard run and marked done, skip training but run evaluation
            if not skip_train and os.path.exists(done_marker):
                print(f"[INFO] Experiment {model_name} @ {snr_db} dB is marked complete. Skipping training and running evaluation.")
                skip_train = True

            import traceback
            try:
                avg_rmse, avg_snr, history, Y_pred_raw = train_and_evaluate(
                    model_name,
                    snr_db,
                    X_train_t, Y_train_t, X_val_t, Y_val_t, X_test_t, Y_test_t,
                    X_test_raw, Y_test_raw, # Dữ liệu thô cho Metrics/Plotting
                    run_type,
                    skip_train=skip_train,
                    custom_checkpoint_dir=custom_ckpt_dir
                )
                
                # Create completion marker only if we actually trained
                if not skip_train:
                    with open(done_marker, "w") as f:
                        f.write(f"Completed at {datetime.now()}")

                # Save result row
                results.append({
                    "Experiment": run_type,
                    "Model": model_label,
                    "Model_Key": model_name,
                    "SNR(dB)": snr_db,
                    "RMSE": avg_rmse,
                    "SNR": avg_snr,
                    "Time": datetime.now().strftime("%Y-%m-%d_%H:%M")
                })

                # Visualization (optional — default: only DW-CNN)
                if run_type == "comparative" and model_name == "DW_CNN":
                    out_dir = os.path.join(RESULTS_PATH, run_type, f"SNR_{snr_db}dB")
                    os.makedirs(out_dir, exist_ok=True)

                    # Dữ liệu X_test_raw, Y_test_raw và Y_pred_raw đều là (B, C, L)
                    plot_ecg_comparison(
                        x_noisy=X_test_raw[0],
                        x_clean=Y_test_raw[0],
                        x_denoised_list=[("DW-CNN", Y_pred_raw[0])],
                        noise_type="Combined",
                        snr_db=snr_db,
                        output_path=out_dir
                    )
            except Exception as e:
                print(f"[ERROR] Failed to process {model_name} @ {snr_db} dB: {e}")
                traceback.print_exc()


    # Save results table
    results_df = pd.DataFrame(results)
    out_csv = os.path.join(
        RESULTS_PATH, f"{run_type}_results_{datetime.now().strftime('%Y%m%d')}.csv"
    )
    results_df.to_csv(out_csv, index=False)
    print(f"\n[INFO] Saved results to: {out_csv}")


# ==============================
# Main
# ==============================
if __name__ == "__main__":

    print("\n--- SETUP COMPLETE ---")
    
    # Display GPU status
    if GPU_AVAILABLE:
        print(f"[INFO] Training will use GPU: {tf.config.list_physical_devices('GPU')[0].name}")
    else:
        print("[INFO] Training will use CPU (GPU not available)")

    # -----------------------------------------------------------------
    # 1. KÍCH HOẠT TIỀN XỬ LÝ (CHỈ CẦN CHẠY MỘT LẦN)
    # -----------------------------------------------------------------
    # print("\n--- BƯỚC 1: TIỀN XỬ LÝ DỮ LIỆU (Đang chạy...) ---")
    # loader = ECGDataLoader(data_root_path=DATA_ROOT)
    # loader.run_preprocessing_and_save() 
    # print("--- TIỀN XỬ LÝ HOÀN THÀNH. Dữ liệu .npy đã được tạo. ---")

    # -----------------------------------------------------------------
    # 2. KÍCH HOẠT THÍ NGHIỆM
    # -----------------------------------------------------------------
    
    # Chạy Thí nghiệm Loại trừ (Ablation Experiments)
    run_experiments(ABLATION_MODELS, run_type="ablation")
    
    # Chạy Thí nghiệm So sánh (Comparative Experiments)
    run_experiments(COMPARATIVE_MODELS, run_type="comparative")