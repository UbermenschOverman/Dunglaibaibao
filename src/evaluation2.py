
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime

from .config import *
from .preprocessing.data_loader import ECGDataLoader
from .models import DW_CNN, ModelNet1, DNN_DAN, FCN

from .utils.plotting import plot_ecg_comparison

# Reuse logic from evaluation.py to ensure consistency
from .evaluation import (
    reconstruct_test_data_with_labels, 
    load_data_specific, 
    get_checkpoint_path,
    MODEL_MAPPING
)

# Ensure eager execution is disabled for consistency
# tf.config.run_functions_eagerly(False) 

def calculate_extended_metrics(y_true_batch, y_pred_batch):
    """
    Calculates MAE, MSE, Cosine, PRD, PSNR for a batch of signals.
    y_true_batch: (B, C, L)
    y_pred_batch: (B, C, L)
    Returns: dictionaries of scalar average metrics
    """
    
    mae_list = []
    mse_list = []
    cos_list = []
    prd_list = []
    psnr_list = []
    
    # Iterate over batch to calculate per-sample metrics
    for i in range(len(y_true_batch)):
        # Flatten: (C*L,)
        y_true = y_true_batch[i].flatten()
        y_pred = y_pred_batch[i].flatten()
        
        diff = y_true - y_pred
        
        # 1. MAE
        mae = np.mean(np.abs(diff))
        mae_list.append(mae)
        
        # 2. MSE
        mse = np.mean(diff ** 2)
        mse_list.append(mse)
        
        # 3. Cosine Similarity
        # dot(a, b) / (|a|*|b|)
        dot = np.dot(y_true, y_pred)
        norm_true = np.linalg.norm(y_true)
        norm_pred = np.linalg.norm(y_pred)
        cosine = dot / ((norm_true * norm_pred) + 1e-12)
        cos_list.append(cosine)
        
        # 4. PRD (%)
        # Definition from user's provided snippet (relative to variance)
        # 100 * sqrt( sum((y - yhat)^2) / sum((y - mean(y))^2) )
        num = np.sum(diff ** 2)
        den = np.sum((y_true - np.mean(y_true)) ** 2) + 1e-12
        prd = 100.0 * np.sqrt(num / den)
        prd_list.append(prd)
        
        # 5. PSNR (dB)
        # 20 * log10( (max - min) / RMSE )
        peak = np.max(y_true) - np.min(y_true)
        rmse = np.sqrt(mse)
        psnr = 20 * np.log10(peak / (rmse + 1e-12))
        psnr_list.append(psnr)
        
    return {
        "MAE": np.mean(mae_list),
        "MSE": np.mean(mse_list),
        "COSINE": np.mean(cos_list),
        "PRD": np.mean(prd_list),
        "PSNR": np.mean(psnr_list)
    }

def run_extended_evaluation():
    print("\n====================================================")
    print("========== STARTING EXTENDED EVALUATION ============")
    print("========== (MAE, MSE, COSINE, PRD, PSNR) ===========")
    print("====================================================")

    # 1. Prepare Data
    loader = ECGDataLoader(data_root_path=DATA_ROOT, snr_levels=SNR_LEVELS)
    
    # We use the same reconstruction cache logic
    try:
        reconstructed_cache = reconstruct_test_data_with_labels(loader)
    except Exception as e:
        print(f"[ERROR] Data reconstruction failed: {e}")
        return

    results = []
    
    # 2. Iterate ALL Combinations
    for snr_db in SNR_LEVELS:
        for noise_type in NOISE_TYPES:
            print(f"\n--- Evaluating {noise_type} @ {snr_db} dB ---")
            
            # Load Data
            X_in, X_raw, Y_raw = load_data_specific(loader, noise_type, snr_db, reconstructed_cache)
            
            if X_in is None:
                continue
                
            for model_key in ALL_MODELS_FOR_EVAL:
                
                # Get Checkpoint
                ckpt_path = get_checkpoint_path(model_key, snr_db, noise_type)
                
                if not os.path.exists(ckpt_path):
                    print(f"[WARN] Checkpoint not found for {model_key}. Skipping.")
                    continue
                
                # Build & Load
                tf.keras.backend.clear_session()
                try:
                    model_builder = MODEL_MAPPING[model_key]
                    model = model_builder(input_shape=(4096, 1))
                    
                    # Dummy pass
                    _ = model(tf.zeros((1, 4096, 1)))
                    
                    model.load_weights(ckpt_path, skip_mismatch=True)
                    
                    # Predict
                    Y_pred_t = model.predict(X_in, batch_size=BATCH_SIZE, verbose=0)
                    Y_pred_raw = np.transpose(Y_pred_t, (0, 2, 1))
                    
                    # Calculate Extended Metrics
                    metrics = calculate_extended_metrics(Y_raw, Y_pred_raw)
                    
                    row = {
                        "Model": model_key,
                        "Noise Type": noise_type,
                        "SNR Level": snr_db,
                        "MAE": metrics["MAE"],
                        "MSE": metrics["MSE"],
                        "COSINE": metrics["COSINE"],
                        "PRD": metrics["PRD"],
                        "PSNR": metrics["PSNR"]
                    }
                    results.append(row)
                    
                    print(f"[OK] {model_key}: MAE={metrics['MAE']:.4f} | PRD={metrics['PRD']:.2f}% | PSNR={metrics['PSNR']:.2f}dB")
                    
                    # Plotting (Sample 0 of this noise type)
                    # We plot for ALL SNRs now to be comprehensive, or restrict if too many.
                    # Let's plot Sample 0 for every combination.
                    out_dir = os.path.join(RESULTS_PATH, "plots_final", f"SNR_{snr_db}dB", noise_type)
                    os.makedirs(out_dir, exist_ok=True)
                    
                    plot_ecg_comparison(
                        x_noisy=X_raw[0],
                        x_clean=Y_raw[0],
                        x_denoised_list=[(model_key, Y_pred_raw[0])],
                        noise_type=noise_type,
                        snr_db=snr_db,
                        output_path=out_dir,
                        filename_suffix=f"_{model_key}"
                    )

                except Exception as e:
                    print(f"[ERROR] Evaluating {model_key}: {e}")
                    import traceback
                    traceback.print_exc()

    # Save Results
    df = pd.DataFrame(results)
    out_csv = os.path.join(RESULTS_PATH, "final_evaluation_metrics_extended.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n[SUCCESS] Extended evaluation results saved to: {out_csv}")
    
    # Pivot Table for easy viewing (Show PSNR)
    if not df.empty:
        print("\n--- Summary Table (PSNR) ---")
        pivot = df.pivot_table(index=["Model", "Noise Type"], columns="SNR Level", values="PSNR")
        print(pivot)

if __name__ == "__main__":
    run_extended_evaluation()