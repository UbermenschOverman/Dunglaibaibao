import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from datetime import datetime

from .config import *
from .preprocessing.data_loader import ECGDataLoader
from .models import DW_CNN, ModelNet1, DNN_DAN, FCN
from .utils.metrics import calculate_rmse, calculate_snr
from .utils.plotting import plot_ecg_comparison

# Ensure eager execution is disabled for consistency with training (or enabled if needed)
# But for evaluation, graph mode is fine.
# tf.config.run_functions_eagerly(False) 

MODEL_MAPPING = {
    "DW_CNN": DW_CNN,
    "ModelNet1": ModelNet1,
    "DNN_DAN": DNN_DAN,
    "FCN": FCN,
}

def get_checkpoint_path(model_key, snr_db, noise_type=None):
    """
    Constructs the checkpoint path.
    Prioritizes specific noise type checkpoints if they exist.
    Falls back to combined checkpoints.
    """
    
    # Determine base run type
    if model_key in ["DW_CNN", "ModelNet1"]:
        run_type = "ablation"
    else:
        run_type = "comparative"

    # 1. Try Specific Noise Type Path (e.g., checkpoints/comparative/SNR_0dB_BW/ModelName)
    # Note: This path structure is hypothetical based on user request, 
    # but we check for it to support future specific training.
    if noise_type:
        specific_dir = os.path.join(CHECKPOINT_PATH, run_type, f"SNR_{snr_db}dB_{noise_type}", model_key)
        specific_ckpt = os.path.join(specific_dir, "best.weights.h5")
        if os.path.exists(specific_ckpt):
            return specific_ckpt

    # 2. Fallback to Standard (Combined) Path
    combined_dir = os.path.join(CHECKPOINT_PATH, run_type, f"SNR_{snr_db}dB", model_key)
    combined_ckpt = os.path.join(combined_dir, "best.weights.h5")
    
    return combined_ckpt

def load_data_specific(loader, noise_type, snr_db, reconstructed_cache=None):
    """
    Loads test data for a specific noise type and SNR.
    
    1. Checks for physical files: processed/noisy_{noise_type}_{snr_db}dB/
    2. If not found, uses the reconstructed cache (filtering from mixed data).
    """
    
    # 1. Try Physical Loading
    # Note: noise_type in config is uppercase (BW), but folders might be lowercase? 
    # We'll try both or assume standard naming.
    # config.py NOISE_TYPES = ["BW", "EM", "MA"]
    
    specific_path = os.path.join(PROCESSED_DATA_PATH, f"noisy_{noise_type}_{snr_db}dB")
    if os.path.exists(os.path.join(specific_path, "X_test.npy")):
        # Load directly
        X_test = np.load(os.path.join(specific_path, "X_test.npy"))
        Y_test = np.load(os.path.join(specific_path, "Y_test.npy"))
        
        # Transpose for Model Input (B, L, C)
        X_test_t = np.transpose(X_test, (0, 2, 1))
        
        return X_test_t, X_test, Y_test

    # 2. Fallback to Reconstruction (Filtering)
    if reconstructed_cache is None:
        # Should not happen if called correctly, but we can reconstruct here if needed
        # Ideally, we reconstruct ONCE at the start and pass the cache.
        raise ValueError("Physical data not found and no cache provided.")
        
    # Filter from cache
    data = reconstructed_cache[snr_db]
    Labels = data["Labels"]
    
    # Match noise type (case insensitive comparison just in case)
    # Labels are from loader.NOISE_TYPES which are lowercase 'bw', 'em', 'ma' usually
    # But config.py has uppercase.
    # Let's normalize.
    
    target_noise = noise_type.lower()
    
    indices = np.where(Labels == target_noise)[0]
    
    if len(indices) == 0:
        # Try uppercase match if labels are uppercase
        indices = np.where(Labels == noise_type)[0]
        
    if len(indices) == 0:
        print(f"[WARN] No samples found for noise type {noise_type} in reconstructed data.")
        return None, None, None

    X_in = data["X_test_t"][indices]
    X_raw = data["X_test_raw"][indices]
    Y_raw = data["Y_test_raw"][indices]
    
    return X_in, X_raw, Y_raw

def reconstruct_test_data_with_labels(loader):
    """
    Replicates the data generation process to reconstruct X_test and Y_test
    ALONG WITH noise type labels.
    """
    print("\n[EVAL] Reconstructing test data with noise labels (for fallback)...")
    
    clean_samples = loader.load_clean_data()
    noise_samples = loader.load_noise_data()
    
    if len(clean_samples) == 0 or not noise_samples:
        return {}

    test_data_by_snr = {}

    for snr_db in loader.SNR_LEVELS:
        snr_datasets = []
        noise_labels = []

        # Iterate loader.NOISE_TYPES (lowercase 'bw', 'em', 'ma')
        for noise_type in loader.NOISE_TYPES:
            if noise_type not in noise_samples:
                continue
                
            raw_noise_samples = noise_samples[noise_type]
            N_clean = len(clean_samples)
            N_noise = len(raw_noise_samples)
            
            if N_noise < N_clean:
                reps = (N_clean // N_noise) + 1
                noise_cycle = np.tile(raw_noise_samples, (reps, 1))
                noise_matched = noise_cycle[:N_clean]
            else:
                noise_matched = raw_noise_samples[:N_clean]
            
            s_clean_flat = clean_samples.flatten()
            noise_flat = noise_matched.flatten()
            
            gain = loader._calculate_gain(s_clean_flat, noise_flat, snr_db)
            
            X_noisy = clean_samples + gain * noise_matched
            Y_clean = clean_samples
            
            snr_datasets.append((X_noisy, Y_clean))
            labels = np.full(len(X_noisy), noise_type) # labels are lowercase
            noise_labels.append(labels)

        X_noisy_combined = np.concatenate([ds[0] for ds in snr_datasets], axis=0)
        Y_clean_combined = np.concatenate([ds[1] for ds in snr_datasets], axis=0)
        Labels_combined = np.concatenate(noise_labels, axis=0)

        X = X_noisy_combined[:, np.newaxis, :]
        Y = Y_clean_combined[:, np.newaxis, :]
        
        X_train, X_temp, Y_train, Y_temp, L_train, L_temp = train_test_split(
            X, Y, Labels_combined, test_size=0.2, random_state=42
        )
        X_val, X_test, Y_val, Y_test, L_val, L_test = train_test_split(
            X_temp, Y_temp, L_temp, test_size=0.5, random_state=42
        )
        
        X_test_t = np.transpose(X_test, (0, 2, 1))
        
        test_data_by_snr[snr_db] = {
            "X_test_t": X_test_t,
            "X_test_raw": X_test,
            "Y_test_raw": Y_test,
            "Labels": L_test
        }
        
    return test_data_by_snr

def run_full_evaluation():
    print("\n====================================================")
    print("========== STARTING COMPREHENSIVE EVALUATION =======")
    print("====================================================")

    # 1. Prepare Data (Reconstruct Cache for Fallback)
    loader = ECGDataLoader(data_root_path=DATA_ROOT, snr_levels=SNR_LEVELS)
    reconstructed_cache = reconstruct_test_data_with_labels(loader)
    
    results = []
    
    # 2. Iterate ALL Combinations
    # NOISE_TYPES from config (uppercase)
    # SNR_LEVELS from config
    # ALL_MODELS_FOR_EVAL from config
    
    for snr_db in SNR_LEVELS:
        for noise_type in NOISE_TYPES:
            print(f"\n--- Evaluating {noise_type} @ {snr_db} dB ---")
            
            # Load Data for this specific combination
            X_in, X_raw, Y_raw = load_data_specific(loader, noise_type, snr_db, reconstructed_cache)
            
            if X_in is None:
                continue
                
            for model_key in ALL_MODELS_FOR_EVAL:
                
                # Get Checkpoint
                ckpt_path = get_checkpoint_path(model_key, snr_db, noise_type)
                
                if not os.path.exists(ckpt_path):
                    print(f"[WARN] Checkpoint not found for {model_key} (Path: {ckpt_path}). Skipping.")
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
                    
                    # Calculate Metrics
                    rmse_list = []
                    snr_list = []
                    
                    for i in range(len(Y_pred_raw)):
                        y_true = Y_raw[i].flatten()
                        y_pred = Y_pred_raw[i].flatten()
                        rmse_list.append(calculate_rmse(y_true, y_pred))
                        snr_list.append(calculate_snr(y_true, y_pred))
                    
                    avg_rmse = np.mean(rmse_list)
                    avg_snr = np.mean(snr_list)
                    
                    results.append({
                        "Model": model_key, # Use Key for consistency
                        "Noise Type": noise_type,
                        "SNR Level": snr_db,
                        "RMSE": avg_rmse,
                        "SNR": avg_snr
                    })
                    
                    print(f"[OK] {model_key}: RMSE={avg_rmse:.4f}, SNR={avg_snr:.2f} dB")
                    
                    # Plotting (Sample 0 of this noise type, for DW_CNN and DAN at 0dB)
                    if snr_db == 0 and model_key in ["DW_CNN", "DNN_DAN"]:
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
                    # import traceback
                    # traceback.print_exc()

    # Save Results
    df = pd.DataFrame(results)
    out_csv = os.path.join(RESULTS_PATH, "final_evaluation_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n[SUCCESS] Final evaluation results saved to: {out_csv}")
    
    # Pivot Table for easy viewing
    if not df.empty:
        print("\n--- Summary Table (SNR) ---")
        pivot = df.pivot_table(index=["Model", "Noise Type"], columns="SNR Level", values="SNR")
        print(pivot)

if __name__ == "__main__":
    run_full_evaluation()
