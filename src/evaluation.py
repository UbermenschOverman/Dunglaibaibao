
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

def reconstruct_test_data_with_labels(loader):
    """
    Replicates the data generation process to reconstruct X_test and Y_test
    ALONG WITH noise type labels.
    """
    print("\n[EVAL] Reconstructing test data with noise labels...")
    
    # 1. Load Clean Data
    clean_samples = loader.load_clean_data()
    if len(clean_samples) == 0:
        raise ValueError("No clean data found.")

    # 2. Load Noise Data
    noise_samples = loader.load_noise_data()
    if not noise_samples:
        raise ValueError("No noise data found.")

    test_data_by_snr = {}

    # 3. Replicate Generation & Split
    for snr_db in loader.SNR_LEVELS:
        snr_datasets = []
        noise_labels = [] # Track labels corresponding to concatenated data

        # Ensure consistent order: iterate sorted keys if necessary, 
        # but loader uses noise_samples.items(). 
        # Python 3.7+ dicts preserve insertion order. 
        # We must assume loader.load_noise_data() inserts in a deterministic order 
        # (which it does: iterates NOISE_TYPES list).
        
        for noise_type in loader.NOISE_TYPES:
            if noise_type not in noise_samples:
                continue
                
            raw_noise_samples = noise_samples[noise_type]
            N_clean = len(clean_samples)
            N_noise = len(raw_noise_samples)
            
            # Replicate matching logic
            if N_noise < N_clean:
                reps = (N_clean // N_noise) + 1
                noise_cycle = np.tile(raw_noise_samples, (reps, 1))
                noise_matched = noise_cycle[:N_clean]
            else:
                noise_matched = raw_noise_samples[:N_clean]
            
            s_clean_flat = clean_samples.flatten()
            noise_flat = noise_matched.flatten()
            
            # Replicate gain calculation
            gain = loader._calculate_gain(s_clean_flat, noise_flat, snr_db)
            
            # Synthesize
            X_noisy = clean_samples + gain * noise_matched
            Y_clean = clean_samples
            
            snr_datasets.append((X_noisy, Y_clean))
            
            # Create labels for this chunk
            labels = np.full(len(X_noisy), noise_type)
            noise_labels.append(labels)

        # Combine
        X_noisy_combined = np.concatenate([ds[0] for ds in snr_datasets], axis=0)
        Y_clean_combined = np.concatenate([ds[1] for ds in snr_datasets], axis=0)
        Labels_combined = np.concatenate(noise_labels, axis=0)

        # Add channel dim
        X = X_noisy_combined[:, np.newaxis, :]
        Y = Y_clean_combined[:, np.newaxis, :]
        
        # Replicate Split (Random State 42 is CRITICAL)
        # Split 1: Train vs Temp
        X_train, X_temp, Y_train, Y_temp, L_train, L_temp = train_test_split(
            X, Y, Labels_combined, test_size=0.2, random_state=42
        )
        # Split 2: Val vs Test
        X_val, X_test, Y_val, Y_test, L_val, L_test = train_test_split(
            X_temp, Y_temp, L_temp, test_size=0.5, random_state=42
        )
        
        # Transpose for Model Input (B, L, C)
        X_test_t = np.transpose(X_test, (0, 2, 1))
        # Y_test_t = np.transpose(Y_test, (0, 2, 1)) # Not needed for metrics, we use raw
        
        # Store
        test_data_by_snr[snr_db] = {
            "X_test_t": X_test_t,   # Input for model
            "X_test_raw": X_test,   # Raw noisy (B, C, L)
            "Y_test_raw": Y_test,   # Raw clean (B, C, L)
            "Labels": L_test        # Noise types
        }
        
        print(f"  [SNR {snr_db}dB] Reconstructed Test Set: {len(X_test)} samples.")
        
    return test_data_by_snr

def run_full_evaluation():
    print("\n====================================================")
    print("========== STARTING FINAL EVALUATION ===============")
    print("====================================================")

    # 1. Reconstruct Data
    loader = ECGDataLoader(data_root_path=DATA_ROOT, snr_levels=SNR_LEVELS)
    test_data = reconstruct_test_data_with_labels(loader)
    
    results = []
    
    # Models to evaluate
    # Combine both lists
    all_models = {**ABLATION_MODELS, **COMPARATIVE_MODELS}
    
    # 2. Iterate Over SNR Levels
    for snr_db in SNR_LEVELS:
        print(f"\n--- Evaluating SNR {snr_db} dB ---")
        
        data = test_data[snr_db]
        X_in = data["X_test_t"]
        X_raw = data["X_test_raw"]
        Y_raw = data["Y_test_raw"]
        Labels = data["Labels"]
        
        # 3. Iterate Over Models
        for model_label, model_key in all_models.items():
            
            # Determine Checkpoint Path
            # Logic: 
            # - DW_CNN is in 'ablation' (reused in comparative, but source is ablation)
            # - ModelNet1 is in 'ablation'
            # - DNN_DAN / FCN are in 'comparative'
            
            if model_key in ["DW_CNN", "ModelNet1"]:
                run_type = "ablation"
            else:
                run_type = "comparative"
                
            ckpt_path = os.path.join(CHECKPOINT_PATH, run_type, f"SNR_{snr_db}dB", model_key, "best.weights.h5")
            
            if not os.path.exists(ckpt_path):
                print(f"[WARN] Checkpoint not found for {model_key} @ {snr_db}dB. Skipping.")
                continue
                
            # Build Model
            tf.keras.backend.clear_session()
            try:
                model_builder = MODEL_MAPPING[model_key]
                model = model_builder(input_shape=(4096, 1))
                
                # Dummy pass to init layers
                _ = model(tf.zeros((1, 4096, 1)))
                
                # Load Weights
                model.load_weights(ckpt_path)
                
                # Predict
                # Note: Predict on full test set once for efficiency
                Y_pred_t = model.predict(X_in, batch_size=BATCH_SIZE, verbose=0)
                Y_pred_raw = np.transpose(Y_pred_t, (0, 2, 1)) # (B, C, L)
                
                # 4. Evaluate per Noise Type
                unique_labels = np.unique(Labels)
                
                # Overall Metrics
                rmse_all = []
                snr_all = []
                
                for i in range(len(Y_pred_raw)):
                    y_true = Y_raw[i].flatten()
                    y_pred = Y_pred_raw[i].flatten()
                    rmse_all.append(calculate_rmse(y_true, y_pred))
                    snr_all.append(calculate_snr(y_true, y_pred))
                
                results.append({
                    "Model": model_label,
                    "Noise Type": "Average",
                    "SNR Level": snr_db,
                    "RMSE": np.mean(rmse_all),
                    "SNR": np.mean(snr_all)
                })
                
                # Specific Noise Metrics
                for noise_type in unique_labels:
                    indices = np.where(Labels == noise_type)[0]
                    
                    rmse_list = []
                    snr_list = []
                    
                    for idx in indices:
                        y_true = Y_raw[idx].flatten()
                        y_pred = Y_pred_raw[idx].flatten()
                        rmse_list.append(calculate_rmse(y_true, y_pred))
                        snr_list.append(calculate_snr(y_true, y_pred))
                        
                    results.append({
                        "Model": model_label,
                        "Noise Type": noise_type.upper(),
                        "SNR Level": snr_db,
                        "RMSE": np.mean(rmse_list),
                        "SNR": np.mean(snr_list)
                    })
                    
                    # Plotting (Only for 0dB and specific models to avoid clutter)
                    if snr_db == 0 and idx == indices[0]: # Plot first sample of this noise type
                         # Only plot DW-CNN and DAN as requested
                         if model_key in ["DW_CNN", "DNN_DAN"]:
                             out_dir = os.path.join(RESULTS_PATH, "plots_final", f"SNR_{snr_db}dB", noise_type)
                             os.makedirs(out_dir, exist_ok=True)
                             
                             plot_ecg_comparison(
                                 x_noisy=X_raw[idx],
                                 x_clean=Y_raw[idx],
                                 x_denoised_list=[(model_label, Y_pred_raw[idx])],
                                 noise_type=noise_type.upper(),
                                 snr_db=snr_db,
                                 output_path=out_dir,
                                 filename_suffix=f"_{model_key}"
                             )

                print(f"[OK] Evaluated {model_label} @ {snr_db}dB")

            except Exception as e:
                print(f"[ERROR] Evaluating {model_key}: {e}")
                import traceback
                traceback.print_exc()

    # Save Results
    df = pd.DataFrame(results)
    out_csv = os.path.join(RESULTS_PATH, "final_evaluation_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n[SUCCESS] Final evaluation results saved to: {out_csv}")
    
    # Print Summary Table
    print("\n--- Summary (Average) ---")
    print(df[df["Noise Type"] == "Average"].pivot(index="Model", columns="SNR Level", values="SNR"))

if __name__ == "__main__":
    run_full_evaluation()
