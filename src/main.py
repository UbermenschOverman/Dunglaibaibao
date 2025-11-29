# src/main.py

import os
import numpy as np
import tensorflow as tf
import pandas as pd
from datetime import datetime

# Import local modules
from src.config import *
from src.preprocessing.data_loader import ECGDataLoader
from src.models import DW_CNN, ModelNet1, DNN_DAN, FCN
from src.utils.metrics import calculate_rmse, calculate_snr
from src.utils.plotting import plot_ecg_comparison


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
    Ensures the returned shape is (batch, length, channels).
    """

    snr_path = os.path.join(PROCESSED_DATA_PATH, f"noisy_{snr_db}dB")

    if not os.path.exists(snr_path):
        print(f"[ERROR] Path does not exist: {snr_path}")
        return None

    try:
        X_train = np.load(os.path.join(snr_path, "X_train.npy"))
        Y_train = np.load(os.path.join(snr_path, "Y_train.npy"))
        X_val = np.load(os.path.join(snr_path, "X_val.npy"))
        Y_val = np.load(os.path.join(snr_path, "Y_val.npy"))
        X_test = np.load(os.path.join(snr_path, "X_test.npy"))
        Y_test = np.load(os.path.join(snr_path, "Y_test.npy"))

        # Expected raw shape: (batch, channels, length)
        # Convert to Keras Conv1D shape: (batch, length, channels)
        X_train = np.transpose(X_train, (0, 2, 1))
        X_val = np.transpose(X_val, (0, 2, 1))
        X_test = np.transpose(X_test, (0, 2, 1))

        Y_train = np.transpose(Y_train, (0, 2, 1))
        Y_val = np.transpose(Y_val, (0, 2, 1))
        Y_test = np.transpose(Y_test, (0, 2, 1))

        return X_train, Y_train, X_val, Y_val, X_test, Y_test

    except FileNotFoundError:
        print(f"[ERROR] Missing .npy files for SNR={snr_db} dB")
        return None


# ==============================
# Training & Evaluation
# ==============================
def train_and_evaluate(model_name, snr_db,
                       X_train, Y_train, X_val, Y_val, X_test, Y_test,
                       run_type):
    """
    Train model, save best checkpoint, evaluate on test set.
    """

    print(f"\n========== TRAINING {model_name} @ {snr_db} dB ==========")

    model_builder = MODEL_MAPPING.get(model_name)
    if model_builder is None:
        raise ValueError(f"Model '{model_name}' not found in MODEL_MAPPING.")

    model = model_builder()
    model.compile(optimizer=OPTIMIZER, loss=LOSS_FUNCTION)

    # Checkpoint directory
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
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[ckpt_callback],
        verbose=1
    )

    # Load best weights
    print(f"[INFO] Loading best weights: {ckpt_path}")
    try:
        model.load_weights(ckpt_path)
    except Exception as e:
        print(f"[WARNING] Could not load best weights: {e}")

    # -----------------------
    # Evaluation
    # -----------------------
    Y_pred = model.predict(X_test, batch_size=BATCH_SIZE)

    rmse_list = []
    snr_list = []

    for i in range(len(Y_pred)):
        rmse_list.append(calculate_rmse(Y_test[i], Y_pred[i]))
        snr_list.append(calculate_snr(Y_test[i], Y_pred[i]))

    avg_rmse = np.mean(rmse_list)
    avg_snr = np.mean(snr_list)

    print(f"\n[RESULT] {model_name} @ {snr_db} dB")
    print(f"Avg RMSE: {avg_rmse:.4f}")
    print(f"Avg SNR : {avg_snr:.2f} dB")

    return avg_rmse, avg_snr, history, Y_pred


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

        X_train, Y_train, X_val, Y_val, X_test, Y_test = data

        for model_label, model_name in model_dict.items():

            avg_rmse, avg_snr, history, Y_pred = train_and_evaluate(
                model_name,
                snr_db,
                X_train, Y_train, X_val, Y_val, X_test, Y_test,
                run_type
            )

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

                plot_ecg_comparison(
                    x_noisy=X_test[0],
                    x_clean=Y_test[0],
                    x_denoised_list=[("DW-CNN", Y_pred[0])],
                    noise_type="Combined",
                    snr_db=snr_db,
                    output_path=out_dir
                )

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
    print("Uncomment run_experiments() to begin.")

    # Example:
    # run_experiments(ABLATION_MODELS, run_type="ablation")
    # run_experiments(COMPARATIVE_MODELS, run_type="comparative")