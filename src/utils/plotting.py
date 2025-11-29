# src/utils/plotting.py

import matplotlib.pyplot as plt
import numpy as np
import os
from .metrics import calculate_rmse, calculate_snr

def plot_ecg_comparison(x_noisy, x_clean, x_denoised_list, noise_type, snr_db, output_path):
    """
    Vẽ biểu đồ so sánh tín hiệu ECG nhiễu, sạch, và đã khử nhiễu (từ các mô hình).

    Tương tự cấu trúc Hình 7, 8, 9 trong bài báo:
    (a) Tín hiệu nhiễu (ECG signal with Noise Type)
    (b) Tín hiệu sạch (Clean ECG signal)
    (c) Tín hiệu đã khử nhiễu (Model 1 predicted ECG signal)
    (d) Tín hiệu đã khử nhiễu (Model 2 predicted ECG signal)
    (e) Tín hiệu đã khử nhiễu (DW-CNN predicted ECG signal)

    :param x_noisy: Mẫu ECG nhiễu (numpy array, 1D hoặc 1xN)
    :param x_clean: Mẫu ECG sạch tương ứng (numpy array, 1D hoặc 1xN)
    :param x_denoised_list: Danh sách các cặp (tên mô hình, mẫu đã khử nhiễu)
    :param noise_type: Loại nhiễu (BW, EM, MA)
    :param snr_db: Mức SNR (dB)
    :param output_path: Đường dẫn để lưu hình ảnh
    """
    # Chuyển đổi về 1D nếu cần (loại bỏ channel dimension 1xN)
    x_noisy = x_noisy.flatten()
    x_clean = x_clean.flatten()
    
    # Tổng số biểu đồ cần vẽ: 2 (noisy, clean) + số lượng mô hình
    num_plots = 2 + len(x_denoised_list)

    # Thiết lập Figure và Subplots (5 subplots cho so sánh 3 mô hình)
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 2 * num_plots))
    
    # Ensure axes is always a list/array for consistent indexing
    if num_plots == 1:
        axes = [axes]
    
    # 1. Tín hiệu nhiễu (a)
    axes[0].plot(x_noisy, color='r', linewidth=1)
    axes[0].set_title(f"(a) ECG signal with {noise_type} ({snr_db} dB)", fontsize=10)
    axes[0].set_ylabel("Amplitude")
    
    # 2. Tín hiệu sạch (b)
    axes[1].plot(x_clean, color='g', linewidth=1)
    axes[1].set_title("(b) Clean ECG signal", fontsize=10)
    axes[1].set_ylabel("Amplitude")
    
    # 3. Các tín hiệu đã khử nhiễu (c, d, e, ...)
    labels = ['(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)']  # Extended for more models
    
    for i, (model_name, x_denoised) in enumerate(x_denoised_list):
        if i < len(axes) - 2: # Bắt đầu từ axes[2]
            x_denoised = x_denoised.flatten()
            rmse = calculate_rmse(x_clean, x_denoised)
            snr = calculate_snr(x_clean, x_denoised)

            # Use label if available, otherwise use index
            label = labels[i] if i < len(labels) else f"({chr(ord('c') + i)})"
            axes[i + 2].plot(x_denoised, color='b', linewidth=1)
            axes[i + 2].set_title(
                f"{label} {model_name} predicted ECG signal (RMSE: {rmse:.4f}, SNR: {snr:.2f} dB)", 
                fontsize=10
            )
            axes[i + 2].set_ylabel("Amplitude")
            axes[i + 2].set_xlabel("Time (Samples)")
        
    # Điều chỉnh layout để tránh chồng lấn
    plt.tight_layout()
    
    # Tạo tên file và lưu
    filename = f"comparison_{noise_type}_{snr_db}dB.png"
    filepath = os.path.join(output_path, filename)
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(filepath)
    plt.close(fig) # Đóng figure để giải phóng bộ nhớ
    print(f"  Đã lưu biểu đồ so sánh tại: {filepath}")