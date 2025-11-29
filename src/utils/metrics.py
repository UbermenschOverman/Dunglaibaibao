# src/utils/metrics.py

import numpy as np

def calculate_rmse(x_clean, x_denoised):
    """
    Tính Root Mean Square Error (RMSE) giữa tín hiệu sạch và tín hiệu đã khử nhiễu.
    RMSE càng nhỏ, hiệu suất khử nhiễu càng tốt.

    Công thức: RMSE = sqrt( (1/N) * sum((x_clean - x_denoised)^2) ) 
    
    :param x_clean: Tín hiệu ECG sạch (numpy array)
    :param x_denoised: Tín hiệu ECG đã khử nhiễu (numpy array)
    :return: Giá trị RMSE
    """
    # Đảm bảo đầu vào là numpy array và có cùng hình dạng
    if x_clean.shape != x_denoised.shape:
        raise ValueError("Tín hiệu sạch và tín hiệu đã khử nhiễu phải có cùng hình dạng.")

    # Tính hiệu số (tín hiệu nhiễu còn lại)
    error = x_clean - x_denoised
    
    # Tính Mean Square Error (MSE)
    mse = np.mean(error ** 2)
    
    # Tính Root Mean Square Error (RMSE) 
    rmse = np.sqrt(mse)
    
    return rmse

def calculate_snr(x_clean, x_denoised):
    """
    Tính Signal-to-Noise Ratio (SNR) giữa tín hiệu sạch và tín hiệu đã khử nhiễu.
    SNR càng lớn, hiệu suất khử nhiễu càng tốt.

    Công thức: SNR = 10 * log10 (Power_signal / Power_noise) 
    Trong đó, tín hiệu nhiễu được coi là sai số (error = x_clean - x_denoised).
    
    :param x_clean: Tín hiệu ECG sạch (numpy array)
    :param x_denoised: Tín hiệu ECG đã khử nhiễu (numpy array)
    :return: Giá trị SNR (dB)
    """
    # Đảm bảo đầu vào là numpy array và có cùng hình dạng
    if x_clean.shape != x_denoised.shape:
        raise ValueError("Tín hiệu sạch và tín hiệu đã khử nhiễu phải có cùng hình dạng.")

    # Tính công suất tín hiệu sạch (Power_signal)
    power_signal = np.sum(x_clean ** 2)
    
    # Tính công suất tín hiệu nhiễu còn lại (Power_noise = công suất của sai số)
    noise_error = x_clean - x_denoised
    power_noise = np.sum(noise_error ** 2)
    
    # Trường hợp không có nhiễu còn lại (lý tưởng) hoặc tín hiệu quá nhỏ
    if power_noise == 0:
        return np.inf  # Vô cùng
    
    # Tính SNR theo công thức 10 * log10 (...) 
    snr = 10 * np.log10(power_signal / power_noise)
    
    return snr

# Có thể thêm hàm tính SNR ban đầu của mẫu nhiễu thô nếu cần
def calculate_initial_snr(x_clean, x_noisy):
    """Tính SNR ban đầu của tín hiệu nhiễu (để kiểm tra quá trình tạo dữ liệu)."""
    # Noise = x_noisy - x_clean
    power_signal = np.sum(x_clean ** 2)
    power_noise = np.sum((x_noisy - x_clean) ** 2)
    
    if power_noise == 0:
        return np.inf
    return 10 * np.log10(power_signal / power_noise)