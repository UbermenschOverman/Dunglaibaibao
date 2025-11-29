# src/models/layers/wavelet_utils.py

import numpy as np

# Các bộ lọc Haar Wavelet cơ bản (Haar Wavelet Filter Bank)
# Theo lý thuyết DWT, bộ lọc được chuẩn hóa bằng sqrt(2) và sắp xếp lại 
# để sử dụng trong các thư viện như PyWavelets.
# Tuy nhiên, trong ngữ cảnh của mạng CNN, chúng ta định nghĩa bộ lọc 
# cơ bản để thực hiện Convolutional DWT (CWT).

# Từ Eq (5) [cite: 242] (mặc dù công thức trong bài báo đơn giản hóa)
# Nếu áp dụng trực tiếp như lớp tích chập, ta sử dụng các hệ số sau:

def haar_low_pass_filter():
    """
    Định nghĩa bộ lọc thông thấp f_L (Approximation/cA) cho Haar Wavelet.
    Cơ sở cho tính toán cA.
    """
    # Bộ lọc cơ sở Haar 1D (đã chuẩn hóa)
    # Tương ứng với phép tính trung bình (average)
    return np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=np.float32)

def haar_high_pass_filter():
    """
    Định nghĩa bộ lọc thông cao f_H (Detail/cD) cho Haar Wavelet.
    Cơ sở cho tính toán cD.
    """
    # Bộ lọc cơ sở Haar 1D (đã chuẩn hóa)
    # Tương ứng với phép tính chênh lệch (difference)
    return np.array([1/np.sqrt(2), -1/np.sqrt(2)], dtype=np.float32)

def get_haar_filters():
    """
    Trả về bộ lọc thông thấp (f_L) và thông cao (f_H) của Haar Wavelet.
    """
    f_L = haar_low_pass_filter()
    f_H = haar_high_pass_filter()
    return f_L, f_H

# IDWT filters (Reconstruction Filters)
# Bộ lọc tái tạo g_L, g_H
def haar_reconstruction_low_pass_filter():
    # g_L = f_L
    return haar_low_pass_filter()

def haar_reconstruction_high_pass_filter():
    # g_H = -f_H (Nếu bộ lọc phân tích là trực giao)
    # Với Haar, đơn giản là sự đảo ngược thứ tự và chuẩn hóa
    return np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=np.float32)

def get_haar_reconstruction_filters():
    """
    Trả về bộ lọc tái tạo (Inverse DWT) g_L và g_H.
    """
    # Chú ý: Trong thư viện PyTorch/TensorFlow, cách triển khai DWT/IDWT
    # như một layer có thể sử dụng hàm tích chập chéo (transposed conv) hoặc 
    # các phép toán ma trận/subband. 
    # Ở đây ta giả định các bộ lọc cơ sở cho reconstruction:
    g_L = haar_reconstruction_low_pass_filter()
    g_H = haar_reconstruction_high_pass_filter()
    return g_L, g_H