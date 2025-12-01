# src/config.py

import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

# --- THAM SỐ DỮ LIỆU ---

# 10 bản ghi ECG sạch từ MITDB được sử dụng
CLEAN_RECORDS = [
    '103', '105', '111', '116', '122', '205', '213', '219', '223', '230'
]
# Kênh được chọn
LEAD = 'MLII'
# Tần số lấy mẫu (sampling frequency)
FS = 360 
# [cite_start]Độ dài mẫu (sampling points) [cite: 191, 182]
SAMPLE_LENGTH = 4096 
# Các loại nhiễu thực tế (Baseline Drift, Electrode Motion, Muscle Artifact)
NOISE_TYPES = ['bw', 'em', 'ma']
# [cite_start]Các mức SNR được tổng hợp [cite: 181]
SNR_LEVELS = [0, 1.25, 5]

# --- CẤU HÌNH ĐƯỜNG DẪN ---

# Thư mục gốc dự án
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Thư mục dữ liệu
DATA_ROOT = os.path.join(ROOT_DIR, "data")
# Thư mục chứa dữ liệu đã được xử lý
PROCESSED_DATA_PATH = os.path.join(DATA_ROOT, "processed")

# --- THAM SỐ ĐÀO TẠO ---

# [cite_start]Hàm mất mát (Loss function) [cite: 470]
LOSS_FUNCTION = MeanSquaredError()
# [cite_start]Bộ tối ưu hóa (Optimizer) [cite: 813]
OPTIMIZER = Adam()
# [cite_start]Kích thước Batch [cite: 817]
BATCH_SIZE = 512
# Số Epochs (Đặt giá trị tham khảo, có thể cần điều chỉnh)
EPOCHS = 100 
# Tỷ lệ chia tập dữ liệu (Được cố định trong data_loader, 80/10/10)
TRAIN_SIZE = 0.8
VAL_SIZE = 0.1
TEST_SIZE = 0.1

# --- CẤU HÌNH MÔ HÌNH ---

# Kích thước đầu vào: (Channels, Length)
INPUT_SHAPE = (1, SAMPLE_LENGTH)
# [cite_start]Kích thước Kernel cho các lớp Conv1D [cite: 313]
CONV_KERNEL_SIZE = 3
# [cite_start]Kích thước Stride cho các lớp Conv1D [cite: 314]
CONV_STRIDE = 1
# [cite_start]Wavelet được sử dụng cho DWT/IDWT [cite: 887]
WAVELET_TYPE = 'haar'

# --- CẤU HÌNH THÍ NGHIỆM ---

# Các mô hình tham gia so sánh
COMPARATIVE_MODELS = {
    "DW-CNN": "DW_CNN",
    "DAN": "DNN_DAN",
    "FCN": "FCN",
}
# Các mô hình tham gia thí nghiệm loại trừ (Ablation)
ABLATION_MODELS = {
    "DW-CNN (Haar)": "DW_CNN",
    "ModelNet1 (Max Pooling)": "ModelNet1",
    # Có thể thêm các mô hình DW-CNN với wavelet khác ở đây
    # Có thể thêm các mô hình DW-CNN với wavelet khác ở đây
}

ALL_MODELS_FOR_EVAL = ["DW_CNN", "ModelNet1", "DNN_DAN", "FCN"]
NOISE_TYPES = ["BW", "EM", "MA"]

# Thư mục lưu trữ kết quả và checkpoints
CHECKPOINT_PATH = os.path.join(ROOT_DIR, "checkpoints")
RESULTS_PATH = os.path.join(ROOT_DIR, "results")