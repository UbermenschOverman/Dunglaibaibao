DW_CNN_ECG_Denoising/
├── data/
│   ├── raw/
│   │   ├── mitdb/              # Dữ liệu ECG sạch MIT-BIH Arrhythmia Database
│   │   └── nstdb/              # Dữ liệu nhiễu MIT-BIH Noise Stress Test Database (BW, MA, EM)
│   └── processed/
│       ├── noisy_0dB/
│       ├── noisy_1.25dB/
│       └── noisy_5dB/
├── src/
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── data_loader.py      # Tải, phân đoạn, tạo tín hiệu nhiễu (dựa trên công thức s_out = s_clean + a*noise + b ), chia tập train/val/test 
│   ├── models/
│   │   ├── __init__.py
│   │   ├── layers/
│   │   │   ├── __init__.py
│   │   │   ├── dwt_layer.py    # Triển khai DWT-based pooling và IDWT-based up-sampling layer (sử dụng Haar wavelet [cite: 884])
│   │   │   └── wavelet_utils.py # Hàm Haar Wavelet filter f_L, f_H [cite: 241, 242]
│   │   ├── dw_cnn.py           # Định nghĩa kiến trúc mạng DW-CNN (Encoder-Decoder, Skip Connections) [cite: 456, 465, 303]
│   │   ├── model_net1.py       # Mô hình so sánh: CNN với Max Pooling 
│   │   ├── dnn_dan.py          # Mô hình so sánh: DNN-based DAN (reproduced) 
│   │   └── fcn.py              # Mô hình so sánh: Full Convolution Network (reproduced) 
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── metrics.py          # Tính toán RMSE [cite: 846] và SNR [cite: 847]
│   │   └── plotting.py         # Hàm vẽ biểu đồ ECG (ví dụ: Hình 7-9 [cite: 909, 911, 913])
│   ├── main.py                 # File điều phối chính: Tải dữ liệu, xây dựng mô hình, đào tạo, và đánh giá
│   └── config.py               # File chứa các tham số (Hyperparameters)
├── experiments/
│   ├── ablation_bw/            # Thí nghiệm loại trừ (Ablation Experiments) với nhiễu BW (0dB, 1.25dB, 5dB) [cite: 864]
│   ├── ablation_em/
│   ├── ablation_ma/
│   └── comparison/             # Thí nghiệm so sánh (Comparative Experiments) với DAN, FCN [cite: 895]
├── checkpoints/                # Nơi lưu trữ mô hình tốt nhất (best models)
├── results/                    # Nơi lưu trữ kết quả đánh giá (RMSE, SNR) và các hình ảnh đầu ra
├── requirements.txt            # Danh sách các thư viện cần thiết (ví dụ: Python, PyWavelets, TensorFlow/PyTorch, WFDB)
└── README.md                   # Mô tả dự án, hướng dẫn cài đặt và chạy