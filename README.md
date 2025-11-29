Phải chạy conda tạo env mới:
conda create -n tf python=3.10
conda activate tf (and conda deactivate after using)
pip install tensorflow==2.13 numpy==1.26

# DW-CNN ECG Denoising

A deep learning framework for ECG denoising using **Discrete Wavelet Transform (DWT)**-based pooling and upsampling. This project implements the DW-CNN architecture, along with several baseline models, and provides a full pipeline from data preparation to experimentation and evaluation.

---

## 🚀 Key Features

* **DW-CNN architecture** (Encoder–Decoder with skip connections)
* **Wavelet-based pooling/upsampling** using Haar DWT
* **Baseline models for comparison:** CNN, DAN, FCN
* **Automatic dataset generation** from MIT-BIH Arrhythmia & NSTDB noise
* **Support multiple noise levels** (0 dB, 1.25 dB, 5 dB)
* **Ablation and comparative experiments** integrated
* **Metrics:** RMSE, SNR

---

## 📦 Installation

### 1. Create Conda Environment

```bash
conda create -n tf python=3.10
conda activate tf
```

### 2. Install Dependencies

```bash
pip install tensorflow==2.13 numpy==1.26
```

Or install from requirement file:

```bash
pip install -r requirements.txt
```

---

## 📁 Project Structure

```
DW_CNN_ECG_Denoising/
├── data/
│   ├── raw/
│   │   ├── mitdb/              # MIT-BIH Arrhythmia Database (clean ECG)
│   │   └── nstdb/              # MIT-BIH Noise Stress Test Database (BW, MA, EM)
│   └── processed/
│       ├── noisy_0dB/
│       ├── noisy_1.25dB/
│       └── noisy_5dB/
├── src/
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── data_loader.py      # Load, segment, mix noise, create train/val/test
│   ├── models/
│   │   ├── __init__.py
│   │   ├── layers/
│   │   │   ├── __init__.py
│   │   │   ├── dwt_layer.py    # DWT pooling + IDWT upsampling (Haar)
│   │   │   └── wavelet_utils.py # Haar filters f_L, f_H
│   │   ├── dw_cnn.py           # DW-CNN architecture
│   │   ├── model_net1.py       # Baseline: CNN + MaxPool
│   │   ├── dnn_dan.py          # Baseline: DAN
│   │   └── fcn.py              # Baseline: FCN
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── metrics.py          # RMSE, SNR
│   │   └── plotting.py         # ECG plotting helpers
│   ├── main.py                 # Training/testing pipeline
│   └── config.py               # Hyperparameters
├── experiments/
│   ├── ablation_bw/            # Ablation for BW noise
│   ├── ablation_em/
│   ├── ablation_ma/
│   └── comparison/             # DAN/FCN comparison experiments
├── checkpoints/                # Saved models
├── results/                    # Evaluation logs, plots
├── requirements.txt
└── README.md
```

---

## ⚙️ Usage

### 1. Prepare Data

Place MIT-BIH datasets:

```
data/raw/mitdb/
data/raw/nstdb/
```

The preprocessing script will automatically:

* Segment ECG
* Mix in BW/MA/EM noise using `s_out = s_clean + a*noise + b`
* Create train/val/test splits

### 2. Train Model

```bash
python src/main.py
```

The training procedure will:

* Load processed data
* Build the DW-CNN model
* Train and evaluate
* Save best checkpoints and results

---

## 📊 Experiments

### Ablation Experiments

Located in:

```
experiments/ablation_bw/
experiments/ablation_ma/
experiments/ablation_em/
```

Each folder contains:

* Configs for noise levels
* Results (RMSE, SNR)
* Model variants without wavelet components

### Comparative Experiments

Located in:

```
experiments/comparison/
```

Includes evaluations of:

* DAN
* FCN
* CNN MaxPool

---

## 📈 Evaluation Metrics

Implemented in `utils/metrics.py`:

* **RMSE** — reconstruction accuracy
* **SNR** — denoising performance

Plots (before/after denoising) are generated via `utils/plotting.py`.

---

## 📝 Citation

If you use this repository, please cite the relevant wavelet, CNN, and DAN/FCN references as indicated in the source code headers.

---

## 📄 License

MIT License. See `LICENSE` if included.

---

## 🙌 Acknowledgements

Datasets:

* MIT-BIH Arrhythmia Database
* MIT-BIH Noise Stress Test Database

Wavelet foundations and network architecture references as noted in code comments.
