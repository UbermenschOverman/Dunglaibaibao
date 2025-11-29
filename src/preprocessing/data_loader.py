# src/preprocessing/data_loader.py

import os
import numpy as np
import wfdb
from sklearn.model_selection import train_test_split

class ECGDataLoader:
    """
    Tải, tiền xử lý và tạo tập dữ liệu ECG (sạch và nhiễu) dựa trên các thông số
    của bài báo 'A novel deep wavelet convolutional neural network...'.
    """

    def __init__(self, data_root_path="./data", snr_levels=[0, 1.25, 5]):
        """
        Khởi tạo DataLoader.

        :param data_root_path: Đường dẫn tới thư mục data/
        :param snr_levels: Danh sách các mức SNR cần tổng hợp
        """
        self.DATA_ROOT = data_root_path
        # Sử dụng os.path.join để đảm bảo đường dẫn chính xác trên mọi OS
        self.MITDB_PATH = os.path.join(data_root_path, "raw", "MITDB")
        self.NSTDB_PATH = os.path.join(data_root_path, "raw", "NSTDB")
        self.PROCESSED_PATH = os.path.join(data_root_path, "processed")
        self.SNR_LEVELS = snr_levels
        
        # Các tham số cố định từ bài báo
        self.CLEAN_RECORDS = [
            '103', '105', '111', '116', '122', '205', '213', '219', '223', '230'
        ] # 10 bản ghi ECG sạch 
        self.LEAD = 'MLII' # Kênh II 
        self.SAMPLE_LENGTH = 4096 # Độ dài mẫu (sampling points) 
        self.FS = 360 # Tần số lấy mẫu MITDB 

        # Các loại nhiễu cần tổng hợp (dữ liệu NSTDB)
        self.NOISE_TYPES = ['bw', 'em', 'ma'] 

    def _calculate_gain(self, s_clean, noise, snr_db):
        """
        Tính toán độ lợi (a) cho nhiễu để đạt mức SNR mong muốn.
        """
        Ps = np.sum(s_clean ** 2)
        Pn_raw = np.sum(noise ** 2)
        
        snr_linear = 10**(snr_db / 10.0)
        
        if Pn_raw < 1e-10: 
            return 0.0
        
        a = np.sqrt(Ps / (Pn_raw * snr_linear))
        return a

    def _read_ecg_signal(self, db_path, record_name, is_noise=False):
        """
        Đọc tín hiệu ECG từ file .dat và trích xuất Lead.
        
        :param is_noise: Nếu True, sẽ đọc kênh đầu tiên (vì NSTDB không có MLII).
        """
        try:
            # Sửa lỗi đọc WFDB: Chuyển đến thư mục để đọc file cục bộ
            original_cwd = os.getcwd()
            os.chdir(db_path)
            
            # Đọc bản ghi
            record = wfdb.rdrecord(record_name)
            
            # Quay lại thư mục gốc
            os.chdir(original_cwd)
            
            if is_noise:
                # Dữ liệu nhiễu (NSTDB): Luôn lấy kênh đầu tiên (hoặc kênh duy nhất)
                signal = record.p_signal[:, 0].astype(np.float32)
                return signal[~np.isnan(signal)]
            
            # Xử lý dữ liệu ECG sạch (MITDB)
            if self.LEAD in record.sig_name:
                lead_index = record.sig_name.index(self.LEAD)
                signal = record.p_signal[:, lead_index].astype(np.float32)
                return signal[~np.isnan(signal)]
            else:
                print(f"Cảnh báo: Không tìm thấy Lead {self.LEAD} trong bản ghi {record_name}.")
                return np.array([])
        except Exception as e:
            # In lỗi đọc file để debug dễ hơn
            print(f"LỖI ĐỌC FILE WFDB: Record={record_name}, Path={db_path}. Chi tiết: {e}")
            return np.array([])
    
    def _segment_signal(self, signal):
        """
        Phân đoạn tín hiệu thành các mẫu có độ dài cố định (4096).
        """
        samples = []
        for i in range(0, len(signal) - self.SAMPLE_LENGTH + 1, self.SAMPLE_LENGTH):
            samples.append(signal[i:i + self.SAMPLE_LENGTH])
        return np.array(samples)

    def load_clean_data(self):
        """Tải và phân đoạn ECG sạch từ MITDB."""
        print("--- 1. Tải và phân đoạn dữ liệu ECG sạch (MITDB) ---")
        clean_samples = []
        for rec_name in self.CLEAN_RECORDS:
            # is_noise=False (Tìm Lead MLII)
            clean_signal = self._read_ecg_signal(self.MITDB_PATH, rec_name, is_noise=False)
            if clean_signal.size > 0:
                samples = self._segment_signal(clean_signal)
                clean_samples.append(samples)
                print(f"  Đã tải bản ghi {rec_name}: {len(samples)} mẫu.")
        
        return np.concatenate(clean_samples, axis=0) if clean_samples else np.array([])

    def load_noise_data(self):
        """Tải và phân đoạn dữ liệu nhiễu từ NSTDB."""
        print("--- 2. Tải và phân đoạn dữ liệu nhiễu (NSTDB) ---")
        noise_samples = {}
        for noise_type in self.NOISE_TYPES:
            # SỬA LỖI: is_noise=True để đọc kênh đầu tiên
            noise_signal = self._read_ecg_signal(self.NSTDB_PATH, noise_type, is_noise=True)
            if noise_signal.size > 0:
                samples = self._segment_signal(noise_signal)
                noise_samples[noise_type] = samples
                print(f"  Đã tải nhiễu {noise_type}: {len(samples)} mẫu.")
            # else: không cần in lỗi ở đây, lỗi đã được in trong _read_ecg_signal
        
        return noise_samples

    def generate_noisy_data(self, clean_samples, noise_samples):
        """Tổng hợp các mẫu ECG nhiễu theo mức SNR."""
        print("--- 3. Tổng hợp tín hiệu nhiễu theo mức SNR ---")
        
        # Nếu không có mẫu nhiễu nào, không thể tạo dataset
        if not noise_samples:
            print("LỖI: Không có mẫu nhiễu nào được tải thành công. Không thể tạo dataset.")
            return {}
            
        full_dataset = {}
        N_clean = len(clean_samples)
        
        for snr_db in self.SNR_LEVELS:
            snr_datasets = []
            output_dir = os.path.join(self.PROCESSED_PATH, f"noisy_{snr_db}dB")
            os.makedirs(output_dir, exist_ok=True)
            
            for noise_type, raw_noise_samples in noise_samples.items():
                
                N_noise = len(raw_noise_samples)
                
                if N_noise < N_clean:
                    reps = (N_clean // N_noise) + 1
                    noise_cycle = np.tile(raw_noise_samples, (reps, 1))
                    noise_matched = noise_cycle[:N_clean]
                else:
                    noise_matched = raw_noise_samples[:N_clean]
                
                s_clean_flat = clean_samples.flatten()
                noise_flat = noise_matched.flatten()
                
                gain = self._calculate_gain(s_clean_flat, noise_flat, snr_db)
                
                # Tổng hợp tín hiệu nhiễu
                X_noisy = clean_samples + gain * noise_matched 
                Y_clean = clean_samples
                
                snr_datasets.append((X_noisy, Y_clean, noise_type))
                print(f"  Tạo {N_clean} mẫu nhiễu {noise_type} ở SNR {snr_db} dB (gain={gain:.4f}).")

            X_noisy_combined = np.concatenate([ds[0] for ds in snr_datasets], axis=0)
            Y_clean_combined = np.concatenate([ds[1] for ds in snr_datasets], axis=0)

            full_dataset[snr_db] = (X_noisy_combined, Y_clean_combined)
            print(f"-> Tổng cộng {len(X_noisy_combined)} mẫu ở SNR {snr_db} dB.")

        return full_dataset

    def split_data(self, X, Y):
        """Chia tập dữ liệu thành Training (80%), Validation (10%), Test (10%)."""
        X_train, X_temp, Y_train, Y_temp = train_test_split(
            X, Y, test_size=0.2, random_state=42
        )
        X_val, X_test, Y_val, Y_test = train_test_split(
            X_temp, Y_temp, test_size=0.5, random_state=42
        )
        
        print(f"  Phân chia dữ liệu: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        return X_train, X_val, X_test, Y_train, Y_val, Y_test

    def run_preprocessing_and_save(self):
        """Chạy toàn bộ quá trình tiền xử lý và lưu trữ các tập tin đã chia."""
        os.makedirs(self.MITDB_PATH, exist_ok=True)
        os.makedirs(self.NSTDB_PATH, exist_ok=True)
        os.makedirs(self.PROCESSED_PATH, exist_ok=True)
        
        # 1. Tải và phân đoạn dữ liệu sạch
        clean_samples = self.load_clean_data()
        if len(clean_samples) == 0:
            print("LỖI QUAN TRỌNG: Không thể đọc dữ liệu ECG sạch. Dừng tiền xử lý.")
            return

        # 2. Tải và phân đoạn dữ liệu nhiễu
        noise_samples = self.load_noise_data()
        if not noise_samples:
            print("LỖI QUAN TRỌNG: Không thể đọc dữ liệu nhiễu nào. Dừng tiền xử lý.")
            return
            
        # 3. Tổng hợp tín hiệu nhiễu
        full_datasets = self.generate_noisy_data(clean_samples, noise_samples)
        
        if not full_datasets:
            print("LỖI QUAN TRỌNG: Không thể tạo datasets từ nhiễu và sạch. Dừng tiền xử lý.")
            return

        # 4. Chia tập dữ liệu và lưu
        print("--- 4. Chia và lưu các tập dữ liệu ---")
        for snr_db, (X, Y) in full_datasets.items():
            
            # Thêm chiều kênh (channel first: 1x4096)
            X = X[:, np.newaxis, :]
            Y = Y[:, np.newaxis, :]
            
            X_train, X_val, X_test, Y_train, Y_val, Y_test = self.split_data(X, Y)
            
            output_dir = os.path.join(self.PROCESSED_PATH, f"noisy_{snr_db}dB")
            os.makedirs(output_dir, exist_ok=True)

            # Lưu tập Training
            np.save(os.path.join(output_dir, "X_train.npy"), X_train)
            np.save(os.path.join(output_dir, "Y_train.npy"), Y_train)
            
            # Lưu tập Validation
            np.save(os.path.join(output_dir, "X_val.npy"), X_val)
            np.save(os.path.join(output_dir, "Y_val.npy"), Y_val)

            # Lưu tập Test
            np.save(os.path.join(output_dir, "X_test.npy"), X_test)
            np.save(os.path.join(output_dir, "Y_test.npy"), Y_test)
            
            print(f"Đã lưu các tập tin cho SNR {snr_db} dB tại {output_dir}")