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
        self.NOISE_TYPES = ['bw', 'em', 'ma'] # Baseline Drift (BW), Electrode Motion (EM), Muscle Artifact (MA) 

    def _calculate_gain(self, s_clean, noise, snr_db):
        """
        Tính toán độ lợi (a) cho nhiễu để đạt mức SNR mong muốn.

        Công thức: s_out = s_clean + a * noise + b (trong đó b=0) [cite: 106]
        SNR (dB) = 10 * log10 (Power_signal / Power_noise)
        Power_noise = a^2 * Power_raw_noise
        Power_signal / (a^2 * Power_raw_noise) = 10^(SNR/10)
        a^2 = Power_signal / (Power_raw_noise * 10^(SNR/10))
        a = sqrt(Power_signal / (Power_raw_noise * 10^(SNR/10)))

        :param s_clean: Tín hiệu ECG sạch (numpy array)
        :param noise: Tín hiệu nhiễu thô (numpy array)
        :param snr_db: Mức SNR mong muốn (dB)
        :return: Độ lợi (a)
        """
        Ps = np.sum(s_clean ** 2)
        Pn_raw = np.sum(noise ** 2)
        
        # Chuyển SNR từ dB sang tỷ lệ tuyến tính
        snr_linear = 10**(snr_db / 10.0)
        
        # Đảm bảo mẫu không bị chia cho 0
        if Pn_raw == 0:
            return 0.0
        
        # Tính độ lợi a
        a = np.sqrt(Ps / (Pn_raw * snr_linear))
        return a

    def _read_ecg_signal(self, db_path, record_name):
        """Đọc tín hiệu ECG từ file .dat và trích xuất Lead MLII."""
        # wfdb.rdrecord chỉ cần tên bản ghi, nó tự tìm kiếm các file liên quan (.dat, .hea)
        try:
            record = wfdb.rdrecord(os.path.join(db_path, record_name), 
                                   pb_dir=os.path.basename(db_path))
            
            # Tìm chỉ mục của Lead MLII (Lead II) 
            if self.LEAD in record.sig_name:
                lead_index = record.sig_name.index(self.LEAD)
                # Lấy tín hiệu, chuyển đổi thành float và loại bỏ NaNs nếu có
                signal = record.p_signal[:, lead_index].astype(np.float32)
                return signal[~np.isnan(signal)]
            else:
                print(f"Cảnh báo: Không tìm thấy Lead {self.LEAD} trong bản ghi {record_name}.")
                return np.array([])
        except Exception as e:
            print(f"Lỗi khi đọc bản ghi {record_name} từ {db_path}: {e}")
            return np.array([])
    
    def _segment_signal(self, signal):
        """
        Phân đoạn tín hiệu thành các mẫu có độ dài cố định.

        :param signal: Tín hiệu (numpy array)
        :return: Danh sách các mẫu (numpy array)
        """
        samples = []
        # Phân đoạn thành các mẫu 4096 điểm lấy mẫu 
        for i in range(0, len(signal) - self.SAMPLE_LENGTH + 1, self.SAMPLE_LENGTH):
            samples.append(signal[i:i + self.SAMPLE_LENGTH])
        return np.array(samples)

    def load_clean_data(self):
        """
        Tải và phân đoạn tất cả các bản ghi ECG sạch từ MITDB.

        :return: Danh sách các mẫu ECG sạch (numpy array)
        """
        print("--- 1. Tải và phân đoạn dữ liệu ECG sạch ---")
        clean_samples = []
        for rec_name in self.CLEAN_RECORDS:
            clean_signal = self._read_ecg_signal(self.MITDB_PATH, rec_name)
            if clean_signal.size > 0:
                samples = self._segment_signal(clean_signal)
                clean_samples.append(samples)
                print(f"  Đã tải bản ghi {rec_name}: {len(samples)} mẫu.")
        
        return np.concatenate(clean_samples, axis=0) if clean_samples else np.array([])

    def load_noise_data(self):
        """
        Tải và phân đoạn tất cả các loại nhiễu từ NSTDB.

        :return: Dictionary chứa các mẫu nhiễu thô cho từng loại nhiễu
        """
        print("--- 2. Tải và phân đoạn dữ liệu nhiễu (NSTDB) ---")
        noise_samples = {}
        for noise_type in self.NOISE_TYPES:
            # NSTDB có 3 bản ghi nhiễu là 'bw', 'em', 'ma'
            noise_signal = self._read_ecg_signal(self.NSTDB_PATH, noise_type)
            if noise_signal.size > 0:
                samples = self._segment_signal(noise_signal)
                noise_samples[noise_type] = samples
                print(f"  Đã tải nhiễu {noise_type}: {len(samples)} mẫu.")
        return noise_samples

    def generate_noisy_data(self, clean_samples, noise_samples):
        """
        Tổng hợp các mẫu ECG nhiễu bằng cách thêm nhiễu vào tín hiệu sạch.

        :param clean_samples: Các mẫu ECG sạch (N, 4096)
        :param noise_samples: Dictionary các mẫu nhiễu thô
        :return: Dictionary các tập dữ liệu nhiễu (SNR: (X_noisy, Y_clean))
        """
        print("--- 3. Tổng hợp tín hiệu nhiễu theo mức SNR ---")
        full_dataset = {}

        # Số lượng mẫu sạch và nhiễu
        N_clean = len(clean_samples)
        
        for snr_db in self.SNR_LEVELS:
            snr_datasets = []
            output_dir = os.path.join(self.PROCESSED_PATH, f"noisy_{snr_db}dB")
            os.makedirs(output_dir, exist_ok=True)
            
            for noise_type, raw_noise_samples in noise_samples.items():
                
                N_noise = len(raw_noise_samples)
                # Lặp lại/cắt nhiễu để khớp với số lượng mẫu sạch
                if N_noise < N_clean:
                    # Lặp lại mẫu nhiễu
                    reps = (N_clean // N_noise) + 1
                    noise_cycle = np.tile(raw_noise_samples, (reps, 1))
                    noise_matched = noise_cycle[:N_clean]
                else:
                    # Cắt mẫu nhiễu
                    noise_matched = raw_noise_samples[:N_clean]
                
                # Tính độ lợi 'a' cho từng cặp mẫu (có thể đơn giản hóa bằng cách tính Ps và Pn_raw trung bình)
                # Tuy nhiên, ta tính một độ lợi trung bình cho toàn bộ tập để đơn giản code
                s_clean_flat = clean_samples.flatten()
                noise_flat = noise_matched.flatten()
                
                # Tính toán độ lợi 'a' (gain)
                gain = self._calculate_gain(s_clean_flat, noise_flat, snr_db)
                
                # Tổng hợp tín hiệu nhiễu: s_out = s_clean + a * noise + b (b=0) [cite: 106]
                X_noisy = clean_samples + gain * noise_matched 
                Y_clean = clean_samples
                
                snr_datasets.append((X_noisy, Y_clean, noise_type))
                print(f"  Tạo {N_clean} mẫu nhiễu {noise_type} ở SNR {snr_db} dB (gain={gain:.4f}).")

            # Ghép tất cả các loại nhiễu lại thành một tập dữ liệu SNR
            # X_noisy_combined là tập hợp tất cả (BW, EM, MA) ở mức SNR này
            X_noisy_combined = np.concatenate([ds[0] for ds in snr_datasets], axis=0)
            Y_clean_combined = np.concatenate([ds[1] for ds in snr_datasets], axis=0)

            full_dataset[snr_db] = (X_noisy_combined, Y_clean_combined)
            print(f"-> Tổng cộng {len(X_noisy_combined)} mẫu ở SNR {snr_db} dB.")

        return full_dataset

    def split_data(self, X, Y):
        """
        Chia tập dữ liệu thành Training (80%), Validation (10%), Test (10%).

        :param X: Mẫu nhiễu
        :param Y: Mẫu sạch tương ứng
        :return: X_train, X_val, X_test, Y_train, Y_val, Y_test
        """
        # Chia 80% Train, 20% còn lại (Val + Test) 
        X_train, X_temp, Y_train, Y_temp = train_test_split(
            X, Y, test_size=0.2, random_state=42
        )
        
        # Chia 20% còn lại thành 10% Val và 10% Test (tức là 50% của 20%) 
        X_val, X_test, Y_val, Y_test = train_test_split(
            X_temp, Y_temp, test_size=0.5, random_state=42
        )
        
        print(f"  Phân chia dữ liệu: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        return X_train, X_val, X_test, Y_train, Y_val, Y_test

    def run_preprocessing_and_save(self):
        """
        Chạy toàn bộ quá trình tiền xử lý và lưu trữ các tập tin đã chia.
        """
        # Đảm bảo các thư mục dữ liệu thô tồn tại (nơi người dùng sẽ tải MITDB/NSTDB)
        os.makedirs(self.MITDB_PATH, exist_ok=True)
        os.makedirs(self.NSTDB_PATH, exist_ok=True)
        
        # 1. Tải và phân đoạn dữ liệu sạch
        clean_samples = self.load_clean_data()
        if len(clean_samples) == 0:
            print("LỖI: Không tìm thấy dữ liệu sạch. Vui lòng tải MITDB vào thư mục raw/MITDB.")
            return

        # 2. Tải và phân đoạn dữ liệu nhiễu
        noise_samples = self.load_noise_data()
        if not noise_samples:
            print("LỖI: Không tìm thấy dữ liệu nhiễu. Vui lòng tải NSTDB vào thư mục raw/NSTDB.")
            return
            
        # 3. Tổng hợp tín hiệu nhiễu
        full_datasets = self.generate_noisy_data(clean_samples, noise_samples)

        # 4. Chia tập dữ liệu và lưu
        print("--- 4. Chia và lưu các tập dữ liệu ---")
        for snr_db, (X, Y) in full_datasets.items():
            
            # Đảm bảo X, Y có hình dạng 1x4096 (channel first) cho CNN 1D
            X = X[:, np.newaxis, :]
            Y = Y[:, np.newaxis, :]
            
            X_train, X_val, X_test, Y_train, Y_val, Y_test = self.split_data(X, Y)
            
            output_dir = os.path.join(self.PROCESSED_PATH, f"noisy_{snr_db}dB")
            
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

# Ví dụ chạy (Bỏ comment nếu muốn kiểm tra):
# if __name__ == '__main__':
#     # Đảm bảo bạn đã tải dữ liệu MITDB và NSTDB vào các thư mục tương ứng 
#     # trước khi chạy hàm này, ví dụ:
#     # data/raw/MITDB/100.dat, data/raw/NSTDB/bw.dat
#     
#     # Bạn có thể cần cài đặt: pip install wfdb numpy scikit-learn
#     loader = ECGDataLoader()
#     loader.run_preprocessing_and_save()