import os
import h5py
import torch
from torch.utils.data import Dataset
from typing import Union, List
import random

class HDF5MapStyleDataset(Dataset):
    """Enhanced sliding window HDF5 dataset for multiple files"""

    def __init__(
        self,
        folder_path: str,
        device: Union[str, torch.device] = "cuda",
    ):
        self.folder_path = folder_path
        self.device = torch.device(device) if isinstance(device, str) else device

        # 讀取所有 HDF5 檔案並計算樣本數
        self.h5_files = sorted([
            os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".h5")
        ])
        
        self.sample_offsets = []  # 儲存每個檔案的累積索引範圍
        self.total_samples = 0

        for file_path in self.h5_files:
            # 從檔名解析時間步數
            T = self._extract_time_steps(file_path)
            num_samples = T - 1  # 由於 stride=1，每個檔案的樣本數為 T-1
            self.sample_offsets.append((self.total_samples, file_path))
            self.total_samples += num_samples

        print(f"Dataset loaded from {folder_path}. Found {len(self.h5_files)} files, total {self.total_samples} samples.")

    def _extract_time_steps(self, file_path: str) -> int:
        """從檔名提取時間步數 T"""
        filename = os.path.basename(file_path)
        parts = filename.split("_")
        try:
            T = int(parts[-1].split(".")[0])  # 取最後的數字部分
            return T
        except ValueError:
            raise ValueError(f"Cannot extract time steps from filename: {filename}")

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        """根據全局索引讀取對應的 HDF5 檔案與索引內的數據"""
        # 找到對應的 HDF5 檔案
        for i in range(len(self.sample_offsets) - 1, -1, -1):  # 反向遍歷，提高查找效率
            start_idx, file_path = self.sample_offsets[i]
            if idx >= start_idx:
                local_idx = idx - start_idx  # 計算該文件內的索引
                break  # 找到對應檔案後跳出迴圈

        # 讀取 HDF5 檔案
        with h5py.File(file_path, "r") as f:
            # 讀取數據，留在 CPU
            pressure_t = torch.tensor(f["pressure"][local_idx], dtype=torch.float32).unsqueeze(0)  # (1, 64, 64)
            pressure_t1 = torch.tensor(f["pressure"][local_idx + 1], dtype=torch.float32).unsqueeze(0)  # (1, 64, 64)

            density = torch.tensor(f["density"][:], dtype=torch.float32).unsqueeze(0)  # (1, 64, 64)
            sound_speed = torch.tensor(f["sound_speed"][:], dtype=torch.float32).unsqueeze(0)  # (1, 64, 64)

            # 堆疊成 (3, 64, 64)
            X = torch.cat([pressure_t, density, sound_speed], dim=0)  # (3, 64, 64)
            Y = pressure_t1  # (1, 64, 64)
            # # 隨機旋轉 0, 90, 180, 270 度
            # k = random.choice([0, 1, 2, 3])  # 旋轉次數 (逆時針 90 度為一次)
            # if k > 0:
            #     X = torch.rot90(X, k, [1, 2])  # 對 (C, H, W) 格式的 H, W 軸旋轉
            #     Y = torch.rot90(Y, k, [1, 2])
                        # 在 X 的第0個特徵中 (即 pressure_t) 添加雜訊
            # 計算 X[0] 的最大值
            max_value = X[0].max()

            # 設定噪聲範圍為最大值的 *0.1
            noise = (torch.rand_like(X[0]) - 0.5) * max_value * 0.1  # 範圍為 [-max_value*0.1, max_value*0.1]

            X[0] += noise

        return X, Y 


