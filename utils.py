from typing import Union
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

import h5py
import torch
from torch.utils.data import Dataset
from typing import Union

class HDF5MapStyleDataset(Dataset):
    """Enhanced sliding window HDF5 dataset"""

    def __init__(
        self,
        file_path: str,
        window_size: int = 2,
        move_size: int = 1,
        device: Union[str, torch.device] = "cuda",
    ):
        self.file_path = file_path
        self.window_size = window_size
        self.move_size = move_size
        self.device = torch.device(device) if isinstance(device, str) else device

        # 預先計算片段數量
        self.num_segments = None  # 這個會在 __getitem__ 中計算
        self.total_samples = None  # 這個會在 __getitem__ 中計算

        print(f"Dataset loaded from {file_path}.")
    
    def __len__(self):
        # 在第一次調用時計算樣本數量
        if self.num_segments is None:
            self._load_h5_file()
        return self.total_samples

    def __getitem__(self, idx):
        # 每次讀取時打開文件，進行讀取
        with h5py.File(self.file_path, "r") as file:
            # 重新計算切片位置
            segment_idx = idx % self.num_segments
            sample_idx = idx // self.num_segments
            start_idx = segment_idx * self.move_size
            end_idx = start_idx + self.window_size

            # 快速讀取數據
            pressure_field = file["pressure"][sample_idx, start_idx:end_idx]
            density_field = file["density"][sample_idx][None]
            sound_speed_field = file["sound_speed"][sample_idx][None]

            # 訓練輸入資料與目標資料
            invar = torch.cat(
                [
                    torch.from_numpy(pressure_field[:1, :, :]),
                    torch.from_numpy(density_field),
                    torch.from_numpy(sound_speed_field),
                ]
            )
            outvar = torch.from_numpy(pressure_field[1:2, :, :])
            noise = (torch.rand_like(invar[0]) - 0.5) * 0.2  # 範圍為 [-0.01, 0.01]

            invar[0] += noise
            # 將資料轉換為設備指定格式
            if self.device.type == "cuda":
                invar = invar.cuda()
                outvar = outvar.cuda()

            return invar.to(torch.float32), outvar.to(torch.float32)

    def _load_h5_file(self):
        """用來計算資料集大小和切片數量的輔助方法"""
        with h5py.File(self.file_path, "r") as file:
            # 計算片段數量
            pressure_field_shape = file["pressure"].shape
            self.num_time_steps = pressure_field_shape[1]
            self.num_segments = (self.num_time_steps - self.window_size) // self.move_size + 1
            self.total_samples = len(file["pressure"]) * self.num_segments

    def __del__(self):
        # 如果是開啟的話，關閉 HDF5 文件
        pass
