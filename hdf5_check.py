import h5py

# 讀取 HDF5 檔案
file_path = "./pressure_data_3d.h5"
with h5py.File(file_path, 'r') as f:
    # 列出所有資料集名稱
    print("資料集名稱及其形狀：")
    for name in f:
        # 取得資料集的形狀
        dataset = f[name]
        print(f"{name} 的形狀: {dataset.shape}")
