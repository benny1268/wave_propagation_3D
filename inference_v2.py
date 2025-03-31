import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import modulus

def weight_():
    """結合高斯加權和動態權重"""
    # 計算動態權重
    weight = np.ones((64, 64, 64))

    return weight

def slide_and_split(input_data, block_size=64, slide_step=32):
    # 假設 input_data 的形狀是 (1, 256, 256)
    H, W, C = input_data.shape[-3], input_data.shape[-2], input_data.shape[-1]
    
    # 確保切割的塊大小和步長
    assert H >= block_size and W >= block_size
    
    # 計算切割的塊數量
    blocks = []
    for i in range(0, H - block_size + 1, slide_step):
        for j in range(0, W - block_size + 1, slide_step):
            for k in range(0, C - block_size + 1, slide_step):
                block = input_data[:, i:i+block_size, j:j+block_size, k:k+block_size]
                blocks.append(block)
    
    return blocks

def combine(blocks, block_size=64, slide_step=32):
    pred_full = np.zeros((128, 128, 256))
    count_full = np.zeros((128, 128, 256))

    weight = weight_()
    for i in range(0, 128 - block_size + 1, slide_step):
        for j in range(0, 128 - block_size + 1, slide_step):
            for k in range(0, 256 - block_size + 1, slide_step):
                block = blocks.pop(0)
                pred_full[i:i+block_size, j:j+block_size, k:k+block_size] += block[0] * weight
                count_full[i:i+block_size, j:j+block_size, k:k+block_size] += weight

    return pred_full / count_full 


def predict_and_store_with_sliding(model, input_data, total_steps=70, device="cuda"):
    result_matrix = np.zeros((total_steps, 128, 128, 256))
    result_matrix[0] = input_data[0].cpu().numpy()

    block_size = 64  # 分割塊大小
    slide_step = 32  # 滑動步長

    # 進行預測並存儲預測結果
    for step in range(1, total_steps):
        blocks = slide_and_split(input_data, block_size, slide_step)  # 假設 input_data 是 (1, 256, 256)
        preds = []
        
        for idx, block in enumerate(blocks):
            # 檢查 patch 內的初始聲場是否全為 0
            if torch.all(block[0] < 0.001):  
                preds.append(torch.full((1, 64, 64, 64), fill_value=0).numpy())  # 避免全 0
            else:
                # 經過模型
                current_input = block.unsqueeze(0).to(device)  # 增加 batch 維度
                with torch.no_grad():
                    output = model(current_input).squeeze(0)  # 模型預測
                preds.append(output.cpu().numpy())

        result_matrix[step] = combine(preds)
        with h5py.File(file_path, "r") as f:
            pressure_t = torch.tensor(f["pressure"][0][step], dtype=torch.float32, device=device)
            #pressure_t = torch.tensor(result_matrix[step] , dtype=torch.float32, device=device)

        # 更新 input_data 用於下一步
        input_data[0].copy_(pressure_t.to(input_data.device))

    return result_matrix

def input_data(file_path, device = "cuda", idx = 0):
    with h5py.File(file_path, "r") as f:

        total_steps =  f["pressure"][idx].shape[0]

        pressure_t = torch.tensor(f["pressure"][idx][0], dtype=torch.float32, device=device).unsqueeze(0)  # (1, 256, 256)

        density = torch.tensor(f["density"][idx], dtype=torch.float32, device=device).unsqueeze(0)  # (1, 256, 256)
        sound_speed = torch.tensor(f["sound_speed"][idx], dtype=torch.float32, device=device).unsqueeze(0)  # (1, 256, 256)

        invar = torch.cat([pressure_t, density, sound_speed], dim=0)  # (3, 256, 256)

    return invar.to(torch.float32), total_steps

def create_comparison_gif(result_matrix, file_path, output_gif="Local_PINO_result.gif", fps=20, total_steps=170, idx = 0):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].set_title("Ground Truth")
    ax[1].set_title("Local PINO Prediction")
    ax[2].set_title("Difference (Error)")

    # Create initial empty images with 0 data
    img_gt = ax[0].imshow(np.zeros((128, 256)), cmap="viridis", animated=True)
    img_pred = ax[1].imshow(np.zeros((128, 256)), cmap="viridis", animated=True)
    img_diff = ax[2].imshow(np.zeros((128, 256)), cmap="viridis", animated=True)

    # Open the HDF5 file and read ground truth data
    with h5py.File(file_path, "r") as f:
        # Read pressure field for ground truth and other data
        pressure_field_gt = np.array(f['pressure'][idx])  # Read ground truth data for 370 time steps

    # Set up the colorbars for the images
    cbar_gt = plt.colorbar(img_gt, ax=ax[0])
    cbar_pred = plt.colorbar(img_pred, ax=ax[1])
    cbar_diff = plt.colorbar(img_diff, ax=ax[2])

    def update_fig(step):
        # Ground truth and model prediction for the current step
        gt_data = pressure_field_gt[step][:][64][:]
        pred_data = result_matrix[step][:][64][:]
        
        # Calculate the difference (error)
        diff_data = np.abs(gt_data - pred_data)

        # Update the images with new data
        img_gt.set_data(gt_data)
        img_pred.set_data(pred_data)
        img_diff.set_data(diff_data)

        # Update titles and adjust color limits for each image to match the new data range
        ax[0].set_title(f"Ground Truth at Step {step}")
        ax[1].set_title(f"Local PINO Prediction at Step {step}")
        ax[2].set_title(f"Difference at Step {step}")

        # Update the color limits for each image using the mappable object (img_gt, img_pred, img_diff)
        img_gt.set_clim(np.min(gt_data), np.max(gt_data))
        img_pred.set_clim(np.min(pred_data), np.max(pred_data))
        img_diff.set_clim(np.min(diff_data), np.max(diff_data))

        # Update the colorbars to reflect the new clim values
        cbar_gt.mappable.set_clim(np.min(gt_data), np.max(gt_data))
        cbar_pred.mappable.set_clim(np.min(pred_data), np.max(pred_data))
        cbar_diff.mappable.set_clim(np.min(diff_data), np.max(diff_data))

        return [img_gt, img_pred, img_diff]

    # Create the animation
    ani = animation.FuncAnimation(fig, update_fig, frames=range(0, total_steps), interval=1000/fps, blit=True)
    # Save the animation as a GIF
    ani.save(output_gif, writer='pillow', fps=fps)

# CUDA support
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# 參數
case_idx = 0
file_path = "./pressure_data_3d.h5" 
model_inf = modulus.Module.from_checkpoint("./outputs/checkpoints/FourierNeuralOperator.0.399.mdlus").to(device)
model_inf.eval()

# 獲取輸入數據
inputdataset, total_steps = input_data(file_path, device, idx = case_idx)

# 初始化預測和存儲函數
result_matrix = predict_and_store_with_sliding(model_inf, inputdataset, total_steps, device)

create_comparison_gif(result_matrix, file_path, fps=20, total_steps = total_steps, idx = 0)



