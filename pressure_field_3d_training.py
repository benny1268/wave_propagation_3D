import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from modulus.launch.logging import LaunchLogger
from modulus.launch.utils.checkpoint import save_checkpoint, load_checkpoint
from modulus.models.fno import FNO
from modulus.models.afno import AFNO
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from utils_test import HDF5MapStyleDataset
import torch.multiprocessing as mp
from modulus.distributed import DistributedManager
import modulus

mp.set_start_method('spawn', force=True)

@hydra.main(version_base="1.3", config_path="conf", config_name="config_pino.yaml")
def main(cfg: DictConfig):

    # CUDA support
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available. Running on GPU.")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Running on CPU.")

    DistributedManager.initialize()  # Only call this once in the entire script!
    dist = DistributedManager()  # call if required elsewhere

    LaunchLogger.initialize()

    dataset = HDF5MapStyleDataset(to_absolute_path("./training_data_3d"), device=dist.device)
    
    # 切分資料集，90% 訓練，10% 驗證
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # 設置 Dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8, drop_last=True, persistent_workers=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    model = FNO(
        in_channels=cfg.model.fno.in_channels,
        out_channels=cfg.model.fno.out_channels,
        decoder_layers=cfg.model.fno.decoder_layers,
        decoder_layer_size=cfg.model.fno.decoder_layer_size,
        dimension=cfg.model.fno.dimension,
        latent_channels=cfg.model.fno.latent_channels,
        num_fno_layers=cfg.model.fno.num_fno_layers,
        num_fno_modes=cfg.model.fno.num_fno_modes,
        padding=cfg.model.fno.padding,
    ).to(dist.device)
    # model = modulus.Module.from_checkpoint("./checkpoint/FourierNeuralOperator.0.399.mdlus").to("cuda")

    # 自定義 RMSE 損失函數
    class RMSELoss(torch.nn.Module):
        def __init__(self):
            super(RMSELoss, self).__init__()

        def forward(self, prediction, target):
            mse_loss = torch.mean((prediction - target) ** 2)  # 計算 MSE
            rmse_loss = torch.sqrt(mse_loss)  # 計算 RMSE
            return rmse_loss

    criterion = RMSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        betas=(0.9, 0.999),
        lr=cfg.start_lr,
        weight_decay=0.0001,
    )

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.gamma)


    for epoch in range(cfg.max_epochs):
        # wrap epoch in launch logger for console logs
        with LaunchLogger("train", epoch=epoch, num_mini_batch=len(train_dataloader), mini_batch_log_freq = 100, epoch_alert_freq=10) as log:
            model.train()
            for data in train_dataloader:
                optimizer.zero_grad()
                invar, outvar = data[0].to(dist.device), data[1].to(dist.device)  # 轉換到 GPU
                # Compute forward pass
                out = model(invar)
                # Compute data loss
                loss_data = criterion(outvar, out)
                # Backward pass
                loss_data.backward()

                # Optimizer
                optimizer.step()
                log.log_minibatch({"Data Loss": loss_data.detach()})

            scheduler.step()
            log.log_epoch({"Learning Rate": optimizer.param_groups[0]["lr"]})

        with LaunchLogger("valid", epoch=epoch) as log:
            model.eval()
            loss_epoch = 0
            with torch.no_grad():
                for data in val_dataloader:
                    invar, outvar = data[0].to(dist.device), data[1].to(dist.device)  # 轉換到 GPU
                    out = model(invar)
                    loss_epoch += criterion(outvar, out)
            val_error = loss_epoch / len(val_dataloader)
            log.log_epoch({"Validation Loss": val_error})

        if(epoch%20 == 0):
            save_checkpoint(
                "./checkpoints",
                models=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
            )
        
        if(epoch == cfg.max_epochs - 1):
            save_checkpoint(
                "./checkpoints",
                models=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
            )

if __name__ == "__main__":
    main()
