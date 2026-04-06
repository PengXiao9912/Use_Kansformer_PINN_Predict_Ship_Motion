import os
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from kansformerPINN import process_data, EnhancedTransformer, MMG_PINN_Loss, EarlyStopping
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
import numpy as np
import pandas as pd
import glob
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import datetime

SEQ_LEN = 100
BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-4
PATIENCE = 7
VIS_INTERVAL = 10


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def log_gpu_memory(rank):
    allocated = torch.cuda.memory_allocated(rank) / 1024 ** 2
    reserved = torch.cuda.memory_reserved(rank) / 1024 ** 2
    print(f"[Rank {rank}] Memory allocated: {allocated:.2f} MB, reserved: {reserved:.2f} MB")

def train_ddp(rank, world_size, data_dir, save_dir):
    setup(rank, world_size)
    DEVICE = torch.device(f"cuda:{rank}")

    all_files = np.random.permutation(glob.glob(os.path.join(data_dir, '*.xlsx')))
    split = int(0.8 * len(all_files))
    X_train, y_train, scaler_input, scaler_target = process_data(all_files[:split])
    X_val, y_val, _, _ = process_data(all_files[split:], scaler_input, scaler_target, False)

    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2, sampler=val_sampler)

    kan_kwargs = dict(grid_size=6, spline_order=3, scale_noise=0.1, scale_base=1.0, scale_spline=1.0)
    model = EnhancedTransformer(input_dim=15, model_dim=128, num_heads=8,
                                num_layers=4, dropout=0.1, output_dim=2,
                                kan_kwargs=kan_kwargs).to(DEVICE)
    model = DDP(model, device_ids=[rank])

    criterion = nn.HuberLoss()
    pinn_criterion = MMG_PINN_Loss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)

    best = float('inf')
    log_file = os.path.join(save_dir, f'train_log_rank{rank}.txt')

    for epoch in range(EPOCHS):
        model.train()
        train_sampler.set_epoch(epoch)
        t_total_loss = t_data_loss = t_pde_loss = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            xb.requires_grad_(True)
            optimizer.zero_grad()
            out, _ = model(xb)
            data_loss = criterion(out, yb)
            _, _, _, pde_loss = pinn_criterion(xb, out)
            loss = data_loss + pde_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_total_loss += loss.item() * xb.size(0)
            t_data_loss += data_loss.item() * xb.size(0)
            t_pde_loss += pde_loss.item() * xb.size(0)

        t_total_loss /= len(train_dataset)
        t_data_loss /= len(train_dataset)
        t_pde_loss /= len(train_dataset)

        model.eval()
        v_total_loss = v_data_loss = v_pde_loss = 0
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            xb.requires_grad_(True)
            with torch.set_grad_enabled(True):
                out, _ = model(xb)
                _, _, _, pde_loss = pinn_criterion(xb, out)
            with torch.no_grad():
                data_loss = criterion(out, yb)
            total_loss = data_loss + pde_loss
            v_total_loss += total_loss.item() * xb.size(0)
            v_data_loss += data_loss.item() * xb.size(0)
            v_pde_loss += pde_loss.item() * xb.size(0)

        v_total_loss /= len(val_dataset)
        v_data_loss /= len(val_dataset)
        v_pde_loss /= len(val_dataset)
        scheduler.step(v_total_loss)

        if rank == 0:
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_line = (f"[{now}] [Epoch {epoch+1}] "
                        f"Train Total: {t_total_loss:.6f}, Data: {t_data_loss:.6f}, PDE: {t_pde_loss:.6f} | "
                        f"Val Total: {v_total_loss:.6f}, Data: {v_data_loss:.6f}, PDE: {v_pde_loss:.6f}")
            print(log_line)
            log_gpu_memory(rank)
            with open(log_file, 'a') as f:
                f.write(log_line + '\n')
            if v_total_loss < best:
                best = v_total_loss
                torch.save({
                    'model': model.module.state_dict(),
                    's_in': scaler_input,
                    's_tg': scaler_target,
                }, os.path.join(save_dir, 'best_model.pth'))

    cleanup()

def main():
    data_dir = "./data"
    save_dir = "./output"
    os.makedirs(save_dir, exist_ok=True)

    # ✅ CUDA 初始化检查
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，请检查驱动和环境变量设置！")

    num_devices = torch.cuda.device_count()
    if num_devices == 0:
        raise RuntimeError("未检测到任何 CUDA GPU，请确保 CUDA_VISIBLE_DEVICES 设置正确！")

    try:
        torch.cuda.get_device_name(0)  # 强制触发一次初始化
    except RuntimeError as e:
        raise RuntimeError(f"CUDA 初始化失败：{str(e)}\n"
                           f"请确保驱动安装正确，且不要在程序中间设置 CUDA_VISIBLE_DEVICES") from e

    world_size = num_devices
    mp.spawn(train_ddp,
             args=(world_size, data_dir, save_dir),
             nprocs=world_size,
             join=True)


if __name__ == '__main__':
    main()