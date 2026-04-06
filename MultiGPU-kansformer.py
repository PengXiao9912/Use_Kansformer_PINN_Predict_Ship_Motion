import os
import math
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
import random
from sklearn.preprocessing import StandardScaler

# ====================
# 配置与初始化
# ====================
SEQ_LEN = 100
PER_DEVICE_BATCH_SIZE = 256  # 每卡实际 batch size
ACCUMULATION_STEPS = 2
EPOCHS = 100
LR = 1e-4
PATIENCE = 7
GRAD_CLIP = 1.0
RANDOM_SEED = 42
NUM_WORKERS = 4


# 统一设置随机种子
def seed_everything(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything()

# 设置 cuDNN
torch.backends.cudnn.benchmark = True

# ====================
# 分布式与 Logger
# ====================
def setup_distributed(rank, world_size):
    master_addr = os.environ['MASTER_ADDR']
    master_port = os.environ['MASTER_PORT']
    init_method = f"tcp://{master_addr}:{master_port}"
    dist.init_process_group(
        backend='nccl',
        init_method=init_method,
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank)

  # ====================
# 数据处理模块
# ====================
INPUT_COLS = [
    "时间(秒)", "航向", "速度(m/s)", "艏摇角速度(°/s)",
    "左舵角", "左转速", "左功率", "左转矩",
    "右舵角", "右转速", "右功率", "右转矩",
    "风速(m/s)", "风向(T/R)"
]
TARGET_COLS = ["相对经度", "相对纬度"]
EPS = 1e-8

# 1) 读取与清洗单文件
def load_and_clean(file_path):
    df = pd.read_excel(file_path)
    # 角度拆分
    if '风向(T/R)' in df.columns:
        df['风向_sin'] = np.sin(df['风向(T/R)'] * np.pi / 180)
        df['风向_cos'] = np.cos(df['风向(T/R)'] * np.pi / 180)
    if '加速度' in df.columns:
        df = df.drop(columns=['加速度'])
    df = df.interpolate(method='linear').fillna(method='bfill')
    return df

# 2) 拟合 scaler
def fit_scalers(dfs):
    inputs = pd.concat([df[[c for c in INPUT_COLS if c!='风向(T/R)'] + ['风向_sin','风向_cos']] for df in dfs])
    targets = pd.concat([df[TARGET_COLS] for df in dfs])
    # 避免零方差
    for col in inputs.columns:
        if inputs[col].std() < EPS:
            inputs[col] += np.random.normal(0, EPS, size=len(inputs))
    scaler_x = StandardScaler().fit(inputs)
    scaler_y = StandardScaler().fit(targets)
    return scaler_x, scaler_y

# 3) 构建序列样本
def make_sequences(df, scaler_x, scaler_y):
    cols = [c for c in INPUT_COLS if c!='风向(T/R)'] + ['风向_sin','风向_cos']
    x = scaler_x.transform(df[cols])
    y = scaler_y.transform(df[TARGET_COLS])
    seq_x, seq_y = [], []
    for i in range(len(x) - SEQ_LEN):
        seq_x.append(x[i:i+SEQ_LEN])
        seq_y.append(y[i+SEQ_LEN])
    return np.array(seq_x), np.array(seq_y)

# 4) 统一数据处理
def process_data(file_list, scaler_x=None, scaler_y=None, is_train=True):
    dfs = [load_and_clean(f) for f in file_list]
    if is_train:
        scaler_x, scaler_y = fit_scalers(dfs)
    X, y = [], []
    for df in dfs:
        sx, sy = make_sequences(df, scaler_x, scaler_y)
        if len(sx):
            X.append(sx); y.append(sy)
    X = np.vstack(X) if X else np.empty((0, SEQ_LEN, len(INPUT_COLS)+1))
    y = np.vstack(y) if y else np.empty((0, len(TARGET_COLS)))
    return X, y, scaler_x, scaler_y

# 5) 早停
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# ====================
# KAN 模块实现
# ====================
class KANLinear(nn.Module):
    def __init__(self, in_features, out_features, **kan_kwargs):
        super().__init__()
        from kan import KANLinear as _KANLinear
        # 过滤只支持的参数
        valid_keys = [
            'grid_size','spline_order','scale_noise','scale_base','scale_spline',
            'enable_standalone_scale_spline','base_activation','grid_eps','grid_range',
            'init_method','spline_init_std'
        ]
        filtered_kwargs = {k: v for k, v in kan_kwargs.items() if k in valid_keys}
        self.kan_linear = _KANLinear(in_features, out_features, **filtered_kwargs)

    def forward(self, x):
        shape = x.shape
        x_flat = x.view(-1, shape[-1])
        y_flat = self.kan_linear(x_flat)
        # 确保所有设备完成计算
        torch.cuda.synchronize()  # 新增同步
        return y_flat.view(*shape[:-1], -1)

class KANFeedForward(nn.Module):
    def __init__(self, model_dim, dim_feedforward, dropout=0.1, use_update=False, **kan_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(model_dim)
        self.kan1 = KANLinear(model_dim, dim_feedforward, **kan_kwargs)
        self.norm2 = nn.LayerNorm(dim_feedforward)
        self.kan2 = KANLinear(dim_feedforward, model_dim, **kan_kwargs)
        self.norm3 = nn.LayerNorm(model_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.use_update = use_update

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.kan1(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.kan2(x)
        x = self.drop(x)
        out = residual + x
        if self.use_update:
            self.kan1.kan_linear.update_grid(x)
            self.kan2.kan_linear.update_grid(x)
        return self.norm3(out)


class VisualTransformerEncoderLayer(nn.Module):
    def __init__(self, model_dim=64, num_heads=8, dim_feedforward=None, dropout=0.1, kan_kwargs=None):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(model_dim, num_heads, dropout=dropout, batch_first=True)
        dim_ff = dim_feedforward if dim_feedforward is not None else model_dim*3
        self.kan_ff = KANFeedForward(model_dim, dim_ff, **(kan_kwargs or {}))
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.attn_weights = None

    def forward(self, src):
        attn_out, attn_w = self.self_attn(src, src, src, need_weights=True, average_attn_weights=False)
        self.attn_weights = attn_w
        src = self.norm1(src + self.dropout1(attn_out))
        ff_out = self.kan_ff(src)
        src = self.norm2(src + self.dropout2(ff_out))
        return src, attn_w

class EnhancedTransformer(nn.Module):
    def __init__(self, input_dim=15, model_dim=64, num_heads=8,
                 num_layers=4, dropout=0.1, output_dim=2, kan_kwargs=None):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_fc = nn.Linear(input_dim, model_dim)
        self.pos_encoder = nn.Parameter(torch.zeros(1, SEQ_LEN, model_dim))
        pos = torch.arange(SEQ_LEN).unsqueeze(1)
        div = torch.exp(torch.arange(0, model_dim, 2) * (-math.log(10000.0) / model_dim))
        self.pos_encoder.data[0, :, 0::2] = torch.sin(pos * div)
        self.pos_encoder.data[0, :, 1::2] = torch.cos(pos * div)
        self.layers = nn.ModuleList([
            VisualTransformerEncoderLayer(model_dim, num_heads, model_dim*4, dropout, kan_kwargs)
            for _ in range(num_layers)
        ])
        self.output_layer = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, output_dim)
        )
        self._init_weights()
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
    def forward(self, x):
        x = self.input_norm(x)
        x = self.input_fc(x) + self.pos_encoder
        attn_maps = []
        for layer in self.layers:
            x, w = layer(x)
            attn_maps.append(w)
        x = x.mean(dim=1)
        out = self.output_layer(x)
        return out, attn_maps

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
generator = torch.Generator()
generator.manual_seed(RANDOM_SEED)

def create_dataloader(dataset, rank, world_size, train=True):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=train, drop_last=train)
    return DataLoader(
        dataset,
        batch_size=PER_DEVICE_BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator
    )



# ====================
# 主训练函数
# ====================
def main(rank, world_size, data_dir, save_dir='./'):
    setup_distributed(rank, world_size)
    try:
        files = sorted(glob.glob(os.path.join(data_dir, '*.xlsx')))
        np.random.seed(RANDOM_SEED)
        perm = np.random.permutation(len(files))
        files = [files[i] for i in perm]
        split = int(len(files)*0.8)
        train_files, val_files = files[:split], files[split:]

        X_tr, y_tr, sx, sy = process_data(train_files, is_train=True)
        X_va, y_va, _, _ = process_data(val_files, sx, sy, is_train=False)
        train_ds = TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr).float())
        val_ds   = TensorDataset(torch.from_numpy(X_va).float(), torch.from_numpy(y_va).float())

        train_loader = create_dataloader(train_ds, rank, world_size, train=True)
        val_loader   = create_dataloader(val_ds, rank, world_size, train=False)

        model = EnhancedTransformer(
            input_dim=(len(INPUT_COLS)-1+2),
            model_dim=256,
            num_heads=8,
            num_layers=6,
            dropout=0.1,
            output_dim=len(TARGET_COLS),
            kan_kwargs=None
        ).to(rank)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        scaler = GradScaler()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)
        early_stop = EarlyStopping(patience=PATIENCE)

        best_val = float('inf')
        for epoch in range(EPOCHS):
            model.train()
            train_loader.sampler.set_epoch(epoch)
            total_loss = 0.0
            for step, (xb, yb) in enumerate(train_loader):
                xb, yb = xb.to(rank, non_blocking=True), yb.to(rank, non_blocking=True)
                with autocast():
                    out, _ = model(xb)
                    loss = F.huber_loss(out, yb)
                scaler.scale(loss).backward()
                if (step+1) % ACCUMULATION_STEPS == 0:
                    dist.barrier()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                total_loss += loss.item() * xb.size(0)

            loss_tensor = torch.tensor(total_loss, device=rank)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            train_loss = loss_tensor.item() / len(train_ds)

            if rank == 0:
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb, yb = xb.to(rank), yb.to(rank)
                        out, _ = model(xb)
                        val_loss += F.huber_loss(out, yb).item() * xb.size(0)
                val_loss /= len(val_ds)
                print(f"Epoch {epoch+1}: Train {train_loss:.4f}, Val {val_loss:.4f}")
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save({'model': model.module.state_dict(), 'scaler_x': sx, 'scaler_y': sy},
                               os.path.join(save_dir, 'best.pth'))
                scheduler.step(val_loss)
                early_stop(val_loss)
                if early_stop.early_stop:
                    break

                dist.barrier()  # 新增
            else:
                # 非rank0进程执行空屏障等待
                dist.barrier()
    finally:
        dist.destroy_process_group()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='./')
    args = parser.parse_args()

    # torchrun 会给我们填充 LOCAL_RANK 和 WORLD_SIZE
    rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    # setup_distributed(rank, world_size)
    main(rank, world_size, args.data_dir, args.save_dir)

