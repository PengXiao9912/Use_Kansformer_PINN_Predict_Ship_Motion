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
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import random

# ====================
# 全局参数
# ====================
SEQ_LEN = 100
BATCH_SIZE = 256
EPOCHS = 100
LR = 1e-4
PATIENCE = 7
VIS_INTERVAL = 10  # 每隔多少步可视化一次
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
dl_generator = torch.Generator()
dl_generator.manual_seed(RANDOM_SEED)


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
# 数据处理函数
# ====================
def process_data(file_list, scaler_input=None, scaler_target=None, is_train=True):
    input_cols = ["时间(秒)", "航向", "速度(m/s)", "艏摇角速度(°/s)",
                "左舵角", "左转速", "左功率", "左转矩",
                "右舵角", "右转速", "右功率", "右转矩",
                "风速(m/s)", "风向(T/R)"]
    target_cols = ["相对经度", "相对纬度"]
    
    def process_wind_direction(df):
        df['风向_sin'] = np.sin(df['风向(T/R)'] * np.pi / 180)
        df['风向_cos'] = np.cos(df['风向(T/R)'] * np.pi / 180)
        return df.drop(columns=['风向(T/R)'])

    all_X, all_y = [], []
    
    if is_train:
        scaler_input = StandardScaler()
        scaler_target = StandardScaler()
        train_inputs = []
        train_targets = []

    for file in file_list:
        df = pd.read_excel(file).pipe(process_wind_direction)
        df = df.drop(columns=["加速度"]) if "加速度" in df.columns else df
        
        current_cols = [c for c in input_cols if c != '风向(T/R)'] + ['风向_sin', '风向_cos']
        if df[current_cols + target_cols].isnull().sum().sum() > 0:
            df = df.interpolate(method='linear').fillna(method='bfill')
        
        if is_train:
            train_inputs.append(df[current_cols])
            train_targets.append(df[target_cols])
        else:
            df[current_cols] = scaler_input.transform(df[current_cols])
            df[target_cols] = scaler_target.transform(df[target_cols])

    if is_train:
        full_inputs = pd.concat(train_inputs)
        full_targets = pd.concat(train_targets)
        
        epsilon = 1e-8
        for col in full_inputs.columns:
            if full_inputs[col].std() < epsilon:
                full_inputs[col] += np.random.normal(0, epsilon, size=len(full_inputs))
        
        scaler_input.fit(full_inputs)
        scaler_target.fit(full_targets)

    for file in file_list:
        df = pd.read_excel(file).pipe(process_wind_direction)
        df = df.drop(columns=["加速度"]) if "加速度" in df.columns else df
        df = df.interpolate(method='linear').fillna(method='bfill')
        
        processed_input = scaler_input.transform(df[current_cols])
        processed_target = scaler_target.transform(df[target_cols])
        
        seq_X, seq_y = [], []
        for i in range(len(processed_input) - SEQ_LEN):
            seq_X.append(processed_input[i:i+SEQ_LEN])
            seq_y.append(processed_target[i+SEQ_LEN])
        
        if len(seq_X) > 0:
            all_X.extend(seq_X)
            all_y.extend(seq_y)

    return np.array(all_X), np.array(all_y), scaler_input, scaler_target


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
        # 绝对位置编码
        self.pos_encoder = nn.Parameter(torch.zeros(1, SEQ_LEN, model_dim))
        pos = torch.arange(SEQ_LEN).unsqueeze(1)
        div = torch.exp(torch.arange(0, model_dim, 2) * (-math.log(10000.0) / model_dim))
        self.pos_encoder.data[0, :, 0::2] = torch.sin(pos * div)
        self.pos_encoder.data[0, :, 1::2] = torch.cos(pos * div)

        self.layers = nn.ModuleList([
            VisualTransformerEncoderLayer(
                model_dim, num_heads, model_dim*4, dropout, kan_kwargs=kan_kwargs
            ) for _ in range(num_layers)
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
        # x: [B, T, input_dim]
        x = self.input_norm(x)
        x = self.input_fc(x) + self.pos_encoder
        attn_maps = []
        for layer in self.layers:
            x, w = layer(x)
            attn_maps.append(w)
        x = x.mean(dim=1)
        out = self.output_layer(x)
        return out, attn_maps

# ====================
# 主训练函数
# ====================
def main(data_dir, save_dir='./'):
    os.makedirs(save_dir, exist_ok=True)
    img_dir = os.path.join(save_dir, 'visualizations')
    os.makedirs(img_dir, exist_ok=True)

    all_files = np.random.permutation(glob.glob(os.path.join(data_dir, '*.xlsx')))
    split = int(0.8 * len(all_files))
    X_train, y_train, scaler_input, scaler_target = process_data(all_files[:split])
    X_val, y_val, _, _ = process_data(all_files[split:], scaler_input, scaler_target, False)

    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=dl_generator
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE*2,
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=dl_generator
    )

    kan_kwargs = dict(grid_size=6, spline_order=3, scale_noise=0.1,
                      scale_base=1.0, scale_spline=1.0)
    model = EnhancedTransformer(input_dim=15, model_dim=128, num_heads=8,
                                num_layers=4, dropout=0.1,
                                output_dim=2, kan_kwargs=kan_kwargs).to(DEVICE)
    criterion = nn.HuberLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)
    early_stop = EarlyStopping(patience=PATIENCE)

    best = float('inf')
    loss_hist = []
    global_step = 0
    for epoch in range(EPOCHS):
        model.train()
        t_loss = 0
        for xb, yb in train_loader:
            global_step += 1
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            out, attn_maps = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step(loss)
            t_loss += loss.item() * xb.size(0)

            # 可视化每 VIS_INTERVAL 步
            if global_step % VIS_INTERVAL == 0:
                # Attention
                with torch.no_grad():
                    _, attn = model(xb)
                for l, layer_attn in enumerate(attn):
                    for h in range(layer_attn.size(1)):
                        A = layer_attn[0, h].cpu().numpy()
                        plt.figure(figsize=(4,4))
                        plt.imshow(A, cmap='viridis')
                        plt.title(f'E{epoch+1}_S{global_step}_L{l}_H{h}')
                        plt.xlabel('Key')
                        plt.ylabel('Query')
                        plt.colorbar()
                        plt.tight_layout()
                        plt.savefig(os.path.join(img_dir, f'attn_e{epoch+1}_s{global_step}_l{l}_h{h}.png'))
                        plt.close()
                # KAN Grid
                for l, layer in enumerate(model.layers):
                    kan1 = layer.kan_ff.kan1.kan_linear
                    grid = kan1.grid.detach().cpu().numpy()
                    plt.figure(figsize=(4,2))
                    plt.plot(grid, marker='o')
                    plt.title(f'E{epoch+1}_S{global_step}_L{l}_KANgrid')
                    plt.xlabel('Node Index')
                    plt.ylabel('Grid Value')
                    plt.tight_layout()
                    plt.savefig(os.path.join(img_dir, f'kan_e{epoch+1}_s{global_step}_l{l}_grid.png'))
                    plt.close()

        t_loss /= len(train_dataset)
        model.eval()
        v_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                out, _ = model(xb)
                v_loss += criterion(out, yb).item() * xb.size(0)
        v_loss /= len(val_dataset)

        scheduler.step(v_loss)
        loss_hist.append({'epoch': epoch+1, 'train': t_loss, 'val': v_loss})
        print(f"Epoch {epoch+1}: train={t_loss:.6f}, val={v_loss:.6f}")
        if v_loss < best:
            best = v_loss
            torch.save({'model': model.state_dict(), 's_in': scaler_input, 's_tg': scaler_target},
                       os.path.join(save_dir, 'kansformer_best_model.pth'))

    pd.DataFrame(loss_hist).to_csv(os.path.join(save_dir, 'kansformer_loss_history.csv'), index=False)

if __name__ == '__main__':
    data_dir = r"./data"
    main(data_dir, save_dir='./output')
