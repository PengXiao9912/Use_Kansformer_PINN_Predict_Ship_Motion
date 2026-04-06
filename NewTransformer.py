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
# 参数设置
# ====================
SEQ_LEN = 100
BATCH_SIZE = 256
EPOCHS = 10
LR = 0.0001
PATIENCE = 7
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

# + 新增：可视化注意力层
class VisualTransformerEncoderLayer(nn.Module):  
    def __init__(self, model_dim=64, num_heads=8, dim_feedforward=None, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(model_dim, num_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(model_dim, dim_feedforward or model_dim*4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward or model_dim*4, model_dim)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.attn_weights = None

    def forward(self, src):
        attn_output, attn_weights = self.self_attn(src, src, src,
                                                  need_weights=True,
                                                  average_attn_weights=False)
        self.attn_weights = attn_weights  # [B, heads, T, T]
        src2 = self.dropout1(attn_output)
        src = self.norm1(src + src2)
        ff = self.linear2(self.dropout(F.gelu(self.linear1(src))))
        src2 = self.dropout2(ff)
        return self.norm2(src + src2)



class EnhancedTransformer(nn.Module):
    def __init__(self, input_dim=15, model_dim=64, num_heads=8, 
                 num_layers=4, dropout=0.1, output_dim=2):
        super().__init__()
        
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_fc = nn.Linear(input_dim, model_dim)
        
        self.pos_encoder = nn.Parameter(torch.zeros(1, SEQ_LEN, model_dim))
        position = torch.arange(SEQ_LEN).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2) * (-math.log(10000.0) / model_dim))
        self.pos_encoder.data[0, :, 0::2] = torch.sin(position * div_term)
        self.pos_encoder.data[0, :, 1::2] = torch.cos(position * div_term)

    
        layers = []
        for _ in range(num_layers):
            layers.append(VisualTransformerEncoderLayer(model_dim, num_heads, model_dim*4, dropout))
        self.layers = nn.ModuleList(layers)
        
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
        attn_maps = []  # + 用于收集所有层的注意力权重
        # + 逐层前向，保存 attn_weights
        for layer in self.layers:
            x = layer(x)
            attn_maps.append(layer.attn_weights)
        x = x.mean(dim=1)  # [B, D]
        out = self.output_layer(x)
        return out, attn_maps  # + 返回注意力图

def main():
    data_dir = r"D:\code\TranformerPINN\data"
    all_files = np.random.permutation(glob.glob(os.path.join(data_dir, "*.xlsx")))
    split_idx = int(0.8 * len(all_files))
    
    X_train, y_train, scaler_input, scaler_target = process_data(all_files[:split_idx])
    X_val, y_val, _, _ = process_data(all_files[split_idx:], scaler_input, scaler_target, False)

    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))

    # train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    # val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE*2, pin_memory=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=dl_generator
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE*2,
        shuffle=False,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=dl_generator
    )

    loss_records = []
    csv_save_path = './loss_history.csv' 

    model = EnhancedTransformer(input_dim=15).to(DEVICE)  # 输入维度修正为15
    criterion = nn.HuberLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)
    early_stop = EarlyStopping(patience=PATIENCE)
    scaler = torch.cuda.amp.GradScaler()  # 使用 CUDA AMP 的 GradScaler

    best_loss = float('inf')
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():  # 更新API
                outputs, _ = model(inputs)
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * inputs.size(0)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(DEVICE, non_blocking=True)
                targets = targets.to(DEVICE, non_blocking=True)
                outputs, _ = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_dataset)
        val_loss /= len(val_dataset)

        loss_records.append({
            'epoch':       epoch + 1,
            'train_loss':  train_loss,
            'val_loss':    val_loss
        })

        scheduler.step(val_loss)
        
        if early_stop(val_loss):
            print(f"Early stopping at epoch {epoch+1}")
            break

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'model': model.state_dict(),
                'scaler_input': scaler_input,
                'scaler_target': scaler_target
            }, 'best_model.pth')

        print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        # ====================
        # 十步一保存注意力图（每10个 epoch 保存一次）
        # ====================
        # + 在训练结束后，可视化第一层第一个头的注意力权重示例
        attn_save_dir = './attn_maps'
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                sample_input = torch.FloatTensor(X_val[:1]).to(DEVICE)
                _, attn_maps = model(sample_input)
            for layer_idx, layer_map in enumerate(attn_maps):
                num_heads = layer_map.size(1)
                for head_idx in range(num_heads):
                    attn = layer_map[0, head_idx].cpu().numpy()
                    save_path = os.path.join(attn_save_dir,
                                             f'epoch{epoch+1}_layer{layer_idx}_head{head_idx}.png')
                    plt.figure(figsize=(6,5))
                    plt.imshow(attn, aspect='auto')
                    plt.colorbar(label='Attention weight')
                    plt.title(f'Epoch{epoch+1} Layer{layer_idx} Head{head_idx}')
                    plt.savefig(save_path)
                    plt.close()

    df_loss = pd.DataFrame(loss_records)
    df_loss.to_csv(csv_save_path, index=False)
    print(f"已将 loss 历史保存到 {csv_save_path}")

if __name__ == "__main__":
    main()