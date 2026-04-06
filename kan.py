import torch
import torch.nn.functional as F
import math

class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,  # 输入/输出特征数
        grid_size=3,   # 网格点数
        spline_order=2,  # B样条阶数
        scale_noise=0.1,  # 噪声缩放因子
        scale_base=1.0,   # 基函数的缩放因子
        scale_spline=1.0,  # 样条的缩放因子
        enable_standalone_scale_spline=True, # 是否独立缩放样条权重
        base_activation=torch.nn.GELU, # 基函数激活函数，改为 GELU
        grid_eps=0.02,  # 网格更新时的混合系数
        grid_range=[-1, 1], # 初始网格范围
        init_method='xavier',  # 新增：初始化方式
        spline_init_std=1e-3   # 新增：样条权重初始化标准差
        ):

        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        # 初始化网格
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        # 定义基权重和样条权重
        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        # 保存新参数
        self.init_method = init_method
        self.spline_init_std = spline_init_std

        self.reset_parameters()

    def reset_parameters(self):
        # 基函数权重初始化：Xavier
        if self.init_method == 'xavier':
            torch.nn.init.xavier_uniform_(self.base_weight)
        else:
            torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)

        # 样条权重初始化为小幅高斯噪声然后插值
        with torch.no_grad():
            noise_shape = (self.grid_size + 1, self.in_features, self.out_features)
            noise = (torch.randn(*noise_shape) * self.spline_init_std)
            coeff = self.curve2coeff(
                self.grid.T[self.spline_order:-self.spline_order],
                noise
            )
            self.spline_weight.data.copy_(coeff)
            if self.enable_standalone_scale_spline:
                torch.nn.init.constant_(self.spline_scaler, self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, :-(k + 1)])
                / (grid[:, k:-1] - grid[:, :-(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1:] - x)
                / (grid[:, k + 1:] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)
        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        result = solution.permute(2, 0, 1)
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        if self.enable_standalone_scale_spline:
            return self.spline_weight * self.spline_scaler.unsqueeze(-1)
        return self.spline_weight

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x_flat = x.view(-1, self.in_features)
        base_out = F.linear(self.base_activation(x_flat), self.base_weight)
        spline_basis = self.b_splines(x_flat)
        spline_flat = spline_basis.view(x_flat.size(0), -1)
        spline_out = F.linear(spline_flat, self.scaled_spline_weight.view(self.out_features, -1))
        out = base_out + spline_out
        return out.view(*original_shape[:-1], self.out_features)

    def visualize_b_splines(self, x: torch.Tensor):
        # 可视化单点 B 样条激活
        spline_bases = self.b_splines(x.unsqueeze(0)).detach().cpu().numpy()[0, 0]
        import matplotlib.pyplot as plt
        plt.bar(range(len(spline_bases)), spline_bases)
        plt.show()

    def visualize_b_splines_over_range(self, feature_index=0):
        import matplotlib.pyplot as plt
        xs = torch.linspace(self.grid[0, self.spline_order], self.grid[0, -self.spline_order-1], 200).unsqueeze(1)
        bases = self.b_splines(xs).detach().cpu().numpy()[:, feature_index, :]
        for i in range(bases.shape[1]):
            plt.plot(xs.numpy().squeeze(), bases[:, i], label=f'B{i}')
        plt.legend()
        plt.show()

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)
        splines = self.b_splines(x).permute(1, 0, 2)
        orig = self.scaled_spline_weight.permute(1, 2, 0)
        unreduced = torch.bmm(splines, orig).permute(1, 0, 2)

        x_sorted = torch.sort(x, dim=0)[0]
        idx = torch.linspace(0, batch-1, self.grid_size+1, device=x.device, dtype=torch.int64)
        grid_adapt = x_sorted[idx]
        uniform_step = (x_sorted[-1]-x_sorted[0]+2*margin)/self.grid_size
        grid_unif = torch.arange(self.grid_size+1, device=x.device).unsqueeze(1) * uniform_step + x_sorted[0] - margin
        new_grid = self.grid_eps*grid_unif + (1-self.grid_eps)*grid_adapt
        full = torch.cat([
            new_grid[:1] - uniform_step*torch.arange(self.spline_order,0,-1,device=x.device).unsqueeze(1),
            new_grid,
            new_grid[-1:] + uniform_step*torch.arange(1,self.spline_order+1,device=x.device).unsqueeze(1)
        ], dim=0)
        self.grid.copy_(full.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        l1 = self.spline_weight.abs().mean(-1)
        act_loss = l1.sum()
        p = l1 / act_loss
        ent_loss = -torch.sum(p * p.log())
        return regularize_activation*act_loss + regularize_entropy*ent_loss

class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.GELU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.layers = torch.nn.ModuleList([
            KANLinear(
                in_features=a,
                out_features=b,
                grid_size=grid_size,
                spline_order=spline_order,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline,
                base_activation=base_activation,
                grid_eps=grid_eps,
                grid_range=grid_range,
                init_method='xavier',
                spline_init_std=1e-3
            ) for a,b in zip(layers_hidden, layers_hidden[1:])
        ])

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, *args, **kwargs):
        return sum(layer.regularization_loss(*args, **kwargs) for layer in self.layers)

# 示例：如何使用可视化
if __name__ == "__main__":
    # 创建一个简单的 KANLinear 层用于可视化
    kan_layer = KANLinear(
        in_features=1,    # 输入维度 1，方便观察单变量样条
        out_features=1,   # 输出维度 1W
        grid_size=5,
        spline_order=3
    )

    # 可视化完整的 B 样条基函数在输入区间 [-1, 1] 上的响应曲线
    kan_layer.visualize_b_splines_over_range()
