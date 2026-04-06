import torch
import torch.nn as nn

class MMG_PINN_Loss(nn.Module):
    def __init__(self, weight_u=1.0, weight_v=1.0, weight_r=1.0):
        super().__init__()
        # 固定船体参数
        self.Lpp = XX # 船长
        self.B = XX  # 型宽
        self.d = XX # 吃水
        self.Rho = XX # 海水密度
        self.Cb = XX # 方形系数
        self.Cm = XX # 中横剖面系数
        self.Cpa = self.Cb / self.Cm # 船后菱形系数
        self.xpp = XX # 螺旋桨纵向无因次化安装位置
        self.Lps = XX # 螺旋桨的艏向力矩力臂
        self.Dp = XX # 螺旋桨直径

        # 推力系数参数
        self.k0 = XX
        self.k1 = XX
        self.k2 = XX

        # 质量与转动惯量
        self.m =XX * XX  # 排水量*海水密度
        self.Iz = 0.25**2 * self.m * self.Lpp**2  # Iz向转动惯量
        # 附加质量
        self.m_x = 

        self.m_y = 

        self.j_z =

        # 权重
        self.weight_u = weight_u
        self.weight_v = weight_v
        self.weight_r = weight_r

    def forward(self, seq_inputs, outputs):
        x_pred, y_pred = outputs[:, 0], outputs[:, 1]
        time = seq_inputs[:, -1, 0]
        psi = seq_inputs[:, -1, 1] # 航向
        u = seq_inputs[:, -1, 2] # 船舶速度
        r = seq_inputs[:, -1, 3] # 船舶角速度
        delta_L, np_L = seq_inputs[:, -1, 4], seq_inputs[:, -1, 5] * 15  # 左舵角，左转速 × 15（rpm）
        delta_R, np_R = seq_inputs[:, -1, 8], seq_inputs[:, -1, 9] * 15  # 右舵角，右转速 × 15（rpm）
        # 提取风速和风向
        Vw = seq_inputs[:, -1, 12]  # 风速（m/s）
        wind_dir = seq_inputs[:, -1, 13]  # 风向（T或R，单位为弧度）

        r_pred = r

        grads_x = torch.autograd.grad(x_pred, seq_inputs, grad_outputs=torch.ones_like(x_pred),
                                     retain_graph=True, create_graph=True)[0]
        grads_y = torch.autograd.grad(y_pred, seq_inputs, grad_outputs=torch.ones_like(y_pred),
                                     retain_graph=True, create_graph=True)[0]
        
        u_pred = grads_x[:, -1, 0]  # dx/dt
        v_pred = grads_y[:, -1, 0]  # dy/dt

        grads_u = torch.autograd.grad(u_pred, seq_inputs,grad_outputs=torch.ones_like(u_pred),
                                     retain_graph=True, create_graph=True)[0]
        grads_v = torch.autograd.grad(v_pred, seq_inputs,grad_outputs=torch.ones_like(v_pred),
                                     retain_graph=True, create_graph=True)[0]
        grads_r = torch.autograd.grad(r_pred, seq_inputs,grad_outputs=torch.ones_like(r_pred),
                                     retain_graph=True, create_graph=True)[0]

        u_dot = grads_u[:, -1, 0]
        v_dot = grads_v[:, -1, 0]
        r_dot = grads_r[:, -1, 3]

        U = torch.sqrt((u * torch.cos(psi) - v_pred * torch.sin(psi))**2 +
                       (u * torch.sin(psi) + v_pred * torch.cos(psi))**2 + 1e-8) #船体坐标系下船舶航速

        v_, r_ = v_pred / (U + 1e-6), r * self.Lpp / (U + 1e-6)
        beta = torch.atan2(-v_pred, u)
        betap = beta - self.xpp * r_

        tp0, Us = 
        tp = torch.where(u < Us, 0.04 + (tp0 - 0.04) * u / Us, tp0)

        wp0 = 
        wp = wp0 * torch.exp(-4 * betap**2)

        # 螺旋桨推力计算
        def calc_Jp(u, v, delta, np):
            return (u * torch.cos(delta) + v * torch.sin(delta)) * (1 - wp) / (np * self.Dp + 1e-6)

        Jp_L = calc_Jp(u, v_pred, delta_L, np_L)
        Jp_R = calc_Jp(u, v_pred, delta_R, np_R)

        kt_L = self.k0 + self.k1 * Jp_L + self.k2 * Jp_L**2
        kt_R = self.k0 + self.k1 * Jp_R + self.k2 * Jp_R**2

        FP_L = (1 - tp) * self.Rho * np_L**2 * self.Dp**4 * kt_L
        FP_R = (1 - tp) * self.Rho * np_R**2 * self.Dp**4 * kt_R

        Xuu=0;
        Xvv=0 ;
        Xvr=0 ;
        Xrr=0  ;    
        Xvvvv=0  ;
        
        Yv=-0  ;
        Yr=0 ;
        Yvvv=-0  ;
        Yvvr=-0 ;
        Yvrr=-0   ;   
        Yrrr=-0 ;
        
        Nv=-0 ;
        Nr=-0 ;
        Nvvv=0   ; 
        Nvvr=-0   ;
        Nvrr=-0     ;
        Nrrr=-0  ;

        X_hydro = 0.5 * self.Rho * self.Lpp * self.d * U**2 * (
            Xuu + Xvv * v_**2 + Xvr * v_ * r_ + Xrr * r_**2 + Xvvvv * v_**4)
        Y_hydro = 0.5 * self.Rho * self.Lpp * self.d * U**2 * (
            Yv * v_ + Yr * r_ + Yvvv * v_**3 + Yvvr * v_**2 * r_ + Yvrr * v_ * r_**2 + Yrrr * r_**3)
        N_hydro = 0.5 * self.Rho * self.Lpp**2 * self.d * U**2 * (
            Nv * v_ + Nr * r_ + Nvvv * v_**3 + Nvvr * v_**2 * r_ + Nvrr * v_ * r_**2 + Nrrr * r_**3)

        X_total = FP_L * torch.cos(delta_L) + FP_R * torch.cos(delta_R) + X_hydro
        Y_total = FP_L * torch.sin(delta_L) + FP_R * torch.sin(delta_R) + Y_hydro
        N_total = -FP_L * self.Lps * torch.sin(delta_L) + FP_R * self.Lps * torch.sin(delta_R) + N_hydro

        # 相对风向 βw = wind_dir - psi
        beta_w = wind_dir - psi
        # 空气密度与风力参数（你可以微调）
        rho_a = 
        Af = .0  # 前视面积 (估值)
        Al = .0  # 侧视面积 (估值)
        La = self.Lpp * 0.25  # 力臂长度
        C_Xw = 
        C_Yw = 
        C_Nw =

        # 风力分量计算
        X_wind = 0.5 * rho_a * Af * C_Xw * Vw**2 * torch.cos(beta_w)
        Y_wind = 0.5 * rho_a * Al * C_Yw * Vw**2 * torch.sin(beta_w)
        N_wind = 0.5 * rho_a * Al * C_Nw * La * Vw**2 * torch.sin(2 * beta_w)

        # 加入总力
        X_total += X_wind
        Y_total += Y_wind
        N_total += N_wind

        res_u = (self.m + self.m_x) * u_dot - X_total
        res_v = (self.m + self.m_y) * v_dot - Y_total
        res_r = (self.Iz + self.j_z) * r_dot - N_total

        loss_MMG_x = (res_u**2).mean()
        loss_MMG_y = (res_v**2).mean()
        loss_MMG_N = (res_r**2).mean()

        pde_total_loss = self.weight_u * loss_MMG_x + self.weight_v * loss_MMG_y + self.weight_r * loss_MMG_N

        return loss_MMG_x, loss_MMG_y, loss_MMG_N, pde_total_loss


