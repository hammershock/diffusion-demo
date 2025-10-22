# train.py
# 训练一个基础 DDPM（去噪扩散概率模型）

from email.policy import strict
import os
import glob
import math
import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# -------------------------
# 数据加载部分
# -------------------------
class ImageFolderDataset(torch.utils.data.Dataset):
    def __init__(self, root, image_size=128, exts=("jpg", "jpeg", "png", "webp")):
        self.files = []
        for e in exts:
            self.files += glob.glob(os.path.join(root, f"**/*.{e}"), recursive=True)
        if len(self.files) == 0:
            raise FileNotFoundError(
                f"在 {root} 下没有找到图像文件（支持扩展名: {exts}）"
            )
        self.image_size = image_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            return self.__getitem__((idx + 1) % len(self.files))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        img = img * 2.0 - 1.0
        img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(img)


# -------------------------
# 噪声调度器（β 线性调度）
# -------------------------
class LinearNoiseScheduler:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

# --- 替换 train.py 中原来的 SimpleUNet 为下方实现 -----------------

# 正弦时间嵌入（位置编码）
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t):  # t: [B]
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            torch.arange(half, device=device, dtype=torch.float32)
            * -(math.log(10000.0) / (half - 1))
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # [B, half]
        emb = torch.cat([args.sin(), args.cos()], dim=-1)   # [B, dim]
        return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, dropout=0.0):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.norm1  = nn.GroupNorm(32, in_ch)
        self.act1   = nn.SiLU()
        self.conv1  = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch)
        )

        self.norm2  = nn.GroupNorm(32, out_ch)
        self.act2   = nn.SiLU()
        self.dropout= nn.Dropout(dropout)
        self.conv2  = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):  # t_emb: [B, time_dim]
        h = self.conv1(self.act1(self.norm1(x)))
        # 注入时间（FiLM 的加法偏置）
        h = h + self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-2)
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        return h + self.skip(x)


class SelfAttention2D(nn.Module):
    def __init__(self, ch, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(32, ch)
        self.qkv  = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)  # [B, 3C, H, W]
        q, k, v = qkv.chunk(3, dim=1)

        # [B, heads, C//heads, HW]
        def reshape_heads(t):
            t = t.view(B, self.num_heads, C // self.num_heads, H * W)
            return t

        q, k, v = map(reshape_heads, (q, k, v))
        attn = torch.softmax(
            (q.transpose(2, 3) @ k) / math.sqrt(C // self.num_heads), dim=-1
        )  # [B, heads, HW, HW]
        out = (attn @ v.transpose(2, 3)).transpose(2, 3)  # [B, heads, C//heads, HW]
        out = out.contiguous().view(B, C, H * W).view(B, C, H, W)
        out = self.proj(out)
        return out + x


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, with_attn=False, dropout=0.0):
        super().__init__()
        self.res1 = ResBlock(in_ch,  out_ch, time_dim, dropout)
        self.res2 = ResBlock(out_ch, out_ch, time_dim, dropout)
        self.attn = SelfAttention2D(out_ch) if with_attn else nn.Identity()
        self.down = nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1)

    def forward(self, x, t_emb):
        x = self.res1(x, t_emb)
        x = self.res2(x, t_emb)
        x = self.attn(x)
        x_down = self.down(x)
        # 关键：把下采样后的特征作为 skip，保证与解码端输入 x 同尺寸
        skip = x_down
        return x_down, skip



class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, with_attn=False, dropout=0.0):
        super().__init__()
        # 注意：in_ch = 当前通道 + skip 通道
        self.res1 = ResBlock(in_ch,  out_ch, time_dim, dropout)
        self.res2 = ResBlock(out_ch, out_ch, time_dim, dropout)
        self.attn = SelfAttention2D(out_ch) if with_attn else nn.Identity()
        self.up   = nn.ConvTranspose2d(out_ch, out_ch, 4, stride=2, padding=1)

    def forward(self, x, skip, t_emb):
        assert x.shape[-2:] == skip.shape[-2:], f"UpBlock spatial mismatch: {x.shape[-2:]} vs {skip.shape[-2:]}"
        x = torch.cat([x, skip], dim=1)
        x = self.res1(x, t_emb)
        x = self.res2(x, t_emb)
        x = self.attn(x)
        x = self.up(x)
        return x


class SimpleUNet(nn.Module):
    """
    输入:  x [B,3,H,W], t [B]
    输出:  预测噪声 eps_hat，与训练/推理接口保持一致
    结构:  128 -> 64 -> 32 -> 16 分辨率，瓶颈带 Attention，Up/Down 两次
    """
    def __init__(self, channels=3, base_channels=64, dropout=0.0):
        super().__init__()
        ch = base_channels
        time_dim = ch * 4

        # 时间嵌入
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(ch),
            nn.Linear(ch, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # 输入卷积
        self.inc = nn.Conv2d(channels, ch, 3, padding=1)

        # Down: (128->64), (64->32), (32->16)
        self.down1 = DownBlock(ch,     ch,     time_dim, with_attn=False, dropout=dropout)
        self.down2 = DownBlock(ch,     ch*2,   time_dim, with_attn=False, dropout=dropout)
        self.down3 = DownBlock(ch*2,   ch*4,   time_dim, with_attn=True,  dropout=dropout)  # 16x16 加注意力

        # Bottleneck
        self.bot1 = ResBlock(ch*4, ch*4, time_dim, dropout)
        self.bot_attn = SelfAttention2D(ch*4)
        self.bot2 = ResBlock(ch*4, ch*4, time_dim, dropout)

        # Up: (16->32), (32->64), (64->128)
        self.up1  = UpBlock(in_ch=ch*4 + ch*4, out_ch=ch*2, time_dim=time_dim, with_attn=True,  dropout=dropout)
        self.up2  = UpBlock(in_ch=ch*2 + ch*2, out_ch=ch,   time_dim=time_dim, with_attn=False, dropout=dropout)
        self.up3  = UpBlock(in_ch=ch   + ch,   out_ch=ch,   time_dim=time_dim, with_attn=False, dropout=dropout)

        # 输出层
        self.out_norm = nn.GroupNorm(32, ch)
        self.out_act  = nn.SiLU()
        self.outc     = nn.Conv2d(ch, channels, 3, padding=1)

    def forward(self, x, t):
        # t: [B] => time embedding
        t_emb = self.time_embed(t)

        # encoder
        x0 = self.inc(x)                     # [B, ch, 128,128]
        d1, s1 = self.down1(x0, t_emb)       # -> [B, ch,   64,64], s1: [B, ch, 128,128]
        d2, s2 = self.down2(d1, t_emb)       # -> [B, 2ch,  32,32], s2: [B, ch,   64,64]
        d3, s3 = self.down3(d2, t_emb)       # -> [B, 4ch,  16,16], s3: [B, 2ch,  32,32]

        # bottleneck
        b  = self.bot1(d3, t_emb)
        b  = self.bot_attn(b)
        b  = self.bot2(b, t_emb)

        # decoder
        u1 = self.up1(b,  s3, t_emb)         # -> [B, 2ch, 32,32]
        u2 = self.up2(u1, s2, t_emb)         # -> [B, ch,  64,64]
        u3 = self.up3(u2, s1, t_emb)         # -> [B, ch, 128,128]

        out = self.outc(self.out_act(self.out_norm(u3)))
        return out
# --- 替换结束 ----------------------------------------------------



# -------------------------
# 训练过程
# -------------------------
def train_ddpm(
    dataset_path="/cache/hanmo/anime_face",
    image_size=128,
    batch_size=64,
    epochs=30,
    lr=1e-4,
    timesteps=1000,
    save_every=10,   # every 10 epoch
    save_dir="./checkpoints",
    ckpt=None, 
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = ImageFolderDataset(dataset_path, image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

    model = SimpleUNet().to(device)
    if ckpt is not None and os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True), strict=False)
        print(f"load model checkpoint from path: {ckpt}")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    noise_scheduler = LinearNoiseScheduler(timesteps=timesteps, device=device)

    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for imgs in pbar:
            imgs = imgs.to(device)
            b = imgs.shape[0]
            t = torch.randint(0, timesteps, (b,), device=device).long()
            noise = torch.randn_like(imgs)
            x_noisy = noise_scheduler.q_sample(imgs, t, noise)
            noise_pred = model(x_noisy, t)
            loss = F.mse_loss(noise_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())

        # if epoch % save_every == 0:
        #     torch.save(model.state_dict(), os.path.join(save_dir, f"ddpm_epoch_{epoch+1}.pt"))
        torch.save(model.state_dict(), os.path.join(save_dir, f"last.pt"))  # save every epoch
    print("训练完成 ✅")


if __name__ == "__main__":
    train_ddpm(ckpt="./checkpoints/last.pt")
