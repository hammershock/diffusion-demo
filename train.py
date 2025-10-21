# train.py
# 训练一个基础 DDPM（去噪扩散概率模型）

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


# -------------------------
# 简单 U-Net 网络
# -------------------------
class SimpleUNet(nn.Module):
    def __init__(self, channels=3, base_channels=64):
        super().__init__()
        self.inc = nn.Conv2d(channels, base_channels, 3, 1, 1)
        self.down1 = nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1)
        self.down2 = nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1)
        self.bot1 = nn.Conv2d(base_channels * 4, base_channels * 4, 3, 1, 1)
        self.up1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1)
        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1)
        self.outc = nn.Conv2d(base_channels, channels, 3, 1, 1)
        self.time_embed = nn.Sequential(
            nn.Linear(1, base_channels * 4),
            nn.ReLU(),
            nn.Linear(base_channels * 4, base_channels * 4),
        )

    def forward(self, x, t):
        # t: [B]
        t = t.float().unsqueeze(-1) / 1000.0
        t_emb = self.time_embed(t).unsqueeze(-1).unsqueeze(-1)
        x1 = F.relu(self.inc(x))
        x2 = F.relu(self.down1(x1))
        x3 = F.relu(self.down2(x2))
        x3 = x3 + t_emb
        x3 = F.relu(self.bot1(x3))
        x = F.relu(self.up1(x3))
        x = F.relu(self.up2(x))
        return self.outc(x)


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
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = ImageFolderDataset(dataset_path, image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

    model = SimpleUNet().to(device)
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

        if epoch % save_every == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f"ddpm_epoch_{epoch+1}.pt"))
    torch.save(model.state_dict(), os.path.join(save_dir, f"last.pt"))
    print("训练完成 ✅")


if __name__ == "__main__":
    train_ddpm()
