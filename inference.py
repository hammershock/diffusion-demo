# inference.py
# 使用训练好的 DDPM 模型生成图像

import os
import math
import torch
from torchvision.utils import save_image
from tqdm import tqdm
from train import SimpleUNet, LinearNoiseScheduler  # 直接复用train.py定义的类


@torch.no_grad()
def p_sample(model, x_t, t, scheduler):
    betas = scheduler.betas
    alphas = scheduler.alphas
    alphas_cumprod = scheduler.alphas_cumprod
    alphas_cumprod_prev = scheduler.alphas_cumprod_prev

    t = t.long()
    beta_t = betas[t].view(-1, 1, 1, 1)
    alpha_t = alphas[t].view(-1, 1, 1, 1)
    alpha_bar_t = alphas_cumprod[t].view(-1, 1, 1, 1)
    alpha_bar_t_prev = alphas_cumprod_prev[t].view(-1, 1, 1, 1)

    # 预测噪声
    eps = model(x_t, t)

    # 均值项（Ho et al. 公式 11）
    mean = (1.0 / torch.sqrt(alpha_t)) * (x_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * eps)

    # 后验方差（Ho et al. 公式 7）
    posterior_var = beta_t * (1 - alpha_bar_t_prev) / (1 - alpha_bar_t)
    posterior_std = torch.sqrt(torch.clamp(posterior_var, min=1e-20))

    # t=0 不加噪声
    nonzero_mask = (t != 0).float().view(-1, 1, 1, 1)
    noise = torch.randn_like(x_t)
    x_prev = mean + nonzero_mask * posterior_std * noise
    return x_prev



@torch.no_grad()
def p_sample_loop(model, scheduler, shape, device, save_dir=None, save_interval=100):
    """
    从纯噪声开始反向生成图像
    """
    timesteps = scheduler.timesteps
    img = torch.randn(shape, device=device)

    os.makedirs(save_dir, exist_ok=True) if save_dir else None

    p_bar = tqdm(reversed(range(timesteps)), total=timesteps)
    for i in p_bar:
        t = torch.tensor([i] * shape[0], device=device)
        img = p_sample(model, img, t, scheduler)

        if save_dir and (i % save_interval == 0 or i == 0):
            save_image((img.clamp(-1, 1) + 1) / 2, os.path.join(save_dir, f"step_{i:04d}.png"))

    return img


def sample_ddpm(
    ckpt_path="checkpoints/last.pt",
    save_dir="./samples",
    timesteps=1000,
    image_size=128,
    num_images=8,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleUNet().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True), strict=False)
    model.eval()

    scheduler = LinearNoiseScheduler(timesteps=timesteps, device=device)

    # 生成 num_images 张图
    imgs = p_sample_loop(model, scheduler, shape=(num_images, 3, image_size, image_size),
                         device=device, save_dir=save_dir, save_interval=100)

    # 保存最终合成图
    final_path = os.path.join(save_dir, "sample.png")
    save_image((imgs.clamp(-1, 1) + 1) / 2, final_path, nrow=int(math.sqrt(num_images)))
    print(f"✅ 采样完成，图像已保存到 {final_path}")


if __name__ == "__main__":
    sample_ddpm()
