from opensora.models.vae.vae import VideoAutoencoderPipeline
import torch

vae = VideoAutoencoderPipeline.from_pretrained("hpcai-tech/OpenSora-VAE-v1.2")

x = torch.randn((1, 3, 10, 64, 64))
# x = batch["video"].to(device, dtype)  # [B, C, T, H, W]
# == vae encoding & decoding ===
z, posterior, x_z = vae.encode(x)
x_rec, x_z_rec = vae.decode(z, num_frames=x.size(2))
x_ref = vae.spatial_vae.decode(x_z)

print(x_ref.shape)