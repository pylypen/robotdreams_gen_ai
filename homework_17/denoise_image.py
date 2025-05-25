import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import io
import argparse
import torchvision.utils as vutils
import torch
import torch.nn as nn
import random
from PIL import Image, ImageFilter
from torchvision.transforms import functional as F


class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))
    
def add_random_noise(img_tensor):
    # Мета цієї функції — імітувати реальні сценарії пошкодження зображень.
    mode = random.choice(["gaussian", "s&p", "occlusion", "blur", "jpeg", "combo"])

    if mode == "gaussian": # традиційний шум камер/датчиків.
        noise = torch.randn_like(img_tensor) * 0.1
        return torch.clamp(img_tensor + noise, 0., 1.)

    elif mode == "s&p": # імпульсний шум, схожий на артефакти передавання даних.
        noisy = img_tensor.clone()
        prob = 0.02
        rand = torch.rand_like(noisy)
        noisy[rand < prob] = 0
        noisy[rand > 1 - prob] = 1
        return noisy

    elif mode == "occlusion": # приховані частини (наприклад, об’єкти в кадрі).
        noisy = img_tensor.clone()
        x = random.randint(0, noisy.shape[2] - 128)
        y = random.randint(0, noisy.shape[1] - 128)
        noisy[:, y:y+128, x:x+128] = 0
        return noisy

    elif mode == "blur": # розмиття
        pil_img = F.to_pil_image(img_tensor)
        blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=random.uniform(1, 3)))
        return F.to_tensor(blurred)

    elif mode == "jpeg": # втрати якості після стиснення.
        pil_img = F.to_pil_image(img_tensor)
        buffer = io.BytesIO()
        pil_img.save(buffer, format='JPEG', quality=random.randint(10, 30))
        jpeg_img = Image.open(buffer)
        return F.to_tensor(jpeg_img)

    elif mode == "combo": # складніші, комбінаційні спотворення.
        img = add_random_noise(img_tensor.clone())
        return add_random_noise(img)

    else:
        return img_tensor

# Аргументи з командного рядка
parser = argparse.ArgumentParser(description="Denoise a single image using a trained autoencoder")
parser.add_argument("input", type=str, help="Path to input image")
parser.add_argument("--output", type=str, default="output.png", help="Path to save the denoised image")
parser.add_argument("--checkpoint", type=str, default="checkpoint.pth", help="Path to model checkpoint")
parser.add_argument("--add_noise", action="store_true", help="If set, add synthetic noise to the input image")
parser.add_argument("--save_all", action="store_true", help="If set, save original, noisy and denoised images side by side")
args = parser.parse_args()

# Пристрій
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Трансформація (відповідає тренуванню)
transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor()
])

# Завантаження зображення
original_image = Image.open(args.input).convert('RGB')
original_tensor = transform(original_image).unsqueeze(0).to(device)

# Додавання шуму (якщо вказано)
if args.add_noise:
    from torchvision.transforms.functional import to_pil_image
    noisy_tensor = add_random_noise(original_tensor.squeeze(0)).unsqueeze(0).to(device)
else:
    noisy_tensor = original_tensor.clone()

# Завантаження моделі
model = DenoisingAutoencoder().to(device)
checkpoint = torch.load(args.checkpoint, map_location=device)
model.load_state_dict(checkpoint['model'])
model.eval()

# Денойзинг
with torch.no_grad():
    denoised_tensor = model(noisy_tensor).squeeze(0).cpu()

# Збереження виходу
output_image = transforms.ToPILImage()(denoised_tensor)
output_image.save(args.output)
# print(f"Збережено відфільтроване зображення")

# Якщо вказано, зберігаємо порівняльну картинку
if args.save_all:
    # Знімаємо batch-дим для збереження
    original = original_tensor.squeeze(0).cpu()
    noisy = noisy_tensor.squeeze(0).cpu()
    denoised = denoised_tensor

    comparison = torch.stack([original, noisy, denoised], dim=0)  # [3, C, H, W]
    vutils.save_image(comparison, "comparison_triplet.png", nrow=3)
    print("Image saved as comparison_triplet.png")
