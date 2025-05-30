import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from scipy import ndimage

# ================================
# Helper functions
# ================================

def add_gaussian_noise(img, mean=0, std=0.1):
    """Add Gaussian noise to an image"""
    noise = np.random.normal(mean, std, img.shape)
    noisy_img = img + noise
    return np.clip(noisy_img, 0, 1)  # Clip to [0,1] interval

def calculate_metrics(original, restored):
    """Calculate PSNR (distortion) and SSIM (perception-related)"""
    psnr_value = psnr(original, restored)
    ssim_value = ssim(original, restored, data_range=1.0)
    return psnr_value, ssim_value

def plot_results(original, noisy, restored_images, titles, metrics=None):
    """Plot original, noisy and restored images with metrics"""
    n = len(restored_images) + 2
    plt.figure(figsize=(15, 5))
    
    # Original and noisy image
    plt.subplot(1, n, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original')
    plt.axis('off')
    
    plt.subplot(1, n, 2)
    plt.imshow(noisy, cmap='gray')
    plt.title('Noisy')
    plt.axis('off')
    
    # Restored images
    for i, (img, title) in enumerate(zip(restored_images, titles)):
        plt.subplot(1, n, i+3)
        plt.imshow(img, cmap='gray')
        if metrics:
            plt.title(f"{title}\nPSNR: {metrics[i][0]:.2f}, SSIM: {metrics[i][1]:.2f}")
        else:
            plt.title(title)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('AU/Information Learning Theory/Project/Information learning theories/python/restoration_results.png', dpi=300)
    plt.show()

# ================================
# Datasets and dataloaders
# ================================

class NoisyImageDataset(Dataset):
    def __init__(self, clean_images, noisy_images, transform=None):
        self.clean_images = clean_images
        self.noisy_images = noisy_images
        self.transform = transform
        
    def __len__(self):
        return len(self.clean_images)
    
    def __getitem__(self, idx):
        clean = self.clean_images[idx]
        noisy = self.noisy_images[idx]
        
        if self.transform:
            clean = self.transform(clean)
            noisy = self.transform(noisy)
            
        return clean, noisy

# ================================
# Simple models for restoration
# ================================

class SimpleDenoiser(nn.Module):
    def __init__(self):
        super(SimpleDenoiser, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        out = out + residual  # Skip connection
        return out

# ================================
# GAN for restoration
# ================================

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # U-Net like structure
        # Encoder
        self.enc1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)
        self.enc2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.enc3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.enc4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        
        # Decoder
        self.dec1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.dec2 = nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1)
        self.dec3 = nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1)
        self.dec4 = nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1)
        
        self.relu = nn.ReLU(inplace=True)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        # Encoder
        e1 = self.leaky_relu(self.enc1(x))
        e2 = self.leaky_relu(self.enc2(e1))
        e3 = self.leaky_relu(self.enc3(e2))
        e4 = self.leaky_relu(self.enc4(e3))
        
        # Decoder with skip connections
        d1 = self.relu(self.dec1(e4))
        d1 = torch.cat([d1, e3], 1)
        d2 = self.relu(self.dec2(d1))
        d2 = torch.cat([d2, e2], 1)
        d3 = self.relu(self.dec3(d2))
        d3 = torch.cat([d3, e1], 1)
        d4 = self.tanh(self.dec4(d3))
        
        return d4

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        
        # Modified the last layer to output a single value
        # After 4 stride-2 convolutions, our 128x128 image is reduced to 8x8
        # Therefore, we use kernel_size=8 to obtain a 1x1 output
        self.conv5 = nn.Conv2d(512, 1, kernel_size=8, stride=1, padding=0)
        
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.batch_norm2 = nn.BatchNorm2d(128)
        self.batch_norm3 = nn.BatchNorm2d(256)
        self.batch_norm4 = nn.BatchNorm2d(512)
        
    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.batch_norm2(self.conv2(x)))
        x = self.leaky_relu(self.batch_norm3(self.conv3(x)))
        x = self.leaky_relu(self.batch_norm4(self.conv4(x)))
        x = self.sigmoid(self.conv5(x))
        return x

# ================================
# Training functions
# ================================

def train_simple_denoiser(model, train_loader, num_epochs=50):
    """Train a simple denoiser with MSE loss"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for clean, noisy in train_loader:
            clean, noisy = clean.to(device), noisy.to(device)
            
            optimizer.zero_grad()
            outputs = model(noisy)
            loss = criterion(outputs, clean)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.6f}')
    
    return model

def train_gan_denoiser(generator, discriminator, train_loader, num_epochs=50):
    """Train a GAN-based denoiser"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    
    criterion_gan = nn.BCELoss()
    criterion_pixel = nn.L1Loss()  # L1 loss for perceptual quality
    
    optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    lambda_pixel = 100  # Weighting of pixel-wise loss vs. GAN loss
    
    for epoch in range(num_epochs):
        for clean, noisy in train_loader:
            clean, noisy = clean.to(device), noisy.to(device)
            batch_size = clean.size(0)
            
            # Ground truths
            valid = torch.ones((batch_size, 1, 1, 1), requires_grad=False).to(device)
            fake = torch.zeros((batch_size, 1, 1, 1), requires_grad=False).to(device)
            
            # -----------------
            #  Train Generator
            # -----------------
            optimizer_g.zero_grad()
            
            # Generate a restored image
            gen_img = generator(noisy)
            
            # GAN loss
            pred_fake = discriminator(gen_img)
            loss_gan = criterion_gan(pred_fake, valid)
            
            # Pixel-wise loss
            loss_pixel = criterion_pixel(gen_img, clean)
            
            # Total loss
            loss_g = loss_gan + lambda_pixel * loss_pixel
            
            loss_g.backward()
            optimizer_g.step()
            
            # -----------------
            #  Train Discriminator
            # -----------------
            optimizer_d.zero_grad()
            
            # Real images
            pred_real = discriminator(clean)
            loss_real = criterion_gan(pred_real, valid)
            
            # Fake images
            pred_fake = discriminator(gen_img.detach())
            loss_fake = criterion_gan(pred_fake, fake)
            
            # Total loss
            loss_d = 0.5 * (loss_real + loss_fake)
            
            loss_d.backward()
            optimizer_d.step()
            
        print(f'Epoch {epoch+1}/{num_epochs}, G Loss: {loss_g.item():.6f}, D Loss: {loss_d.item():.6f}')
    
    return generator

# ================================
# Main function
# ================================

def main():
    # Load original image
    try:
        # Load original image and convert to grayscale if needed
        img = Image.open('AU/Information Learning Theory/Project/Information learning theories/python/original.jpg')
        if img.mode != 'L':  # If not already grayscale
            img = img.convert('L')
        
        # Convert to numpy array and normalize to [0,1]
        original = np.array(img).astype(np.float32) / 255.0
        
        # Resize the image if needed to fit the network architecture
        # GAN architecture requires dimensions divisible by 16 (due to 4 downsampling layers)
        target_size = (128, 128)  # Choose a size divisible by 16
        if original.shape[0] % 16 != 0 or original.shape[1] % 16 != 0:
            img = img.resize(target_size)
            original = np.array(img).astype(np.float32) / 255.0
            print(f"The image has been resized to {target_size} to fit the network architecture")
        
        print(f"Loaded image with size: {original.shape}")
    except Exception as e:
        print(f"Error loading image: {e}")
        print("Generating a synthetic image instead...")
        
        # Fallback to synthetic image if original.jpg cannot be loaded
        size = 128
        original = np.zeros((size, size)).astype(np.float32)
        y, x = np.ogrid[-size/2:size/2, -size/2:size/2]
        mask = x**2 + y**2 <= (size/3)**2
        original[mask] = 1.0
    
    # Add Gaussian noise
    noisy = add_gaussian_noise(original, std=0.1)
    
    # Save original and noisy image for reference
    plt.imsave('AU/Information Learning Theory/Project/Information learning theories/python/original_image.png', original, cmap='gray')
    plt.imsave('AU/Information Learning Theory/Project/Information learning theories/python/noisy_image.png', noisy, cmap='gray')
    
    # Convert to tensors
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Explicitly convert to float32
        transforms.Lambda(lambda x: x.float()),
    ])
    
    # Create a mini-dataset for training
    clean_images = [original for _ in range(100)]
    noisy_images = [add_gaussian_noise(original, std=0.1) for _ in range(100)]
    
    dataset = NoisyImageDataset(clean_images, noisy_images, transform)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # 1. Gaussian filter (classical method)
    gaussian_restored = ndimage.gaussian_filter(noisy, sigma=1)
    
    # 2. Train a simple CNN denoiser (optimization of MSE - distortion-oriented)
    simple_denoiser = SimpleDenoiser()
    # Explicitly convert model parameters to float32
    simple_denoiser = simple_denoiser.float()
    simple_denoiser = train_simple_denoiser(simple_denoiser, train_loader, num_epochs=5)
    
    # Test the simple denoiser
    with torch.no_grad():
        # Ensure tensor is float32
        noisy_tensor = transform(noisy).unsqueeze(0).float()
        simple_restored = simple_denoiser(noisy_tensor).squeeze(0).squeeze(0).cpu().numpy()
    
    # 3. Train a GAN-based denoiser (perception-oriented)
    generator = Generator().float()  # Explicit float32
    discriminator = Discriminator().float()  # Explicit float32
    generator = train_gan_denoiser(generator, discriminator, train_loader, num_epochs=5)
    
    # Test the GAN denoiser
    with torch.no_grad():
        gan_restored = generator(noisy_tensor).squeeze(0).squeeze(0).cpu().numpy()
    
    # Calculate metrics
    metrics = []
    for img in [gaussian_restored, simple_restored, gan_restored]:
        psnr_val, ssim_val = calculate_metrics(original, img)
        metrics.append((psnr_val, ssim_val))
    
    # Plot results
    plot_results(
        original, 
        noisy, 
        [gaussian_restored, simple_restored, gan_restored],
        ["Gaussian Filter", "MSE-Denoiser", "GAN-Denoiser"],
        metrics
    )
    
    # Create perception-distortion curve (trade-off)
    plt.figure(figsize=(8, 6))
    plt.scatter([m[0] for m in metrics], [m[1] for m in metrics], s=100)
    for i, (title, metric) in enumerate(zip(["Gaussian", "MSE-CNN", "GAN"], metrics)):
        plt.annotate(title, (metric[0], metric[1]), fontsize=12)
    plt.xlabel('PSNR (dB) - Distortion', fontsize=14)
    plt.ylabel('SSIM - Perception Quality', fontsize=14)
    plt.title('Perception-Distortion Trade-off', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('AU/Information Learning Theory/Project/Information learning theories/python/perception_distortion_tradeoff.png', dpi=1200)
    plt.show()
    
    print("Results:")
    print(f"{'Method':<15} {'PSNR (dB)':<10} {'SSIM':<10}")
    print("-" * 35)
    print(f"{'Gaussian':<15} {metrics[0][0]:<10.2f} {metrics[0][1]:<10.4f}")
    print(f"{'MSE-CNN':<15} {metrics[1][0]:<10.2f} {metrics[1][1]:<10.4f}")
    print(f"{'GAN':<15} {metrics[2][0]:<10.2f} {metrics[2][1]:<10.4f}")

if __name__ == "__main__":
    main()
