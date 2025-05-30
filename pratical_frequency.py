import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from PIL import Image
import cv2
from scipy import ndimage
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import os

# Use SSIM as a replacement for NIQE - both SSIM and MS-SSIM are mentioned by Blau and Michaeli
# as perceptual metrics in some contexts
def calculate_perceptual_score(img1, img2=None):
    """
    Calculate a perceptual score for the image
    Higher values indicate better perceptual quality
    """
    # If we don't have a reference image, calculate a simplified 
    # perceptual score based on edge information
    if img2 is None:
        # Calculate edge intensity (higher values indicate sharper edges)
        edges = ndimage.sobel(img1)
        edge_intensity = np.mean(np.abs(edges))
        
        # Calculate local variance (higher values indicate more texture)
        local_var = ndimage.generic_filter(img1, np.var, size=5)
        var_score = np.mean(local_var)
        
        # Combine into a perceptual score
        # Higher value is assumed to correlate with better perceptual quality
        return 0.5 * edge_intensity + 0.5 * var_score
    else:
        # If we have a reference image, use SSIM
        return ssim(img1, img2, data_range=1.0)

def add_gaussian_noise(img, mean=0, sigma=0.1):
    """Add Gaussian noise to an image"""
    noise = np.random.normal(mean, sigma, img.shape)
    noisy_img = img + noise
    return np.clip(noisy_img, 0, 1)

def apply_frequency_filter(img, filter_type, cutoff=0.1, bandwidth=None):
    """
    Apply a frequency domain filter to an image
    
    Parameters:
    - img: Input image [0,1]
    - filter_type: 'lowpass', 'highpass', or 'bandpass'
    - cutoff: Cut-off frequency (normalized [0,1])
    - bandwidth: Bandwidth for bandpass filter (normalized [0,1])
    
    Returns:
    - Filtered image
    """
    # Convert to frequency domain
    f = fft2(img)
    fshift = fftshift(f)
    
    # Create frequency domain coordinates
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    
    # Create meshgrid for the frequency domain
    y, x = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
    r = np.sqrt(x*x + y*y) / (rows/2)  # Normalized radius
    
    # Create mask based on filter type
    if filter_type == 'lowpass':
        mask = r <= cutoff
    elif filter_type == 'highpass':
        mask = r >= cutoff
    elif filter_type == 'bandpass':
        if bandwidth is None:
            bandwidth = cutoff / 2
        mask = (r >= cutoff - bandwidth/2) & (r <= cutoff + bandwidth/2)
    else:
        raise ValueError("Unknown filter type. Use 'lowpass', 'highpass', or 'bandpass'")
    
    # Save the filter mask for visualization
    filter_mask = mask.astype(float)
    
    # Apply filter
    fshift_filtered = fshift * mask
    
    # Convert back to image domain
    f_ishift = ifftshift(fshift_filtered)
    img_filtered = np.real(ifft2(f_ishift))
    
    # Normalize the result to [0,1]
    img_filtered = np.clip(img_filtered, 0, 1)
    
    return img_filtered, filter_mask, fshift, fshift_filtered

def plot_frequency_results(original, noisy, filtered_results, metric_results, output_path):
    """
    Plot original image, noisy image, and filtered results with corresponding frequency domain representations
    
    Parameters:
    - original: Original image
    - noisy: Noisy image
    - filtered_results: List of (filtered_image, filter_mask, fshift_original, fshift_filtered) tuples
    - metric_results: Dictionary results with metrics for each filter
    - output_path: Path to output image
    """
    n_filters = len(filtered_results)
    filter_names = ['Original', 'Noisy'] + [result[0] for result in filtered_results]
    
    # Create figure
    fig, axes = plt.subplots(3, n_filters + 2, figsize=(4*(n_filters + 2), 12))
    
    # Plot original and noisy image in the first row
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(noisy, cmap='gray')
    axes[0, 1].set_title('Noisy')
    axes[0, 1].axis('off')
    
    # Calculate FFT of original and noisy image for the row with FFT amplitude
    f_original = fftshift(fft2(original))
    f_noisy = fftshift(fft2(noisy))
    
    # Plot FFT amplitude spectrum of original and noisy image in the second row
    axes[1, 0].imshow(np.log1p(np.abs(f_original)), cmap='viridis')
    axes[1, 0].set_title('Original FFT Amplitude')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(np.log1p(np.abs(f_noisy)), cmap='viridis')
    axes[1, 1].set_title('Noisy FFT Amplitude')
    axes[1, 1].axis('off')
    
    # Plot FFT difference between original and noisy
    fft_diff = np.abs(f_noisy) - np.abs(f_original)
    im = axes[2, 0].imshow(fft_diff, cmap='coolwarm')
    axes[2, 0].set_title('FFT Difference\n(Noisy - Original)')
    axes[2, 0].axis('off')
    plt.colorbar(im, ax=axes[2, 0], fraction=0.046, pad=0.04)
    
    # Second row in column 1 is empty (empty cell below noisy image)
    axes[2, 1].axis('off')
    
    # Plot filtered results and their frequency domain representations
    for i, ((name, filtered_img, filter_mask, fshift_original, fshift_filtered), metrics) in enumerate(zip(filtered_results, metric_results[1:])):  # Skip noisy in metric_results
        col = i + 2  # Start from the third column
        
        # Plot filtered image in the first row
        axes[0, col].imshow(filtered_img, cmap='gray')
        axes[0, col].set_title(f'{name}\nPSNR: {metrics["psnr"]:.2f}, Perc: {metrics["perceptual"]:.2f}')
        axes[0, col].axis('off')
        
        # Plot FFT amplitude spectrum in the second row
        axes[1, col].imshow(np.log1p(np.abs(fshift_filtered)), cmap='viridis')
        axes[1, col].set_title(f'{name} FFT Amplitude')
        axes[1, col].axis('off')
        
        # Plot filter mask in the third row
        im = axes[2, col].imshow(filter_mask, cmap='coolwarm')
        axes[2, col].set_title(f'{name} Filter Mask')
        axes[2, col].axis('off')
        plt.colorbar(im, ax=axes[2, col], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def plot_perception_distortion(metric_results, output_path):
    """
    Plot perception-distortion tradeoff for the different frequency filters
    
    Parameters:
    - metric_results: Dictionary results with metrics for each filter
    - output_path: Path to output image
    """
    plt.figure(figsize=(10, 8))
    
    # Distortion (x-axis) is PSNR (higher = better)
    # Perception (y-axis) is perceptual score (higher = better)
    x = [m["psnr"] for m in metric_results]
    y = [m["perceptual"] for m in metric_results]
    labels = [m["name"] for m in metric_results]
    
    # Plot points
    plt.scatter(x, y, s=100)
    
    # Add labels to each point
    for i, label in enumerate(labels):
        plt.annotate(label, (x[i], y[i]), fontsize=12, xytext=(5, 5), 
                    textcoords='offset points')
    
    plt.xlabel('PSNR (dB) - Distortion ↑', fontsize=14)
    plt.ylabel('Perceptual Quality ↑', fontsize=14)
    plt.title('Perception-Distortion Tradeoff for Frequency Domain Filtering', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def main():
    # Create output directory with correct structure
    output_dir = "AU/Information Learning Theory/Project/Information learning theories/python/frequency/"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image or use a demo image if no image is specified
    try:
        img = Image.open('AU/Information Learning Theory/Project/Information learning theories/python/original.jpg')
        img_gray = np.array(img.convert('L')).astype(np.float32) / 255.0
        print(f"Loaded image with size: {img_gray.shape}")
    except:
        print("Could not load 'original.jpg', generating a demo image")
        # Generate a demo image (e.g., a sinusoidal wave)
        size = 512
        img_gray = np.zeros((size, size))
        x = np.linspace(0, 10, size)
        y = np.linspace(0, 10, size)
        X, Y = np.meshgrid(x, y)
        img_gray = (np.sin(X) + np.sin(Y) + 2) / 4.0
    
    # Add noise to the image
    noisy_img = add_gaussian_noise(img_gray, sigma=0.1)
    
    # Save original and noisy image
    plt.imsave(os.path.join(output_dir, "original.png"), img_gray, cmap='gray')
    plt.imsave(os.path.join(output_dir, "noisy.png"), noisy_img, cmap='gray')
    
    # Apply different frequency domain filters
    filters_to_apply = [
        ('Lowpass', 'lowpass', 0.2, None),
        ('Highpass', 'highpass', 0.2, None),
        ('Bandpass', 'bandpass', 0.3, 0.1)
    ]
    
    filtered_results = []
    metric_results = []
    
    # Calculate metrics for noisy image
    noisy_psnr = psnr(img_gray, noisy_img, data_range=1.0)
    noisy_ssim = ssim(img_gray, noisy_img, data_range=1.0)
    noisy_perceptual = calculate_perceptual_score(noisy_img)
    
    print(f"Noisy Image - PSNR: {noisy_psnr:.2f}, SSIM: {noisy_ssim:.4f}, Perceptual: {noisy_perceptual:.4f}")
    
    # Add noisy metrics
    metric_results.append({
        "name": "Noisy",
        "psnr": noisy_psnr,
        "ssim": noisy_ssim,
        "perceptual": noisy_perceptual
    })
    
    # Apply each filter and calculate metrics
    for name, filter_type, cutoff, bandwidth in filters_to_apply:
        # Apply filter
        filtered_img, filter_mask, fshift_original, fshift_filtered = apply_frequency_filter(
            noisy_img, filter_type, cutoff, bandwidth
        )
        
        # Calculate metrics
        filtered_psnr = psnr(img_gray, filtered_img, data_range=1.0)
        filtered_ssim = ssim(img_gray, filtered_img, data_range=1.0)
        filtered_perceptual = calculate_perceptual_score(filtered_img)
        
        print(f"{name} Filter - PSNR: {filtered_psnr:.2f}, SSIM: {filtered_ssim:.4f}, Perceptual: {filtered_perceptual:.4f}")
        
        # Save results
        filtered_results.append((name, filtered_img, filter_mask, fshift_original, fshift_filtered))
        metric_results.append({
            "name": name,
            "psnr": filtered_psnr,
            "ssim": filtered_ssim,
            "perceptual": filtered_perceptual
        })
        
        # Save individual images
        plt.imsave(os.path.join(output_dir, f"{name.lower()}_filtered.png"), filtered_img, cmap='gray')
    
    # Plot results
    comparison_path = os.path.join(output_dir, "frequency_filtering_comparison.png")
    plot_frequency_results(img_gray, noisy_img, filtered_results, metric_results, comparison_path)
    
    # Plot perception-distortion tradeoff
    pd_path = os.path.join(output_dir, "perception_distortion_frequency.png")
    plot_perception_distortion(metric_results, pd_path)
    
    print(f"Results saved in directory: {output_dir}")
    print(f"Comparison image: {comparison_path}")
    print(f"Perception-Distortion plot: {pd_path}")

if __name__ == "__main__":
    main()
