<!-- adding gaussian noise  -->

### `add_gaussian_noise`

<details>
    
```python
def add_gaussian_noise(img, mean=0, std=0.1):
    """Add Gaussian noise to an image"""
    noise = np.random.normal(mean, std, img.shape)
    noisy_img = img + noise
    return np.clip(noisy_img, 0, 1)  # Clip to [0,1] interval
```

This function adds Gaussian (normal) noise to an image. Below is a breakdown of each line:

```python
def add_gaussian_noise(img, mean=0, std=0.1):
```

* **Defines the function** `add_gaussian_noise` which takes three arguments:

  * `img`: the input image (assumed to be a NumPy array with values in $\[0, 1]\$),
  * `mean`: the mean of the Gaussian noise (default is 0),
  * `std`: the standard deviation of the noise (default is 0.1).

```python
    noise = np.random.normal(mean, std, img.shape)
```

* **Generates a noise matrix** with the same shape as the image.
* The values are drawn from a Gaussian (normal) distribution with specified `mean` and `std`.

```python
    noisy_img = img + noise
```

* **Adds the generated noise** to the original image element-wise.
* This may result in pixel values that go outside the valid range of $\[0, 1]\$.

```python
    return np.clip(noisy_img, 0, 1)  # Clip to [0,1] interval
```

* **Clips the result** so that all pixel values stay within the valid range $\[0, 1]\$.
* Any values below 0 become 0; values above 1 become 1.

---

This function is commonly used to simulate noise in image processing experiments.
</details>

### calculate_metrics

<details>
    
```python
def calculate_metrics(original, restored):
    """Calculate PSNR (distortion) and SSIM (perception-related)"""
    psnr_value = psnr(original, restored)
    ssim_value = ssim(original, restored, data_range=1.0)
    return psnr_value, ssim_value
```

This function calculates distortion and perception-related metrics for comparing an original image and a restored version.

```python
def calculate_metrics(original, restored):
```

* **Defines the function** `calculate_metrics` which takes two arguments:

  * `original`: the original reference image.
  * `restored`: the reconstructed or processed image to compare with.

```python
    psnr_value = psnr(original, restored)
```

* **Computes the PSNR (Peak Signal-to-Noise Ratio)** between the original and restored images.
* PSNR is a distortion metric; a higher value indicates that the restored image is more similar to the original in pixel-wise terms.

```python
    ssim_value = ssim(original, restored, data_range=1.0)
```

* **Computes the SSIM (Structural Similarity Index Measure)**.
* SSIM is a perceptual metric that considers structural information, luminance, and contrast.
* `data_range=1.0` tells the function that the images are normalized to the range $\[0, 1]\$.

```python
    return psnr_value, ssim_value
```

* **Returns** the two computed metrics as a tuple: `(PSNR, SSIM)`.
* These values can be used to evaluate the quality of a restoration algorithm from both distortion and perceptual perspectives.
    
</details>

### plot_results

<details>

This function visualizes the original, noisy, and restored images along with their associated metrics.

```python
def plot_results(original, noisy, restored_images, titles, metrics=None):
```

* Defines a function that plots the original, noisy, and multiple restored images.
* Takes in optional metrics (like PSNR and SSIM) for annotated display.

```python
    n = len(restored_images) + 2
    plt.figure(figsize=(15, 5))
```

* Computes total number of subplots (`n`) and initializes a figure of size 15x5.

```python
    plt.subplot(1, n, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original')
    plt.axis('off')
```

* Plots the original image in the first subplot without axis labels.

```python
    plt.subplot(1, n, 2)
    plt.imshow(noisy, cmap='gray')
    plt.title('Noisy')
    plt.axis('off')
```

* Plots the noisy image in the second subplot.

```python
    for i, (img, title) in enumerate(zip(restored_images, titles)):
        plt.subplot(1, n, i+3)
        plt.imshow(img, cmap='gray')
        if metrics:
            plt.title(f"{title}\nPSNR: {metrics[i][0]:.2f}, SSIM: {metrics[i][1]:.2f}")
        else:
            plt.title(title)
        plt.axis('off')
```

* Loops through restored images and titles to plot each one.
* If `metrics` are provided, it displays PSNR and SSIM under each title.

```python
    plt.tight_layout()
    plt.savefig('AU/Information Learning Theory/Project/Information learning theories/python/restoration_results.png', dpi=300)
    plt.show()
```

* Adjusts spacing, saves the figure as a PNG, and displays it.

</details>

### NoisyImageDataset Class

<details>

This custom PyTorch `Dataset` handles pairs of clean and noisy images for training or evaluation purposes.

```python
class NoisyImageDataset(Dataset):
```

* Declares a class that inherits from `torch.utils.data.Dataset`, allowing it to work with PyTorch `DataLoader`.

```python
    def __init__(self, clean_images, noisy_images, transform=None):
        self.clean_images = clean_images
        self.noisy_images = noisy_images
        self.transform = transform
```

* **Constructor (`__init__`)**:

  * `clean_images`: List or array of clean (reference) images.
  * `noisy_images`: Corresponding list or array of noisy (distorted) versions.
  * `transform`: Optional transform function (e.g., converting to tensor, normalization) applied to both clean and noisy images.

```python
    def __len__(self):
        return len(self.clean_images)
```

* **Returns** the total number of samples in the dataset (same length as clean images).

```python
    def __getitem__(self, idx):
        clean = self.clean_images[idx]
        noisy = self.noisy_images[idx]
```

* Retrieves the `clean` and `noisy` image pair at index `idx`.

```python
        if self.transform:
            clean = self.transform(clean)
            noisy = self.transform(noisy)
```

* Applies any specified transformations to both the clean and noisy images (e.g., data augmentation, tensor conversion).

```python
        return clean, noisy
```

* Returns a tuple `(clean, noisy)` for use in training or evaluation loops.
</details>

### class SimpleDenoiser class

<details>

    
</details>



### class SimpleDenoiser class

<details>

    
</details>
