<!-- adding gaussian noise  -->

## Helper functions
### add_gaussian_noise

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


## Datasets and dataloaders
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

## Simple models for restoration
### class SimpleDenoiser class

<details>

This class defines a simple convolutional neural network (CNN) for image denoising using PyTorch.

```python
class SimpleDenoiser(nn.Module):
```

* Defines the class `SimpleDenoiser` which inherits from PyTorch's base class `nn.Module`.

```python
    def __init__(self):
        super(SimpleDenoiser, self).__init__()
```

* Constructor method: Initializes the neural network components and calls the constructor of the superclass `nn.Module`.

```python
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
```

* First convolutional layer: Takes a 1-channel input image and outputs 64 feature maps using a \$3\times3\$ kernel with padding of 1 (preserving spatial dimensions).

```python
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
```

* Second convolutional layer: Keeps the number of channels at 64.

```python
        self.conv3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
```

* Third convolutional layer: Reduces the feature maps back to a single channel (reconstructed image).

```python
        self.relu = nn.ReLU(inplace=True)
```

* ReLU activation function: Applies non-linearity after convolution. The `inplace=True` saves memory by modifying the input directly.

```python
    def forward(self, x):
```

* Defines the forward pass of the network, i.e., how the input `x` flows through the layers.

```python
        residual = x
```

* Stores the original input as `residual` to be added later (skip connection).

```python
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
```

* Applies the first and second convolutional layers followed by ReLU activation.

```python
        out = self.conv3(out)
```

* Applies the final convolution to bring the output back to a single channel.

```python
        out = out + residual  # Skip connection
```

* Adds the original input to the final output to form a residual (skip) connection. This helps the model learn corrections to the input.

```python
        return out
```

* Returns the denoised image.
    
</details>



## GAN for restoration
### Generator class

<details>

This class defines a generator model using a U-Net-like architecture implemented with PyTorch. It is designed for image-to-image translation tasks such as denoising or super-resolution.

```python
class Generator(nn.Module):
```

* Defines the class `Generator` inheriting from PyTorch's `nn.Module`.

```python
    def __init__(self):
        super(Generator, self).__init__()
```

* Constructor method that initializes the neural network and its layers.

##### Encoder (Downsampling Path)

```python
        self.enc1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)
```

* First encoder layer: Downsamples input from 1 channel to 64.

```python
        self.enc2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
```

* Second encoder layer: Further downsamples to 128 channels.

```python
        self.enc3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
```

* Third encoder layer: Outputs 256 feature maps.

```python
        self.enc4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
```

* Fourth encoder layer: Outputs deep features with 512 channels.

##### Decoder (Upsampling Path)

```python
        self.dec1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
```

* First decoder layer: Upsamples from 512 to 256 channels.

```python
        self.dec2 = nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1)
```

* Second decoder layer: Takes concatenated input (256+256), reduces to 128 channels.

```python
        self.dec3 = nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1)
```

* Third decoder layer: Takes concatenated input (128+128), reduces to 64.

```python
        self.dec4 = nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1)
```

* Final decoder layer: Takes concatenated input (64+64), outputs a single-channel image.

##### Activations

```python
        self.relu = nn.ReLU(inplace=True)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.tanh = nn.Tanh()
```

* ReLU for decoder layers, LeakyReLU for encoder layers, and Tanh to scale the final output.

#### Forward function

```python
    def forward(self, x):
```

* Defines the forward pass through the network.

```python
        e1 = self.leaky_relu(self.enc1(x))
        e2 = self.leaky_relu(self.enc2(e1))
        e3 = self.leaky_relu(self.enc3(e2))
        e4 = self.leaky_relu(self.enc4(e3))
```

* Encoder steps with LeakyReLU activations.

```python
        d1 = self.relu(self.dec1(e4))
        d1 = torch.cat([d1, e3], 1)
```

* Decoder step 1: Upsample `e4`, concatenate with encoder output `e3`.

```python
        d2 = self.relu(self.dec2(d1))
        d2 = torch.cat([d2, e2], 1)
```

* Decoder step 2: Repeat with `e2`.

```python
        d3 = self.relu(self.dec3(d2))
        d3 = torch.cat([d3, e1], 1)
```

* Decoder step 3: Repeat with `e1`.

```python
        d4 = self.tanh(self.dec4(d3))
```

* Final output with Tanh activation to produce values in the range $\[-1, 1]\$.

```python
        return d4
```

* Returns the final generated image.    
</details>

### Discriminator class

<details>

This class defines a convolutional neural network (CNN) for discriminating between real and fake images, typically used in GAN architectures.

```python
class Discriminator(nn.Module):
```

* Defines the class `Discriminator` inheriting from PyTorch's `nn.Module`.

```python
    def __init__(self):
        super(Discriminator, self).__init__()
```

* Constructor: Initializes the network and calls the base class constructor.

```python
        self.conv1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)
```

* First convolution layer: Converts 1 input channel into 64 feature maps.

```python
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
```

* Deeper convolutional layers: Downsample the input image and increase feature depth.

```python
        self.conv5 = nn.Conv2d(512, 1, kernel_size=8, stride=1, padding=0)
```

* Final convolution: Reduces the \$8\times8\$ feature map to a single output value per image.

```python
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
```

* Activation function: Applies a leaky ReLU non-linearity.

```python
        self.sigmoid = nn.Sigmoid()
```

* Final activation: Converts the output to a probability (0 to 1).

```python
        self.batch_norm2 = nn.BatchNorm2d(128)
        self.batch_norm3 = nn.BatchNorm2d(256)
        self.batch_norm4 = nn.BatchNorm2d(512)
```

* Batch normalization layers: Normalize the outputs of intermediate convolutions to stabilize training.

#### forward function
```python
    def forward(self, x):
```

* Defines the forward pass.

```python
        x = self.leaky_relu(self.conv1(x))
```

* First conv layer with activation (no batch norm here).

```python
        x = self.leaky_relu(self.batch_norm2(self.conv2(x)))
        x = self.leaky_relu(self.batch_norm3(self.conv3(x)))
        x = self.leaky_relu(self.batch_norm4(self.conv4(x)))
```

* Sequential convolutions with batch normalization and activation.

```python
        x = self.sigmoid(self.conv5(x))
```

* Final output: A single scalar probability indicating real (close to 1) or fake (close to 0).

```python
        return x
```

* Returns the output prediction.
</details>

## Traning functions
### train_simple_denoiser

<details>

This function trains a simple denoising neural network using MSE loss.

```python
def train_simple_denoiser(model, train_loader, num_epochs=50):
```

* Defines a function to train the denoiser model.
* Takes a model, a data loader for training data, and number of epochs (default is 50).

```python
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

* Checks if a GPU is available and sets the computation device accordingly.

```python
    model = model.to(device)
```

* Moves the model to the selected device (GPU or CPU).

```python
    criterion = nn.MSELoss()
```

* Uses Mean Squared Error loss for training.

```python
    optimizer = optim.Adam(model.parameters(), lr=0.001)
```

* Initializes the Adam optimizer with a learning rate of 0.001.

```python
    for epoch in range(num_epochs):
```

* Loops over the number of epochs specified.

```python
        running_loss = 0.0
```

* Initializes a variable to accumulate loss during the epoch.

```python
        for clean, noisy in train_loader:
```

* Iterates over the data loader, retrieving pairs of clean and noisy images.

```python
            clean, noisy = clean.to(device), noisy.to(device)
```

* Moves the batch data to the same device as the model.

```python
            optimizer.zero_grad()
```

* Clears the gradients from the previous step.

```python
            outputs = model(noisy)
```

* Feeds the noisy images through the model to produce denoised outputs.

```python
            loss = criterion(outputs, clean)
```

* Calculates the MSE loss between the model outputs and clean images.

```python
            loss.backward()
```

* Backpropagates the loss through the network.

```python
            optimizer.step()
```

* Updates the model parameters using the optimizer.

```python
            running_loss += loss.item()
```

* Accumulates the batch loss to calculate average epoch loss.

```python
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.6f}')
```

* Prints the average loss for the epoch.

```python
    return model
```

* Returns the trained model after all epochs.    
</details>

### train_gan_denoiser

<details>

This function trains a GAN-based denoising network consisting of a generator and a discriminator.

```python
def train_gan_denoiser(generator, discriminator, train_loader, num_epochs=50):
```
- Defines the training function for a GAN-based denoiser.
- Takes a generator model, discriminator model, training data loader, and number of epochs.

```python
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = generator.to(device)
    discriminator = discriminator.to(device)
```
- Sets the device and transfers the models to it.

```python
    criterion_gan = nn.BCELoss()
    criterion_pixel = nn.L1Loss()
```
- Defines the GAN loss (binary cross entropy) and the pixel-wise loss (L1 loss).

```python
    optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
```
- Initializes optimizers for generator and discriminator.

```python
    lambda_pixel = 100
```
- Sets the weight of the pixel-wise loss.

```python
    for epoch in range(num_epochs):
        for clean, noisy in train_loader:
```
- Iterates through epochs and the data loader.

```python
            clean, noisy = clean.to(device), noisy.to(device)
            batch_size = clean.size(0)
```
- Moves data to device and gets batch size.

```python
            valid = torch.ones((batch_size, 1, 1, 1), requires_grad=False).to(device)
            fake = torch.zeros((batch_size, 1, 1, 1), requires_grad=False).to(device)
```
- Defines labels for real (1) and fake (0) samples.

### Train Generator

```python
            optimizer_g.zero_grad()
            gen_img = generator(noisy)
```
- Zeroes gradients and generates denoised images.

```python
            pred_fake = discriminator(gen_img)
            loss_gan = criterion_gan(pred_fake, valid)
            loss_pixel = criterion_pixel(gen_img, clean)
            loss_g = loss_gan + lambda_pixel * loss_pixel
```
- Computes GAN and pixel losses. Combines them.

```python
            loss_g.backward()
            optimizer_g.step()
```
- Backpropagates generator loss and updates weights.

### Train Discriminator

```python
            optimizer_d.zero_grad()
            pred_real = discriminator(clean)
            loss_real = criterion_gan(pred_real, valid)
            pred_fake = discriminator(gen_img.detach())
            loss_fake = criterion_gan(pred_fake, fake)
            loss_d = 0.5 * (loss_real + loss_fake)
```
- Computes real and fake losses for discriminator.

```python
            loss_d.backward()
            optimizer_d.step()
```
- Backpropagates discriminator loss and updates weights.

```python
        print(f'Epoch {epoch+1}/{num_epochs}, G Loss: {loss_g.item():.6f}, D Loss: {loss_d.item():.6f}')
```
- Logs progress for each epoch.

```python
    return generator
```
- Returns the trained generator model.    
</details>

## Main function
### Main function

<details>

This function coordinates the image restoration pipeline, including noise addition, denoising using classical and learning-based techniques, and evaluation via perception-distortion metrics.

```python
def main():
```
- The main entry point for the restoration pipeline.

#### Load Original Image
```python
    try:
        img = Image.open('...')
        ...
    except Exception as e:
        ...
```
- Tries to load an image from disk, converting it to grayscale.
- If loading fails, generates a synthetic image (a white circle on black background).
- Ensures image dimensions are compatible with network input (128x128).

#### Add Gaussian Noise
```python
    noisy = add_gaussian_noise(original, std=0.1)
```
- Adds Gaussian noise to simulate image degradation.

#### Save Images for Reference
```python
    plt.imsave(...)
```
- Saves both original and noisy images for later inspection.

#### Preprocessing and Dataset Creation
```python
    transform = transforms.Compose([...])
    clean_images = [original for _ in range(100)]
    noisy_images = [...]
    dataset = NoisyImageDataset(...)
    train_loader = DataLoader(...)
```
- Defines a PyTorch dataset and data loader from noisy/clean image pairs.

#### Classical Denoising
```python
    gaussian_restored = ndimage.gaussian_filter(noisy, sigma=1)
```
- Applies a classical Gaussian blur filter to reduce noise.

#### CNN-based Denoising
```python
    simple_denoiser = train_simple_denoiser(...)
    with torch.no_grad():
        simple_restored = ...
```
- Trains a simple CNN denoiser using MSE.
- Restores the noisy image using the trained model.

#### GAN-based Denoising
```python
    generator = Generator()
    discriminator = Discriminator()
    generator = train_gan_denoiser(...)
    with torch.no_grad():
        gan_restored = ...
```
- Trains a GAN with a generator and discriminator for perceptual-quality restoration.

#### Evaluate Restoration
```python
    metrics = []
    for img in [gaussian_restored, simple_restored, gan_restored]:
        psnr_val, ssim_val = calculate_metrics(original, img)
        metrics.append(...)
```
- Calculates PSNR and SSIM for each restored image.

#### Plot Results
```python
    plot_results(...)
```
- Plots and saves the original, noisy, and restored images along with metric values.

#### Plot Perception-Distortion Trade-off
```python
    plt.figure(...)
    plt.scatter(...)
    ...
    plt.savefig(...)
    plt.show()
```
- Visualizes the trade-off by plotting PSNR (distortion) vs. SSIM (perception quality).

#### Output Summary
```python
    print(...)
```
- Prints a table summarizing the metric values for each method.

#### Run
```python
if __name__ == "__main__":
    main()
```
- Executes `main()` when the script is run directly.    
</details>
