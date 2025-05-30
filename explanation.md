
### `add_gaussian_noise`
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
