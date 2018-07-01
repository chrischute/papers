import numpy as np


def normalize_hu(volume, min_val, max_val, avg_val):
    """Normalize an ndarray of Hounsfield Units to [-1, 1].

    Clips the values to [min, max] and scales into [0, 1],
    then subtracts the mean pixel (min, max, mean are defined in constants.py).

    Args:
        volume: NumPy ndarray to convert. Any shape.
        min_val: Minimum HU value. All lower values get clipped here.
        max_val: Maximum HU value. All higher values get clipped here.
        avg_val: Average HU value of training set *after clipping and scaling to [0, 1]*.

    Returns:
        NumPy ndarray with normalized pixels in [-1, 1]. Same shape as input.
    """

    volume = volume.astype(np.float32)
    volume = (volume - min_val) / (max_val - min_val)
    volume = np.clip(volume, 0., 1.) - avg_val

    return volume


def apply_window(img, w_center, w_width, y_min=0., y_max=255., dtype=np.uint8):
    """Window a NumPy array of raw Hounsfield Units.

    Args:
        img: Image to apply the window to. NumPy array of any shape.
        w_center: Center of window.
        w_width: Width of window.
        y_min: Min value for output image.
        y_max: Max value for output image
        dtype: Data type for elements in output image ndarray.

    Returns:
        img_np: NumPy array of after windowing. Values in range [y_min, y_max].
    """
    img_np = np.zeros_like(img, dtype=np.float64)

    # Clip to the lower edge
    x_min = w_center - 0.5 - (w_width - 1.) / 2.
    img_np[img <= x_min] = y_min

    # Clip to the upper edge
    x_max = w_center - 0.5 + (w_width - 1.) / 2.
    img_np[img > x_max] = y_max

    # Scale everything in the middle
    img_np[(img > x_min) & (img <= x_max)] = (((img[(img > x_min) & (img <= x_max)] - (w_center - 0.5))
                                               / (w_width - 1.) + 0.5) * (y_max - y_min) + y_min)

    return img_np.astype(dtype)


def un_normalize_hu(tensor, pixel_dict, dtype=np.int16):
    """Un-normalize a PyTorch Tensor seen by the model into a NumPy array of
    pixels fit for visualization. If using raw Hounsfield Units, window the input.

    Args:
        tensor: Tensor with pixel values in range (-1, 1).
            If video, shape (batch_size, num_channels, num_frames, height, width).
            If image, shape (batch_size, num_channels, height, width).
        pixel_dict: Dictionary containing min, max, avg of pixel data; window center, width.
        dtype: Desired output data type for each entry the image array.

    Returns:
        pixels_np: Numpy ndarray with entries of type `dtype`.
    """
    pixels_np = tensor.cpu().float().numpy()

    # Reverse pre-processing steps for visualization
    pixels_np = ((pixels_np + pixel_dict['avg_val'])
                 * (pixel_dict['max_val'] - pixel_dict['min_val'])
                 + pixel_dict['min_val'])
    pixels_np = pixels_np.astype(dtype=dtype)
    pixels_np = apply_window(pixels_np, pixel_dict['w_center'], pixel_dict['w_width'])

    return pixels_np
