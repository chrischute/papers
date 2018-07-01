import matplotlib.pyplot as plt
import numpy as np


def _make_rgb(image):
    """Tile a NumPy array to make sure it has 3 channels."""
    if image.shape[-1] != 3:
        tiling_shape = [1] * (len(image.shape) - 1) + [3]
        return np.tile(image, tiling_shape)
    else:
        return image


def _normalize_png(image):
    """Normalize pixels to the range 0-255."""
    image -= np.amin(image)
    image /= (np.amax(image) + 1e-7)
    image *= 255

    return image


def add_heat_map(pixels_np, intensities_np, alpha_img=0.33, color_map='magma', normalize=True):
    """Add a CAM heat map as an overlay on a PNG image.

    Args:
        pixels_np: Pixels to add the heat map on top of. Must be in range (0, 1).
        intensities_np: Intensity values for the heat map. Must be in range (0, 1).
        alpha_img: Weight for image when summing with heat map. Must be in range (0, 1).
        color_map: Color map scheme to use with PyPlot.
        normalize: If True, normalize the intensities to range exactly from 0 to 1.

    Returns:
        Original pixels with heat map overlaid.
    """
    assert(np.max(intensities_np) <= 1 and np.min(intensities_np) >= 0)
    color_map_fn = plt.get_cmap(color_map)
    if normalize:
        intensities_np = _normalize_png(intensities_np)
    else:
        intensities_np *= 255
    heat_map = color_map_fn(intensities_np.astype(np.uint8))
    if len(heat_map.shape) == 3:
        heat_map = heat_map[:, :, :3]
    else:
        heat_map = heat_map[:, :, :, :3]

    new_img = alpha_img * pixels_np.astype(np.float32) + (1. - alpha_img) * heat_map.astype(np.float32)
    new_img = np.uint8(_normalize_png(new_img))

    return new_img


def stack_videos(img_list):
    """Stacks a sequence of image numpy arrays of shape (num_images x w x h x c) to display side-by-side."""
    # If not RGB, stack to make num_channels consistent
    img_list = [_make_rgb(img) for img in img_list]
    stacked_array = np.concatenate(img_list, axis=2)
    return stacked_array
