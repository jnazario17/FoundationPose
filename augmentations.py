import cv2
import numpy as np


def apply_occlusion(rgb, depth, mask, occlusion_ratio=0.4, seed=None):
    """
    Applies synthetic occlusion blocking a portion of the object.
    Modifies RGB and depth in the occluded region and returns the modified inputs
    along with the binary occlusion mask.
    """
    if seed is not None:
        np.random.seed(seed)

    rgb_aug = rgb.copy()
    depth_aug = depth.copy()
    H, W = rgb.shape[:2]

    vs, us = np.where(mask > 0)
    if len(vs) == 0:
        return rgb_aug, depth_aug, np.zeros((H, W), dtype=np.uint8)

    umin, umax = us.min(), us.max()
    vmin, vmax = vs.min(), vs.max()
    box_w = umax - umin
    box_h = vmax - vmin

    # Occlusion box covering occlusion_ratio of the bounding box
    occ_w = int(box_w * np.sqrt(occlusion_ratio))
    occ_h = int(box_h * np.sqrt(occlusion_ratio))

    # Center or offset occlusion box over the object
    ox = np.random.randint(umin, max(umin + 1, umax - occ_w + 1))
    oy = np.random.randint(vmin, max(vmin + 1, vmax - occ_h + 1))

    occ_mask = np.zeros((H, W), dtype=np.uint8)
    occ_mask[oy:oy+occ_h, ox:ox+occ_w] = 1

    # Render a realistic synthetic obstacle texture (e.g., grey obstacle or textured occluder)
    noise_patch = np.random.randint(60, 110, (occ_h, occ_w, 3), dtype=np.uint8)
    rgb_aug[oy:oy+occ_h, ox:ox+occ_w] = noise_patch

    # In depth, set occluder closer (or invalid 0)
    median_z = np.median(depth[mask > 0]) if (depth[mask > 0] > 0).any() else 0.8
    depth_aug[oy:oy+occ_h, ox:ox+occ_w] = max(0.1, median_z - 0.15)  # 15cm in front of object

    return rgb_aug, depth_aug, occ_mask


def apply_blur(rgb, depth, mask, kernel_size=21, sigma=8.0):
    """Applies strong Gaussian blur simulating rapid motion or severe defocus."""
    rgb_aug = cv2.GaussianBlur(rgb, (kernel_size, kernel_size), sigma)
    # Depth is also slightly blurred at edges simulating defocus
    depth_aug = cv2.GaussianBlur(depth.astype(np.float32), (5, 5), 2.0)
    return rgb_aug, depth_aug, np.zeros(rgb.shape[:2], dtype=np.uint8)


def apply_sensor_noise(rgb, depth, mask, rgb_sigma=35.0, depth_dropout_rate=0.20, seed=None):
    """Applies Gaussian sensor noise to RGB and random dropouts/noise to depth."""
    if seed is not None:
        np.random.seed(seed)

    rgb_float = rgb.astype(np.float32)
    noise = np.random.normal(0, rgb_sigma, rgb.shape).astype(np.float32)
    rgb_aug = np.clip(rgb_float + noise, 0, 255).astype(np.uint8)

    depth_aug = depth.copy().astype(np.float32)
    # Depth Gaussian jitter
    depth_noise = np.random.normal(0, 0.015, depth.shape).astype(np.float32)  # 1.5cm noise
    depth_aug = np.where(depth_aug > 0, depth_aug + depth_noise, 0.0)
    # Random depth dropout in object region
    dropout_mask = np.random.rand(*depth.shape) < depth_dropout_rate
    depth_aug[dropout_mask & (mask > 0)] = 0.0

    return rgb_aug, depth_aug, np.zeros(rgb.shape[:2], dtype=np.uint8)


def apply_low_light(rgb, depth, mask, factor=0.25, noise_sigma=15.0, seed=None):
    """Simulates extreme low-light illumination with sensor shot noise."""
    if seed is not None:
        np.random.seed(seed)

    rgb_dim = (rgb.astype(np.float32) * factor)
    noise = np.random.normal(0, noise_sigma, rgb.shape).astype(np.float32)
    rgb_aug = np.clip(rgb_dim + noise, 0, 255).astype(np.uint8)
    return rgb_aug, depth.copy(), np.zeros(rgb.shape[:2], dtype=np.uint8)


def apply_combined(rgb, depth, mask, seed=None):
    """Applies simultaneous occlusion and motion blur."""
    rgb_occ, depth_occ, occ_mask = apply_occlusion(rgb, depth, mask, occlusion_ratio=0.35, seed=seed)
    rgb_comb, _, _ = apply_blur(rgb_occ, depth_occ, mask, kernel_size=15, sigma=5.0)
    return rgb_comb, depth_occ, occ_mask
