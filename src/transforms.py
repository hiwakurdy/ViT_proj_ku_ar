from typing import Dict

import numpy as np
import torch
from PIL import Image


_PIL_INTERPOLATION = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
}


def load_grayscale_image(image_path: str) -> np.ndarray:
    with Image.open(image_path) as image:
        return np.array(image.convert("L"), dtype=np.uint8)


def otsu_threshold(image: np.ndarray) -> int:
    histogram = np.bincount(image.reshape(-1), minlength=256).astype(np.float64)
    total = image.size
    value_sum = np.dot(np.arange(256), histogram)
    background_weight = 0.0
    background_sum = 0.0
    best_threshold = 0
    best_between = -1.0

    for threshold in range(256):
        background_weight += histogram[threshold]
        if background_weight == 0:
            continue

        foreground_weight = total - background_weight
        if foreground_weight == 0:
            break

        background_sum += threshold * histogram[threshold]
        background_mean = background_sum / background_weight
        foreground_mean = (value_sum - background_sum) / foreground_weight
        between = background_weight * foreground_weight * (background_mean - foreground_mean) ** 2

        if between > best_between:
            best_between = between
            best_threshold = threshold

    return best_threshold


def binarize_image(gray_image: np.ndarray) -> np.ndarray:
    threshold = otsu_threshold(gray_image)
    return np.where(gray_image <= threshold, 255, 0).astype(np.uint8)


def crop_to_foreground(image: np.ndarray, mask: np.ndarray, pad: int = 2) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return image

    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(image.shape[1] - 1, int(xs.max()) + pad)
    y1 = min(image.shape[0] - 1, int(ys.max()) + pad)
    return image[y0 : y1 + 1, x0 : x1 + 1]


def resize_keep_aspect_and_pad(
    image: np.ndarray,
    img_h: int,
    img_w: int,
    interpolation: str = "bilinear",
    pad_value: int = 255,
) -> np.ndarray:
    height, width = image.shape
    if height <= 0 or width <= 0:
        return np.full((img_h, img_w), pad_value, dtype=np.uint8)

    scale = img_h / float(height)
    new_width = max(1, min(img_w, int(round(width * scale))))
    pil_image = Image.fromarray(image)
    resized = pil_image.resize(
        (new_width, img_h),
        resample=_PIL_INTERPOLATION.get(interpolation, Image.Resampling.BILINEAR),
    )

    canvas = np.full((img_h, img_w), pad_value, dtype=np.uint8)
    canvas[:, :new_width] = np.array(resized, dtype=np.uint8)
    return canvas


def preprocess_image(image_path: str, transform_config: Dict) -> np.ndarray:
    gray_image = load_grayscale_image(image_path)
    return preprocess_gray_image(gray_image, transform_config)


def preprocess_gray_image(gray_image: np.ndarray, transform_config: Dict) -> np.ndarray:
    binarize = bool(transform_config.get("binarize", False))
    crop_to_text = bool(transform_config.get("crop_to_text", False))
    crop_pad = int(transform_config.get("crop_pad", 2))
    interpolation = str(transform_config.get("interpolation", "bilinear"))
    img_h = int(transform_config["img_h"])
    img_w = int(transform_config["img_w"])

    if binarize:
        processed = binarize_image(gray_image)
        mask = processed
        pad_value = 0
    else:
        processed = gray_image
        mask = binarize_image(gray_image)
        pad_value = 255

    if crop_to_text:
        processed = crop_to_foreground(processed, mask, pad=crop_pad)

    return resize_keep_aspect_and_pad(
        processed,
        img_h=img_h,
        img_w=img_w,
        interpolation=interpolation,
        pad_value=pad_value,
    )


def image_to_tensor(processed_image: np.ndarray, transform_config: Dict) -> torch.Tensor:
    image = processed_image.astype(np.float32) / 255.0
    image = np.stack([image] * int(transform_config.get("in_chans", 3)), axis=0)
    mean = np.array(transform_config.get("mean", [0.5, 0.5, 0.5]), dtype=np.float32).reshape(-1, 1, 1)
    std = np.array(transform_config.get("std", [0.5, 0.5, 0.5]), dtype=np.float32).reshape(-1, 1, 1)
    image = (image - mean) / std
    return torch.from_numpy(image.astype(np.float32))
