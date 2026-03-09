"""Projection-profile helpers for word-image classification."""

from typing import Tuple

import numpy as np

from src.transforms import binarize_image


def compute_projection_feature_dim(h_proj_length: int, v_proj_length: int) -> int:
    return int(h_proj_length) + int(v_proj_length)


def normalize_projection_vector(vector: np.ndarray) -> np.ndarray:
    """Scale a 1D projection vector to [0, 1] by its max value."""
    vector = vector.astype(np.float32)
    max_value = float(vector.max()) if vector.size else 0.0
    if max_value <= 0.0:
        return np.zeros_like(vector, dtype=np.float32)
    return vector / max_value


def resize_projection_vector(vector: np.ndarray, target_length: int) -> np.ndarray:
    """Interpolate a 1D projection vector to a fixed length."""
    target_length = int(target_length)
    if target_length <= 0:
        raise ValueError("Projection target length must be positive.")
    if vector.size == 0:
        return np.zeros(target_length, dtype=np.float32)
    if vector.size == 1:
        return np.full(target_length, float(vector[0]), dtype=np.float32)
    if vector.size == target_length:
        return vector.astype(np.float32)

    src_positions = np.linspace(0.0, 1.0, num=vector.size, dtype=np.float32)
    dst_positions = np.linspace(0.0, 1.0, num=target_length, dtype=np.float32)
    return np.interp(dst_positions, src_positions, vector).astype(np.float32)


def build_foreground_signal(image: np.ndarray, input_mode: str = "binary_count") -> np.ndarray:
    """Create the per-pixel foreground signal used to build projection profiles."""
    image_uint8 = image.astype(np.uint8)
    foreground_mask = (binarize_image(image_uint8) > 0).astype(np.float32)

    if input_mode == "binary_count":
        return foreground_mask

    if input_mode == "foreground_intensity":
        inverted = 1.0 - (image_uint8.astype(np.float32) / 255.0)
        return inverted * foreground_mask

    raise ValueError(f"Unsupported projection input_mode: {input_mode}")


def compute_horizontal_projection(foreground_signal: np.ndarray) -> np.ndarray:
    """Sum foreground values for each row -> vector of length image_height."""
    return foreground_signal.sum(axis=1).astype(np.float32)


def compute_vertical_projection(foreground_signal: np.ndarray) -> np.ndarray:
    """Sum foreground values for each column -> vector of length image_width."""
    return foreground_signal.sum(axis=0).astype(np.float32)


def extract_projection_profiles(
    image: np.ndarray,
    h_proj_length: int,
    v_proj_length: int,
    input_mode: str = "binary_count",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute, normalize, and resize horizontal/vertical projection profiles."""
    foreground_signal = build_foreground_signal(image=image, input_mode=input_mode)
    h_proj = compute_horizontal_projection(foreground_signal)
    v_proj = compute_vertical_projection(foreground_signal)
    h_proj = resize_projection_vector(normalize_projection_vector(h_proj), h_proj_length)
    v_proj = resize_projection_vector(normalize_projection_vector(v_proj), v_proj_length)
    return h_proj.astype(np.float32), v_proj.astype(np.float32)
