"""
Video processing utilities for VideoLLaMA3.

This module extracts the core video frame processing components from their
original locations in the repository:

.. list-table:: Source-file map
   :header-rows: 1
   :widths: 35 55 10

   * - Function
     - Original file
     - Line(s)
   * - ``frame_sample``
     - ``inference/transformers_api/processing_videollama3.py``
     - 96–124
   * - ``load_video_from_ids``
     - ``inference/transformers_api/processing_videollama3.py``
     - 127–184
   * - ``spatial_downsampling``
     - ``videollama3/model/videollama3_arch.py``
     - 31–50
   * - ``get_compression_mask``  (AVT + DiffF)
     - ``videollama3/model/videollama3_arch.py``
     - 202–239  (``Videollama3MetaForCausalLM._get_compression_mask``)
   * - ``compress_visual_tokens``
     - ``videollama3/model/videollama3_arch.py``
     - 241–268  (``Videollama3MetaForCausalLM._compress_visual_tokens``)
"""

import math
import os
from typing import List, Optional

import cv2
import imageio
import numpy as np
import torch
import torch.nn as nn
from decord import VideoReader, cpu


# Default FPS assumptions for non-standard video sources
_FRAME_DIR_FPS = 3   # directory of extracted frame images
_GIF_FPS = 25        # animated GIF


# ---------------------------------------------------------------------------
# Frame Sampling
# ---------------------------------------------------------------------------

def frame_sample(duration: int, mode: str = 'uniform', num_frames: int = None,
                 vid_fps: float = None, fps: float = None) -> np.ndarray:
    """Sample frame indices from a video.

    Source:
        ``inference/transformers_api/processing_videollama3.py``, lines 96–124.

    Args:
        duration: Total number of frames in the video (or clip).
        mode: Sampling strategy. Either ``'uniform'`` (evenly-spaced indices) or
            ``'fps'`` (fixed frames-per-second rate).
        num_frames: Number of frames to sample. Required when ``mode='uniform'``.
        vid_fps: Native frame-rate of the source video. Required when
            ``mode='fps'``.
        fps: Target frame-rate for sampling. Required when ``mode='fps'``.

    Returns:
        1-D integer array of sampled frame indices.
    """
    if mode == 'uniform':
        assert num_frames is not None, "Number of frames must be provided for uniform sampling."
        if duration <= num_frames:
            return np.arange(duration).astype(int)
        return np.linspace(0, duration - 1, num_frames, dtype=int)
    elif mode == 'fps':
        assert vid_fps is not None, "FPS must be provided for FPS sampling."
        assert fps is not None, "FPS must be provided for FPS sampling."
        segment_len = min(vid_fps // fps, duration)
        return np.arange(segment_len // 2, duration, segment_len, dtype=int)
    else:
        raise ValueError(f'Unsupported frame sampling mode: {mode}')


# ---------------------------------------------------------------------------
# Video Loading
# ---------------------------------------------------------------------------

def load_video_from_ids(
    video_path: str,
    s: float = None,
    e: float = None,
    fps: float = None,
    max_frames: int = 128,
    temporal_factor: int = 1,
):
    """Load frames from a video file (or directory of frames / GIF).

    Source:
        ``inference/transformers_api/processing_videollama3.py``, lines 127–184.

    Supports three input formats:
    * A directory of image files (assumed ``_FRAME_DIR_FPS`` FPS).
    * An animated GIF (assumed ``_GIF_FPS`` FPS).
    * Any video container readable by *decord* (mp4, avi, …).

    Args:
        video_path: Path to the video file, GIF, or directory of frame images.
        s: Start time in seconds. ``None`` means the beginning of the video.
        e: End time in seconds. ``None`` means the end of the video.
        fps: Desired frames-per-second for FPS-based sampling. When ``None``
            uniform sampling is used.
        max_frames: Maximum number of frames to return.
        temporal_factor: Pad the frame sequence to the next multiple of this
            value by repeating the last frame. ``1`` disables padding.

    Returns:
        frames: List of RGB frames, each with shape ``(C, H, W)``.
        timestamps: List of float timestamps (in seconds) for each frame.
    """
    if s is not None and e is not None:
        s = max(s, 0.)
        e = max(e, 0.)
        if s > e:
            s, e = e, s
        elif s == e:
            e = s + 1

    # 1. Load video metadata
    if os.path.isdir(video_path):
        frame_files = sorted(os.listdir(video_path))
        vid_fps = _FRAME_DIR_FPS
        num_frames_of_video = len(frame_files)
    elif video_path.endswith('.gif'):
        gif_reader = imageio.get_reader(video_path)
        vid_fps = _GIF_FPS
        num_frames_of_video = len(gif_reader)
    else:
        vreader = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        vid_fps = vreader.get_avg_fps()
        num_frames_of_video = len(vreader)

    # 2. Determine frame range
    f_start = 0 if s is None else max(int(s * vid_fps) - 1, 0)
    f_end = num_frames_of_video - 1 if e is None else min(int(e * vid_fps) - 1, num_frames_of_video - 1)
    frame_indices = list(range(f_start, f_end + 1))

    # 3. Sample frame indices
    duration = len(frame_indices)
    if fps is not None and duration / vid_fps < max_frames:
        sampled_frame_indices = [frame_indices[i] for i in frame_sample(duration, mode='fps', vid_fps=vid_fps, fps=fps)]
    else:
        sampled_frame_indices = [frame_indices[i] for i in frame_sample(duration, mode='uniform', num_frames=max_frames)]

    # 4. Decode frames
    if os.path.isdir(video_path):
        frames = np.array([
            cv2.cvtColor(cv2.imread(os.path.join(video_path, frame_files[idx])), cv2.COLOR_BGR2RGB)
            for idx in sampled_frame_indices
        ])
    elif video_path.endswith('.gif'):
        frames = np.array([
            cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            for idx, frame in enumerate(gif_reader)
            if idx in sampled_frame_indices
        ])
    else:
        frames = vreader.get_batch(sampled_frame_indices).asnumpy()

    # (T, H, W, C) -> (T, C, H, W)
    frames = frames.transpose(0, 3, 1, 2)
    timestamps = [x / vid_fps for x in sampled_frame_indices]

    # 5. Optional temporal padding
    if temporal_factor > 1:
        pad_length = (temporal_factor - len(frames) % temporal_factor) % temporal_factor
        if pad_length > 0:
            frames = np.concatenate([frames, frames[-1:].repeat(pad_length, axis=0)])
            frame_interval = 1 / fps if fps is not None else 1 / vid_fps
            for _ in range(pad_length):
                timestamps.append(timestamps[-1] + frame_interval)

    frames = [frame for frame in frames]
    return frames, timestamps


# ---------------------------------------------------------------------------
# Spatial Downsampling
# ---------------------------------------------------------------------------

def spatial_downsampling(features: torch.Tensor, grid_thws, stride: int = 2) -> torch.Tensor:
    """Spatially downsample packed visual feature tokens via bilinear interpolation.

    Source:
        ``videollama3/model/videollama3_arch.py``, lines 31–50
        (module-level function ``spatial_downsampling``).

    Args:
        features: Packed feature tensor of shape ``(N, C)`` where ``N`` is the
            total number of tokens across all images/frames in the batch.
        grid_thws: Nested list (batch × items) of ``(T, H, W)`` grid size
            tensors, one per image/frame sequence.
        stride: Downsampling factor applied to both spatial dimensions.

    Returns:
        Downsampled feature tensor of shape ``(N', C)``.
    """
    n, c = features.shape

    flatten_grid_thws = torch.cat([grid_thw for batch_grid_thws in grid_thws for grid_thw in batch_grid_thws])
    split_sizes = [grid_thw.prod() for grid_thw in flatten_grid_thws]
    features = torch.split(features, split_sizes)

    new_features = []
    for feature, grid_thw in zip(features, flatten_grid_thws):
        # NOTE: adapted for reshape in image processor
        feature = feature.view(
            grid_thw[0], grid_thw[1] // stride, grid_thw[2] // stride,
            stride, stride, c
        ).permute(0, 1, 3, 2, 4, 5)
        feature = feature.reshape(grid_thw[0], grid_thw[1], grid_thw[2], c).permute(0, 3, 1, 2)
        new_feature = nn.functional.interpolate(
            feature,
            (math.ceil(grid_thw[1] / stride), math.ceil(grid_thw[2] / stride)),
            mode='bilinear',
        )
        new_features.append(new_feature.permute(0, 2, 3, 1).view(-1, c))
    return torch.cat(new_features)


# ---------------------------------------------------------------------------
# Adaptive Visual Tokenization (AVT) + Differential Frame (DiffF)
# ---------------------------------------------------------------------------

def get_compression_mask(
    pixel_values: torch.Tensor,
    batched_num_patches: torch.Tensor,
    grid_sizes: torch.Tensor,
    merge_sizes: torch.Tensor,
    modals: List[str],
    threshold: float = 0.1,
    min_tokens: int = 1,
) -> torch.BoolTensor:
    """Compute the Adaptive Visual Tokenization (AVT) compression mask.

    Source:
        ``videollama3/model/videollama3_arch.py``, lines 202–239
        (``Videollama3MetaForCausalLM._get_compression_mask``).

    For each video in the batch the mask identifies which spatial tokens are
    *sufficiently different* from the previous frame (Differential Frame /
    DiffF criterion).  Tokens belonging to image inputs — or to the first
    frame of a video — are always kept.

    Algorithm (per video):
        1. Reshape packed tokens to ``(T, spatial_patches, C)``.
        2. Compute frame-level L1 pixel differences:
           ``diff[t] = mean(|frame[t] - frame[t-1]|) * 255``.
        3. Always keep the first frame (seed diff > threshold).
        4. Retain tokens where ``diff > threshold``.
        5. Guarantee at least ``min_tokens`` active tokens per frame.

    Args:
        pixel_values: Packed visual tokens, shape ``(total_patches, C)``.
        batched_num_patches: Number of projected patches per input item, shape
            ``(B,)``.
        grid_sizes: ``(T, H, W)`` grid sizes per item, shape ``(B, 3)``.
        merge_sizes: Spatial merge factor per item, shape ``(B,)``.
        modals: Modality string (``'image'``, ``'video'``, or ``'text'``) for
            each item in the batch.
        threshold: Normalised pixel-difference threshold in ``[0, 1]``
            (effectively ``[0, 255]`` after scaling). Default: ``0.1``.
        min_tokens: Minimum number of active tokens to retain per frame.
            Default: ``1``.

    Returns:
        Boolean mask of shape ``(total_patches,)`` — ``True`` means the token
        is retained.
    """
    batched_images = pixel_values.split(grid_sizes.prod(dim=1).tolist(), dim=0)
    compression_masks = []

    for images, num_patches, grid_size, merge_size, modal in zip(
        batched_images, batched_num_patches, grid_sizes, merge_sizes, modals
    ):
        t, h, w = grid_size
        if modal == "image" or (modal == "video" and t == 1):
            # Images and single-frame videos: keep all tokens.
            compression_masks.append(torch.ones((num_patches,), dtype=torch.bool, device=images.device))

        elif modal == "video":
            # --- DiffF: Differential Frame token selection ---
            # Reshape to (T, spatial_patches, C)
            images = images.view(t, (h // merge_size) * (w // merge_size), -1)

            # Compute per-patch absolute frame differences, scaled to [0, 255]
            pixel_diff = images[1:] - images[:-1]
            pixel_diff = torch.abs(pixel_diff).mean(dim=-1) * 255

            # Prepend a sentinel value for frame 0 so it is always retained
            pixel_diff = torch.cat([torch.full_like(pixel_diff[0:1], threshold + 1), pixel_diff], dim=0)

            # Threshold: True = token is significantly different → keep it
            mask = pixel_diff > threshold

            # Ensure each frame has at least min_tokens active tokens
            padding_ids = torch.nonzero(mask.sum(dim=1) < min_tokens)[:, 0]
            mask[padding_ids, :min_tokens] = True

            compression_masks.append(mask.flatten())

        else:
            # Pseudo-image / text placeholder: no tokens
            compression_masks.append(torch.ones((0,), dtype=torch.bool, device=images.device))

    return torch.cat(compression_masks)


def compress_visual_tokens(
    compression_mask: torch.BoolTensor,
    mm_features: torch.Tensor,
    input_ids: torch.Tensor,
    image_token_index: int,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
):
    """Apply the AVT compression mask to remove redundant visual tokens.

    Source:
        ``videollama3/model/videollama3_arch.py``, lines 241–268
        (``Videollama3MetaForCausalLM._compress_visual_tokens``).

    Given a flat (batch-merged) sequence of token IDs and corresponding
    multimodal features, this function:
    * Filters ``mm_features`` according to ``compression_mask``.
    * Removes the corresponding image-token positions from ``input_ids``
      (and optionally from ``attention_mask``, ``labels``, and
      ``position_ids``).
    * Re-indexes ``position_ids`` so each sequence still starts at 0.

    Args:
        compression_mask: Boolean mask produced by :func:`get_compression_mask`,
            shape ``(total_visual_tokens,)``.
        mm_features: Projected visual features, shape
            ``(total_visual_tokens, D)``.
        input_ids: Flat 1-D tensor of token IDs (batch dimension merged).
        image_token_index: Token ID used as a placeholder for image/video
            content in ``input_ids``.
        attention_mask: Optional flat 1-D attention mask.
        position_ids: Optional flat 1-D position IDs.
        labels: Optional flat 1-D label IDs.

    Returns:
        Tuple ``(mm_features, input_ids, attention_mask, position_ids, labels)``
        with compressed / filtered tensors.
    """
    mm_features = mm_features[compression_mask]
    image_selected = (input_ids == image_token_index)

    # Build a mask over the full token sequence (text + image placeholders)
    text_masks = torch.logical_not(image_selected)
    text_masks[image_selected] = compression_mask
    input_ids = input_ids[text_masks]

    if attention_mask is not None:
        attention_mask = attention_mask[text_masks]
    if labels is not None:
        labels = labels[text_masks]
    if position_ids is not None:
        # FIXME: assume the first position_id is always 0
        position_ids = position_ids[text_masks]
        pos_start = [0] + torch.nonzero(position_ids == 0)[:, 0].tolist()
        pos_end = pos_start[1:] + [len(input_ids)]
        position_ids = torch.cat([
            torch.arange(end - start, device=input_ids.device)
            for start, end in zip(pos_start, pos_end)
        ])

    return mm_features, input_ids, attention_mask, position_ids, labels
