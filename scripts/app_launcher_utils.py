"""Helpers for bootstrapping Isaac Lab entrypoints before AppLauncher starts."""

from __future__ import annotations

import os


def pin_process_to_requested_cuda_device(device: str | None) -> str | None:
    """Pin the current process to the requested CUDA device.

    On some Isaac Sim / PyTorch setups, leaving CUDA visibility unpinned can make
    lazy CUDA initialization observe an invalid device index. When the caller asks
    for a single CUDA device and the process has not already been pinned through
    ``CUDA_VISIBLE_DEVICES``, expose only that physical GPU and remap the in-process
    device string to ``cuda:0``.
    """
    if device is None or "cuda" not in device:
        return device

    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return device

    device_index = 0
    if ":" in device:
        suffix = device.split(":", 1)[1]
        if suffix:
            device_index = int(suffix)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_index)
    return "cuda:0"
