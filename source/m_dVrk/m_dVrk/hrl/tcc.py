"""TCC (Temporal Cycle-Consistency) model and checkpoint utilities.

This module contains:
  - XIRLResnet18: the pretrained visual encoder used for reward computation.
  - resolve_tcc_checkpoint / get_best_available_checkpoint: helpers to locate
    the best checkpoint in an XIRL experiment directory.
  - compute_average_goal_embedding: builds the goal embedding from a dataset
    of goal-state frames.

All scripts (training, collection, evaluation) should import from here instead
of defining their own copies of XIRLResnet18.
"""

from __future__ import annotations

import glob
import os

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.io as io
import torchvision.transforms as T


# ---------------------------------------------------------------------------
# TCC visual encoder
# ---------------------------------------------------------------------------

class XIRLResnet18(nn.Module):
    """ResNet-18 visual encoder compatible with XIRL / TCC checkpoints.

    Matches the XIRL config::

        model_type = "resnet18_linear"
        embedding_size = 32
        normalize_embeddings = False
        learnable_temp = False

    Compatible with checkpoints containing ``backbone.*`` or ``encoder.*``
    keys (with optional ``module.`` / ``model.`` prefixes).
    """

    def __init__(self, embedding_size: int = 32) -> None:
        super().__init__()

        resnet = models.resnet18(weights=None)
        num_ftrs = resnet.fc.in_features

        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.encoder = nn.Linear(num_ftrs, embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: RGB image batch of shape ``(B, C, H, W)``.

        Returns:
            Embedding tensor of shape ``(B, embedding_size)``.
        """
        feats = self.backbone(x)         # (B, 512, 1, 1)
        feats = torch.flatten(feats, 1)  # (B, 512)
        embs = self.encoder(feats)       # (B, 32)
        # Important: config.model.normalize_embeddings = False
        return embs


# ---------------------------------------------------------------------------
# Checkpoint resolution
# ---------------------------------------------------------------------------

def get_best_available_checkpoint(experiment_dir: str) -> tuple[str | None, int | None]:
    """Find the best checkpoint in *experiment_dir* by reading TensorBoard metrics.

    Prefers ``downstream/valid/kendalls_tau`` (higher is better). Falls back
    to ``pretrain/valid/total_loss`` (lower is better). If neither metric is
    available, returns the checkpoint with the highest step number.

    Args:
        experiment_dir: Root of an XIRL experiment (must contain
            ``checkpoints/`` and optionally ``tb/``).

    Returns:
        ``(checkpoint_path, step)`` tuple, or ``(None, None)`` if no
        checkpoints exist.
    """
    ckpt_dir = os.path.join(experiment_dir, "checkpoints")
    ckpts = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
    if not ckpts:
        return None, None

    ckpt_steps: list[int] = []
    ckpt_map: dict[int, str] = {}
    for c in ckpts:
        base = os.path.basename(c)
        name, _ = os.path.splitext(base)
        try:
            step = int(name)
            ckpt_steps.append(step)
            ckpt_map[step] = c
        except ValueError:
            pass

    if not ckpt_steps:
        last_ckpt = sorted(ckpts)[-1]
        return last_ckpt, None

    # --- Try to read TensorBoard events ---
    tb_dir = os.path.join(experiment_dir, "tb")
    event_files = glob.glob(os.path.join(tb_dir, "events.out.tfevents.*"))
    if not event_files:
        latest_step = max(ckpt_steps)
        print(
            f"[XIRL Resolver] No TensorBoard files in {tb_dir}. "
            f"Using latest checkpoint on disk: {latest_step}.ckpt"
        )
        return ckpt_map[latest_step], latest_step

    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
        latest_step = max(ckpt_steps)
        print("[XIRL Resolver] tensorboard not available; using latest checkpoint.")
        return ckpt_map[latest_step], latest_step

    eval_data: list[tuple[int, float, str]] = []

    # Prefer Kendall's Tau
    for event_file in event_files:
        ea = event_accumulator.EventAccumulator(
            event_file, size_guidance={event_accumulator.SCALARS: 0}
        )
        try:
            ea.Reload()
            tags = ea.Tags().get("scalars", [])
            if "downstream/valid/kendalls_tau" in tags:
                for event in ea.Scalars("downstream/valid/kendalls_tau"):
                    eval_data.append((event.step, event.value, "Kendall's Tau"))
        except Exception:
            pass

    # Fall back to validation loss
    is_loss = False
    if not eval_data:
        is_loss = True
        for event_file in event_files:
            ea = event_accumulator.EventAccumulator(
                event_file, size_guidance={event_accumulator.SCALARS: 0}
            )
            try:
                ea.Reload()
                tags = ea.Tags().get("scalars", [])
                if "pretrain/valid/total_loss" in tags:
                    for event in ea.Scalars("pretrain/valid/total_loss"):
                        eval_data.append((event.step, event.value, "Validation Loss"))
            except Exception:
                pass

    if not eval_data:
        latest_step = max(ckpt_steps)
        print(
            "[XIRL Resolver] No valid metrics in TensorBoard events. "
            f"Using latest checkpoint on disk: {latest_step}.ckpt"
        )
        return ckpt_map[latest_step], latest_step

    best_ckpt_step: int | None = None
    best_val: float = -1.0 if not is_loss else float("inf")
    best_eval_step: int | None = None

    for ckpt_step in ckpt_steps:
        closest_eval = min(eval_data, key=lambda x: abs(x[0] - ckpt_step))
        eval_step, eval_val, metric_name = closest_eval

        if is_loss:
            if eval_val < best_val:
                best_val = eval_val
                best_ckpt_step = ckpt_step
                best_eval_step = eval_step
        else:
            if eval_val > best_val:
                best_val = eval_val
                best_ckpt_step = ckpt_step
                best_eval_step = eval_step

    print(f"[XIRL Resolver] Best checkpoint on disk: {best_ckpt_step}.ckpt")
    print(
        f"                (Evaluated at step {best_eval_step} "
        f"with {metric_name}: {best_val:.6f})"
    )
    return ckpt_map[best_ckpt_step], best_ckpt_step


def resolve_tcc_checkpoint(path_or_dir: str) -> str:
    """Resolve a TCC checkpoint path.

    If *path_or_dir* points to a file, returns it directly. If it points to
    an experiment directory (or a subdirectory), locates the best checkpoint
    using :func:`get_best_available_checkpoint`.

    Args:
        path_or_dir: Either a direct ``.ckpt`` file path or an experiment
            directory root.

    Returns:
        Absolute path to the resolved checkpoint file.
    """
    if os.path.isfile(path_or_dir):
        return path_or_dir

    exp_dir = path_or_dir
    if not os.path.isdir(exp_dir):
        if "checkpoints" in path_or_dir:
            exp_dir = path_or_dir.split("checkpoints")[0]
        else:
            exp_dir = os.path.dirname(path_or_dir)

    if os.path.isdir(exp_dir):
        ckpt_dir = os.path.join(exp_dir, "checkpoints")
        if os.path.isdir(ckpt_dir):
            best_ckpt, _ = get_best_available_checkpoint(exp_dir)
            if best_ckpt:
                return best_ckpt

            # Fallback to most recent by name
            ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "*.ckpt")))
            if ckpts:
                print(f"[XIRL Resolver] Fallback to most recent checkpoint: {ckpts[-1]}")
                return ckpts[-1]

    return path_or_dir


def load_tcc_model(
    experiment_dir: str,
    device: torch.device | str,
    embedding_size: int = 32,
) -> XIRLResnet18:
    """Instantiate and load pretrained TCC weights from *experiment_dir*.

    Args:
        experiment_dir: Path to the XIRL experiment directory (must contain
            a ``checkpoints/`` subdirectory).
        device: Target device for the model.
        embedding_size: Embedding dimension (must match the checkpoint).

    Returns:
        Loaded ``XIRLResnet18`` model in eval mode.

    Raises:
        RuntimeError: If the checkpoint cannot be loaded.
    """
    tcc = XIRLResnet18(embedding_size=embedding_size).to(device)

    try:
        ckpt_path = resolve_tcc_checkpoint(experiment_dir)
        print(f"[TCC] Loading XIRL weights from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)

        if "model" in ckpt:
            sd = ckpt["model"]
        elif "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        else:
            sd = ckpt

        # Strip DDP / XIRL model-wrapper prefixes
        clean_sd: dict[str, torch.Tensor] = {}
        for k, v in sd.items():
            if k.startswith("module."):
                k = k[len("module."):]
            if k.startswith("model."):
                k = k[len("model."):]
            clean_sd[k] = v

        load_result = tcc.load_state_dict(clean_sd, strict=True)
        print("[TCC] Weights loaded successfully.")
        print("[TCC] missing keys:", load_result.missing_keys)
        print("[TCC] unexpected keys:", load_result.unexpected_keys)

    except Exception as exc:
        raise RuntimeError(
            f"TCC checkpoint failed to load from '{experiment_dir}'."
        ) from exc

    tcc.eval()
    return tcc


# ---------------------------------------------------------------------------
# Goal embedding
# ---------------------------------------------------------------------------

def compute_average_goal_embedding(
    tcc_model: XIRLResnet18,
    preprocess: T.Compose,
    dataset_path: str,
    device: torch.device | str,
) -> torch.Tensor:
    """Compute the mean goal embedding from the last frame of each demo video.

    Args:
        tcc_model: Loaded TCC model (must be in eval mode).
        preprocess: Torchvision transform pipeline (e.g. ``Resize((112, 112))``).
        dataset_path: Directory whose immediate subdirectories are demo videos,
            each containing frame images (``*.jpg`` / ``*.png``).
        device: Device on which embeddings are computed.

    Returns:
        Mean goal embedding of shape ``(embedding_size,)``.

    Raises:
        ValueError: If no valid frames are found in *dataset_path*.
    """
    embeddings: list[torch.Tensor] = []
    video_dirs = sorted(
        d
        for d in (
            os.path.join(dataset_path, name)
            for name in os.listdir(dataset_path)
        )
        if os.path.isdir(d)
    )
    print(f"[TCC] Computing mean goal embedding from {len(video_dirs)} videos in {dataset_path}...")

    for v_dir in video_dirs:
        frames = sorted(
            glob.glob(os.path.join(v_dir, "*.jpg"))
            + glob.glob(os.path.join(v_dir, "*.png"))
        )
        if not frames:
            continue
        last_frame_path = frames[-1]
        try:
            print(f"[TCC] Processing {last_frame_path}...")
            img = io.read_image(last_frame_path)[:3].unsqueeze(0).float().to(device) / 255.0
            with torch.no_grad():
                emb = tcc_model(preprocess(img)).squeeze()
            embeddings.append(emb)
        except Exception as exc:
            print(f"[TCC] Warning: cannot process {last_frame_path}: {exc}")

    if not embeddings:
        raise ValueError(
            f"No valid frames found in dataset '{dataset_path}' for goal embedding."
        )

    stacked = torch.stack(embeddings)
    mean_goal = torch.mean(stacked, dim=0)
    print(f"[TCC] Mean goal embedding computed from {len(embeddings)} final frames.")
    return mean_goal
