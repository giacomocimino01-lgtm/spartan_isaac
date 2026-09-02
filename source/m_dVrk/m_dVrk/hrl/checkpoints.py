"""Checkpoint companion-file discovery shared by training utilities."""

from __future__ import annotations

import re
from pathlib import Path


def vecnormalize_candidates(checkpoint: str | Path) -> list[Path]:
    """Return likely VecNormalize files, most specific first.

    SB3's ``CheckpointCallback`` inserts ``_vecnormalize`` before the step
    suffix, whereas final/best checkpoints append it to the model stem.
    """
    checkpoint = Path(checkpoint)
    candidates = [checkpoint.with_name(f"{checkpoint.stem}_vecnormalize.pkl")]
    step_match = re.fullmatch(r"(.+)_(\d+)_steps", checkpoint.stem)
    if step_match:
        prefix, steps = step_match.groups()
        candidates.insert(0, checkpoint.with_name(f"{prefix}_vecnormalize_{steps}_steps.pkl"))
    return candidates


def find_vecnormalize(checkpoint: str | Path) -> Path | None:
    """Find the checkpoint-specific VecNormalize companion, if present."""
    return next((path for path in vecnormalize_candidates(checkpoint) if path.is_file()), None)
