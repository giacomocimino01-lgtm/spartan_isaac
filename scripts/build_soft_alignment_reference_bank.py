#!/usr/bin/env python3
"""Encode complete phase videos into a multi-phase soft-alignment bank."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import torch
import torchvision.io as io
import torchvision.transforms as T


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "source" / "m_dVrk" / "m_dVrk"


def load_repo_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


router_module = load_repo_module(
    "soft_bank_router", PACKAGE_ROOT / "hrl" / "calibrated_phase_router.py"
)
tcc_module = load_repo_module("soft_bank_tcc", PACKAGE_ROOT / "hrl" / "tcc.py")


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        nargs=3,
        action="append",
        required=True,
        metavar=("NAME", "TCC_CHECKPOINT", "REFERENCE_DATASET"),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--reuse_bank",
        action="append",
        default=[],
        help="Reuse every phase contained in an existing reference-bank artifact.",
    )
    parser.add_argument("--capture_hz", type=float, default=2.0)
    parser.add_argument("--embedding_size", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--reference_limit", type=int, default=None)
    return parser.parse_args()


def natural_key(path: Path) -> tuple[int, int | str]:
    return (0, int(path.stem)) if path.stem.isdigit() else (1, path.name)


def video_directories(root: Path, limit: int | None) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Missing reference dataset: {root}")
    values = sorted((path for path in root.iterdir() if path.is_dir()), key=lambda p: p.name)
    if limit is not None:
        values = values[:limit]
    if not values:
        raise ValueError(f"No reference videos under {root}")
    return values


def frame_paths(video: Path) -> list[Path]:
    values = sorted(
        (path for path in video.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES),
        key=natural_key,
    )
    if len(values) < 2:
        raise ValueError(f"Reference video needs at least two frames: {video}")
    return values


def encode_video(
    paths: list[Path],
    model: torch.nn.Module,
    preprocess: T.Compose,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    encoded: list[torch.Tensor] = []
    for offset in range(0, len(paths), batch_size):
        batch_paths = paths[offset : offset + batch_size]
        images = torch.stack(
            [io.read_image(str(path))[:3].float() / 255.0 for path in batch_paths]
        )
        with torch.no_grad():
            encoded.append(model(preprocess(images).to(device)).detach().cpu().float())
    return torch.cat(encoded)


def main() -> None:
    args = parse_args()
    if args.capture_hz <= 0 or args.batch_size <= 0 or args.embedding_size <= 0:
        raise ValueError("capture_hz, batch_size, and embedding_size must be positive")
    if args.reference_limit is not None and args.reference_limit <= 0:
        raise ValueError("reference_limit must be positive")

    phase_names: list[str] = []
    references_by_phase: dict[str, tuple[torch.Tensor, ...]] = {}
    phase_metadata: dict[str, object] = {}
    device = torch.device(args.device)
    preprocess = T.Compose([T.Resize((112, 112), antialias=True)])
    for raw_bank_path in args.reuse_bank:
        bank_path = Path(raw_bank_path).resolve()
        payload = torch.load(bank_path, map_location="cpu", weights_only=False)
        reused = router_module.SoftAlignmentReferenceBank.from_payload(payload)
        for name in reused.phase_names:
            if name in references_by_phase:
                raise ValueError(f"Duplicate reused phase: {name}")
            phase_names.append(name)
            references_by_phase[name] = tuple(reused.references_by_phase[name])
            phase_metadata[name] = {
                "reused_bank": str(bank_path),
                "source_metadata": dict(reused.metadata),
            }
            print(
                f"[soft-bank] Reused {len(references_by_phase[name])} "
                f"references for {name}",
                flush=True,
            )
    for name, raw_checkpoint, raw_dataset in args.phase:
        if name in references_by_phase:
            raise ValueError(f"Duplicate phase: {name}")
        checkpoint = Path(raw_checkpoint).resolve()
        dataset = Path(raw_dataset).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing TCC checkpoint for {name}: {checkpoint}")
        directories = video_directories(dataset, args.reference_limit)
        print(f"[soft-bank] Loading {name} encoder", flush=True)
        model = tcc_module.load_tcc_model(
            str(checkpoint), device=device, embedding_size=args.embedding_size
        )
        references: list[torch.Tensor] = []
        rows: list[dict[str, object]] = []
        for index, directory in enumerate(directories, start=1):
            paths = frame_paths(directory)
            print(
                f"[soft-bank] {name} {index}/{len(directories)}: "
                f"{directory.name} ({len(paths)} frames)",
                flush=True,
            )
            references.append(
                encode_video(paths, model, preprocess, device, args.batch_size)
            )
            rows.append(
                {
                    "trajectory": directory.name,
                    "frames": len(paths),
                    "duration_seconds": (len(paths) - 1) / args.capture_hz,
                }
            )
        phase_names.append(name)
        references_by_phase[name] = tuple(references)
        phase_metadata[name] = {
            "checkpoint": str(checkpoint),
            "dataset": str(dataset),
            "videos": rows,
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    bank = router_module.SoftAlignmentReferenceBank(
        phase_names=tuple(phase_names),
        references_by_phase=references_by_phase,
        metadata={
            "capture_hz": args.capture_hz,
            "embedding_size": args.embedding_size,
            "phases": phase_metadata,
        },
    )
    bank.validate()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bank.to_payload(), output)
    print(f"[soft-bank] Saved {output}", flush=True)


if __name__ == "__main__":
    main()
