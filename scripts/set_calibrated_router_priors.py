#!/usr/bin/env python3
"""Attach a sparse annotated transition prior to a calibrated router artifact."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "source" / "m_dVrk" / "m_dVrk" / "hrl" / "calibrated_phase_router.py"


def load_router_module():
    spec = importlib.util.spec_from_file_location("sparse_prior_router", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load router module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Existing calibrated_router.pt")
    parser.add_argument("--output", required=True, help="Output artifact path")
    parser.add_argument(
        "--transition_count",
        nargs=3,
        action="append",
        required=True,
        metavar=("FROM_PHASE", "TO_PHASE", "COUNT"),
        help="Annotated transition count; repeat once per observed edge.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    router = load_router_module()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    artifact = router.CalibratedRouterArtifact.from_payload(
        torch.load(input_path, map_location="cpu")
    )

    counts: dict[tuple[str, str], int] = {}
    for source, target, raw_count in args.transition_count:
        edge = (source, target)
        if edge in counts:
            raise ValueError(f"Duplicate transition count for {source}->{target}")
        try:
            count = int(raw_count)
        except ValueError as exc:
            raise ValueError(
                f"Invalid transition count for {source}->{target}: {raw_count!r}"
            ) from exc
        if count <= 0:
            raise ValueError("Transition counts must be positive integers")
        counts[edge] = count

    priors = router.transition_priors_from_counts(counts, artifact.phase_names)
    updated = replace(
        artifact,
        transition_priors=priors,
        transition_counts=counts,
        metadata={
            **dict(artifact.metadata),
            "transition_prior_source": "annotated_transition_counts",
            "transition_prior_parent_artifact": str(input_path),
        },
    )
    updated.validate()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(updated.to_payload(), output_path)
    print(f"Saved sparse-prior artifact: {output_path}")
    for (source, target), probability in sorted(priors.items()):
        print(f"{source}->{target}: count={counts[(source, target)]} prior={probability:.8f}")


if __name__ == "__main__":
    main()
