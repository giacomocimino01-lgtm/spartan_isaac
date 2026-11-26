#!/usr/bin/env python3
"""Build presentation-ready plots, thumbnails, and policy clips on the lab server."""

from __future__ import annotations

import csv
import math
import re
import shutil
import subprocess
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


REPO = Path("/home/aiprah/Documents/m_dVrk")
EVAL = REPO / "artifacts/evaluations/full_task_soft_alignment_no_prior_N0_4_20ep_seed20261129"
TCC_ROOT = Path("/home/aiprah/Documents/tmp/xirl/sim_pretrain_runs")
DATA_ROOT = Path("/mnt/data/aiprah/data/random_sim_dataset_tcc")
CONTROL_HZ = 10.0
VIDEO_STRIDE = 5

PHASE_COLORS = {
    "phase_0": "#2878B5",
    "phase_1": "#D95319",
    "phase_2": "#2A9D8F",
    "phase_3": "#7B2CBF",
    "phase_4": "#6C757D",
}
TCC_RUNS = {
    "phase_0": ("random_sim_phase0_tcc", 4001),
    "phase_1": ("random_sim_phase1_tcc", 31000),
    "phase_2": ("random_sim_phase2_tcc", 4000),
    "phase_3": ("random_sim_phase3_tcc", 2600),
    "phase_4": ("random_sim_phase4_tcc", 2600),
}
RL_LOGS = {
    "phase_0": REPO / "artifacts/logs/random_sb3_log_sim_phase_0_adaptive075_minimalmask_finetune_3m/progress.csv",
    "phase_1": REPO / "artifacts/logs/random_sb3_log_sim_phase_1_native_minimal_recovery_mixed_replay_probe_300k/progress.csv",
    "phase_2": REPO / "artifacts/logs/random_sb3_log_sim_phase_2_phase1_teacherterminal_replay_kl_ppo_probe400k/progress.csv",
    "phase_3": REPO / "artifacts/logs/random_sb3_log_sim_phase_3_bimanual_minimal_bc80_replay_probe_400k/progress.csv",
}


def style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.titlesize": 14,
            "axes.labelsize": 11.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "grid.alpha": 0.25,
        }
    )


def save(fig: plt.Figure, output: Path, stem: str) -> None:
    fig.savefig(output / f"{stem}.png", dpi=260, bbox_inches="tight")
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def scalars(run: str, tag: str):
    ea = EventAccumulator(str(TCC_ROOT / run / "tb"), size_guidance={"scalars": 0})
    ea.Reload()
    return ea.Scalars(tag)


def build_tcc_plot(output: Path) -> None:
    fig, (ax_loss, ax_tau) = plt.subplots(1, 2, figsize=(13.2, 5.1))
    summaries = []
    for phase, (run, selected) in TCC_RUNS.items():
        color = PHASE_COLORS[phase]
        train = scalars(run, "pretrain/train/total_loss")
        valid = scalars(run, "pretrain/valid/total_loss")
        tau = scalars(run, "downstream/valid/kendalls_tau")
        max_step = max(event.step for event in train)
        ax_loss.plot(
            [100 * event.step / max_step for event in train],
            [event.value for event in train],
            color=color,
            linewidth=1.45,
            alpha=0.55,
        )
        ax_loss.plot(
            [100 * event.step / max_step for event in valid],
            [event.value for event in valid],
            color=color,
            linewidth=2.1,
            marker="o",
            markersize=3.5,
            label=phase,
        )
        ax_tau.plot(
            [100 * event.step / max_step for event in tau],
            [event.value for event in tau],
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=3.5,
            label=phase,
        )
        nearest_loss = min(valid, key=lambda event: abs(event.step - selected))
        nearest_tau = min(tau, key=lambda event: abs(event.step - selected))
        ax_tau.scatter(
            [100 * nearest_tau.step / max_step],
            [nearest_tau.value],
            color=color,
            edgecolor="black",
            linewidth=0.6,
            s=64,
            zorder=5,
        )
        train_count = len([p for p in (DATA_ROOT / "train" / phase).iterdir() if p.is_dir()])
        valid_count = len([p for p in (DATA_ROOT / "valid" / phase).iterdir() if p.is_dir()])
        summaries.append(
            {
                "phase": phase,
                "train_videos_current_dataset": train_count,
                "valid_videos_current_dataset": valid_count,
                "selected_checkpoint": selected,
                "nearest_logged_validation_step": nearest_tau.step,
                "validation_loss": nearest_loss.value,
                "validation_kendalls_tau": nearest_tau.value,
            }
        )
    ax_loss.set_yscale("log")
    ax_loss.set_xlabel("Training progress (%)")
    ax_loss.set_ylabel("TCC regression loss (log scale)")
    ax_loss.set_title("TCC training and validation loss")
    ax_loss.grid()
    ax_loss.legend(frameon=False, ncol=2)
    ax_loss.text(0.02, 0.03, "thin: train   thick + markers: validation", transform=ax_loss.transAxes, fontsize=9, color="#555555")
    ax_tau.set_xlabel("Training progress (%)")
    ax_tau.set_ylabel(r"Validation Kendall's $\tau$")
    ax_tau.set_ylim(0, 1)
    ax_tau.set_title("Temporal alignment quality")
    ax_tau.grid()
    ax_tau.text(0.02, 0.03, "black-edged marker: selected checkpoint\nphase 3/4 metric logged at step 2500 for checkpoint 2600", transform=ax_tau.transAxes, fontsize=8.7, color="#555555")
    fig.suptitle("Phase-specific TCC encoders", fontsize=16, y=1.02)
    fig.tight_layout()
    save(fig, output, "05_tcc_training_validation")
    with (output / "05_tcc_selected_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)


def read_progress(path: Path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def parse_phase4_bc(path: Path):
    pattern = re.compile(
        r"Epoch (\d+)/80 \| Loss: ([0-9.]+) \| Acc L-Arm: \[Verb: ([0-9.]+)%, Tgt: ([0-9.]+)%\] \| Acc R-Arm: \[Verb: ([0-9.]+)%, Tgt: ([0-9.]+)%\]"
    )
    records = []
    text = path.read_text(errors="ignore")
    for match in pattern.finditer(text):
        epoch, loss, *accuracy = match.groups()
        records.append((int(epoch), float(loss), float(np.mean([float(value) for value in accuracy]))))
    return records


def build_policy_training_plot(output: Path) -> None:
    fig, (ax_success, ax_bc) = plt.subplots(1, 2, figsize=(13.2, 5.1))
    for phase, path in RL_LOGS.items():
        rows = read_progress(path)
        x, y = [], []
        for row in rows:
            if row.get("rollout/success_rate", "") and row.get("time/total_timesteps", ""):
                x.append(float(row["time/total_timesteps"]) / 1e6)
                y.append(100 * float(row["rollout/success_rate"]))
        if x:
            ax_success.plot(x, y, color=PHASE_COLORS[phase], linewidth=2, label=phase)
    ax_success.set_xlabel("PPO environment steps (millions)")
    ax_success.set_ylabel("Training-rollout success rate (%)")
    ax_success.set_ylim(-2, 102)
    ax_success.set_title("PPO phases: online rollout learning curves")
    ax_success.grid()
    ax_success.legend(frameon=False, ncol=2)
    ax_success.text(0.02, 0.03, "Not held-out evaluation; curves come from SB3 rollout logs.", transform=ax_success.transAxes, fontsize=8.8, color="#555555")

    bc = parse_phase4_bc(REPO / "artifacts/jobs/phase4_teacher_terminal_bc80_seed42/train_bc.log")
    epochs = [item[0] for item in bc]
    losses = [item[1] for item in bc]
    accuracy = [item[2] for item in bc]
    ax_bc.plot(epochs, losses, color=PHASE_COLORS["phase_4"], linewidth=2.0, label="BC cross-entropy")
    ax_bc.set_yscale("log")
    ax_bc.set_xlabel("BC epoch")
    ax_bc.set_ylabel("Summed four-head CE loss (log scale)")
    ax_bc.set_title("Phase 4: pure behavior cloning")
    ax_bc.grid()
    ax_acc = ax_bc.twinx()
    ax_acc.plot(epochs, accuracy, color="#D95319", linewidth=1.7, alpha=0.8, label="mean head accuracy")
    ax_acc.set_ylabel("Mean training head accuracy (%)")
    ax_acc.set_ylim(70, 100.5)
    lines = ax_bc.get_lines() + ax_acc.get_lines()
    ax_bc.legend(lines, [line.get_label() for line in lines], frameon=False, loc="center right")
    fig.suptitle("Policy training evidence: PPO+BC replay versus BC-only", fontsize=16, y=1.02)
    fig.tight_layout()
    save(fig, output, "06_policy_training_curves")


def build_policy_inference_plot(output: Path) -> None:
    phases = ["phase_0", "phase_1", "phase_2", "phase_3", "phase_4"]
    successes = [4, 49, 49, 48, 20]
    totals = [4, 50, 50, 50, 20]
    protocols = [
        "isolated\n(small n)",
        "isolated\ncanonical reset",
        "isolated\nparked entry",
        "isolated\ncanonical reset",
        "true phase-3\nhandoff",
    ]
    rates = [100 * s / n for s, n in zip(successes, totals)]
    fig, ax = plt.subplots(figsize=(10.4, 5.6))
    bars = ax.bar(phases, rates, color=[PHASE_COLORS[p] for p in phases], alpha=0.82)
    for bar, rate, success, total, protocol in zip(bars, rates, successes, totals, protocols):
        ax.text(bar.get_x() + bar.get_width() / 2, rate + 1.2, f"{success}/{total}", ha="center", fontweight="bold")
        ax.text(bar.get_x() + bar.get_width() / 2, 4, protocol, ha="center", va="bottom", color="white", fontsize=8.4, fontweight="bold")
    ax.set_ylim(0, 107)
    ax.set_ylabel("Recorded inference success rate (%)")
    ax.set_title("Selected policy checkpoints — available deterministic inference evidence")
    ax.grid(axis="y")
    fig.text(0.5, 0.015, "Protocols are not identical. Phase 0 has only four isolated recordings; phase 4 is measured after a real learned phase-3 handoff.", ha="center", fontsize=9, color="#555555")
    fig.subplots_adjust(bottom=0.15)
    save(fig, output, "07_policy_inference_success")


def load_source_frames(path: Path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def frame_for_step(frames, step: int):
    index = max(0, min(len(frames) - 1, (max(1, step) - 1) // VIDEO_STRIDE))
    return frames[index]


def build_phase_contact_sheet(output: Path) -> None:
    frames = []
    labels = [
        "phase 0\n4 rings on green peg",
        "phase 1\n2 red + 2 blue",
        "phase 2\n4 rings at board centre",
        "phase 3\n1 ring per coloured peg",
        "phase 4\n2 rings per outside region",
    ]
    for phase in range(4):
        frames.append(cv2.imread(str(EVAL / f"episode_011_phase_{phase}_transition.png")))
    source = load_source_frames(EVAL / "episode_011_success.mp4")
    frames.append(source[-1])
    fig, axes = plt.subplots(1, 5, figsize=(15.5, 3.35))
    for phase, (ax, frame, label) in enumerate(zip(axes, frames, labels)):
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax.set_title(label, color=PHASE_COLORS[f"phase_{phase}"], fontweight="bold", fontsize=10)
        ax.axis("off")
    fig.suptitle("SPARTAN bimanual peg-and-ring task: five sequential phases", fontsize=15)
    fig.tight_layout()
    save(fig, output, "08_spartan_five_phase_contact_sheet")


def write_real_time_policy_clip(output: Path, frames, phase: str, end_step: int, duration_s: float = 5.0) -> None:
    start_step = max(1, end_step - int(duration_s * CONTROL_HZ) + 1)
    raw = output / f"09_{phase}_policy_completion_raw.mp4"
    final = output / f"09_{phase}_policy_completion.mp4"
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), CONTROL_HZ, (640, 480))
    for step in range(start_step, end_step + 1):
        image = cv2.resize(frame_for_step(frames, step), (640, 480), interpolation=cv2.INTER_AREA)
        cv2.rectangle(image, (0, 0), (640, 34), (0, 0, 0), -1)
        cv2.putText(image, f"{phase} | real control time {(step - start_step) / CONTROL_HZ:.1f} s", (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(image)
    writer.release()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(final)], check=True)
    raw.unlink()


def build_policy_clips(output: Path) -> None:
    frames = load_source_frames(EVAL / "episode_011_success.mp4")
    # Episode 11 transition/end steps from the recorded no-prior run.
    for phase, end_step in {"phase_0": 257, "phase_1": 600, "phase_4": 1367}.items():
        write_real_time_policy_clip(output, frames, phase, end_step)


def copy_existing(output: Path) -> None:
    source_plots = EVAL / "no_prior_ablation_plots"
    source_media = EVAL / "presentation_media"
    for name in [
        "01_closed_loop_outcomes.png",
        "02_no_prior_decision_margins.png",
        "03_no_prior_candidate_score_heatmap.png",
        "04_episode11_phase1_score_trace.png",
    ]:
        shutil.copy2(source_plots / name, output / name)
    for name in [
        "05_pure_decoder_delay_by_edge.png",
        "08_clipA_episode11_phase1_to_phase2.mp4",
        "09_filmstripA_episode11_phase1_to_phase2.png",
        "10_clipB_episode17_phase1_timeout.mp4",
        "12_full_task_episode011_success.mp4",
    ]:
        shutil.copy2(source_media / name, output / f"decoder_{name}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    style()
    copy_existing(args.output)
    build_tcc_plot(args.output)
    build_policy_training_plot(args.output)
    build_policy_inference_plot(args.output)
    build_phase_contact_sheet(args.output)
    build_policy_clips(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
