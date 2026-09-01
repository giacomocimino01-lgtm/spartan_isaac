#!/usr/bin/env python3
"""Comprehensive training analysis and report.

Usage:
    python evaluation/training_analysis.py --task_phase phase_0
    python evaluation/training_analysis.py --task_phase phase_1
"""
import argparse
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ─── Args ─────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Generate training analysis report.")
parser.add_argument(
    "--task_phase",
    type=str,
    choices=["phase_0", "phase_1"],
    default=None,
    help="Task phase to analyse. If omitted, the most recent log folder is used.",
)
parser.add_argument(
    "--log_dir",
    type=str,
    default=None,
    help="Override: explicit path to the log folder containing progress.csv.",
)
args = parser.parse_args()


# ─── Paths ────────────────────────────────────────────────────────────────────

def _load_project_paths():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cfg_path = os.path.join(repo_root, "configs", "defaults.yaml")
    try:
        import yaml
        if os.path.exists(cfg_path):
            with open(cfg_path, "r") as f:
                cfg = yaml.safe_load(f) or {}
            return cfg.get("paths", {})
    except Exception:
        pass
    return {}


_paths = _load_project_paths()
ARTIFACTS_DIR = _paths.get("artifacts_dir", "artifacts/")
LOGS_DIR = os.path.join(ARTIFACTS_DIR, "logs")

# Determine log directory
if args.log_dir:
    log_dir = args.log_dir
elif args.task_phase:
    # Try canonical name first
    candidate = os.path.join(LOGS_DIR, f"random_sb3_log_sim_{args.task_phase}")
    if os.path.isdir(candidate):
        log_dir = candidate
    else:
        # Fall back to most recent folder containing the phase name
        matches = [
            d for d in os.listdir(LOGS_DIR)
            if args.task_phase in d and os.path.isdir(os.path.join(LOGS_DIR, d))
        ]
        if not matches:
            print(f"[ERROR] No log folder found for {args.task_phase} in {LOGS_DIR}")
            sys.exit(1)
        matches.sort(key=lambda d: os.path.getmtime(os.path.join(LOGS_DIR, d)), reverse=True)
        log_dir = os.path.join(LOGS_DIR, matches[0])
else:
    # No phase specified: pick the most recently modified folder
    all_dirs = [d for d in os.listdir(LOGS_DIR) if os.path.isdir(os.path.join(LOGS_DIR, d))]
    if not all_dirs:
        print(f"[ERROR] No log folders found in {LOGS_DIR}")
        sys.exit(1)
    all_dirs.sort(key=lambda d: os.path.getmtime(os.path.join(LOGS_DIR, d)), reverse=True)
    log_dir = os.path.join(LOGS_DIR, all_dirs[0])

csv_file = os.path.join(log_dir, "progress.csv")
if not os.path.exists(csv_file):
    print(f"[ERROR] progress.csv not found in {log_dir}")
    sys.exit(1)

# Infer task phase label for display
log_dir_name = os.path.basename(log_dir)
if "phase_0" in log_dir_name:
    phase_label = "Phase 0"
elif "phase_1" in log_dir_name:
    phase_label = "Phase 1"
else:
    phase_label = log_dir_name

print(f"[INFO] Loading: {csv_file}")
print(f"[INFO] Task phase: {phase_label}")

# ─── Load & clean ─────────────────────────────────────────────────────────────

df = pd.read_csv(csv_file)
df_clean = df[df["rollout/ep_rew_mean"].notna()].copy()
df_clean["total_timesteps_m"] = df_clean["time/total_timesteps"] / 1e6
df_clean["reward_per_step"] = df_clean["rollout/ep_rew_mean"] / df_clean["rollout/ep_len_mean"]

# ─── Dynamic summary ──────────────────────────────────────────────────────────

rew_first   = df_clean["rollout/ep_rew_mean"].iloc[0]
rew_last    = df_clean["rollout/ep_rew_mean"].iloc[-1]
rew_best    = df_clean["rollout/ep_rew_mean"].max()
rew_mean    = df_clean["rollout/ep_rew_mean"].mean()
rew_median  = df_clean["rollout/ep_rew_mean"].median()
total_steps = df_clean["time/total_timesteps"].max()
n_episodes  = len(df_clean)
window      = min(10, n_episodes // 5)

has_success = "rollout/success_rate" in df_clean.columns
if has_success:
    final_success = df_clean["rollout/success_rate"].iloc[-1]
    best_success  = df_clean["rollout/success_rate"].max()
    mean_success  = df_clean["rollout/success_rate"].mean()
else:
    final_success = best_success = mean_success = float("nan")

ep_len_last  = df_clean["rollout/ep_len_mean"].iloc[-1]
time_limit   = 600
pct_timeout  = (df_clean["rollout/ep_len_mean"] >= time_limit - 1).mean() * 100

rew_improved = rew_last > rew_first  # less-negative = better

# Build status lines dynamically
status_lines = []

if rew_improved:
    delta = rew_last - rew_first
    status_lines.append(f"✓ Reward improved by {delta:+.1f}  ({rew_first:.1f} → {rew_last:.1f})")
else:
    delta = rew_first - rew_last
    status_lines.append(f"✗ Reward degraded by {delta:.1f}  ({rew_first:.1f} → {rew_last:.1f})")

if has_success:
    if final_success >= 0.7:
        status_lines.append(f"✓ Final success rate: {final_success*100:.1f}%  (best: {best_success*100:.1f}%)")
    elif final_success >= 0.3:
        status_lines.append(f"~ Success rate converging: {final_success*100:.1f}%  (best: {best_success*100:.1f}%)")
    else:
        status_lines.append(f"✗ Success rate still low: {final_success*100:.1f}%")

if pct_timeout < 10:
    status_lines.append(f"✓ Episodes completing well (only {pct_timeout:.1f}% timeouts)")
else:
    status_lines.append(f"✗ {pct_timeout:.1f}% episodes hit time limit ({time_limit} steps)")

# Recommendations
recommendations = []
if not rew_improved:
    recommendations.append("• Reduce learning rate or check reward shaping")
if has_success and final_success < 0.3:
    recommendations.append("• Verify state machine / task configuration")
if has_success and best_success > final_success + 0.15:
    recommendations.append("• Possible overfitting — try early stopping or reduce entropy coef")
if not recommendations:
    recommendations.append("• Training looks healthy — consider a longer run or fine-tuning")

summary_text = (
    f"TRAINING SUMMARY — {phase_label.upper()}\n"
    f"{'─'*44}\n"
    + "\n".join(status_lines)
    + "\n\n"
    f"Total steps:     {total_steps:,.0f}\n"
    f"Episodes logged: {n_episodes}\n"
    f"Best reward:     {rew_best:.2f}\n"
    f"Mean reward:     {rew_mean:.2f}\n"
    f"Avg episode len: {ep_len_last:.0f} steps\n"
    f"\nRECOMMENDATIONS:\n"
    + "\n".join(recommendations)
)

# ─── Plot ─────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(16, 15))
gs  = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3)

# 1) Reward curve
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(df_clean["total_timesteps_m"], df_clean["rollout/ep_rew_mean"],
         "b-", linewidth=1, alpha=0.5, label="Raw Reward")
rolling_mean = df_clean["rollout/ep_rew_mean"].rolling(window=window).mean()
ax1.plot(df_clean["total_timesteps_m"], rolling_mean,
         "r-", linewidth=2.5, label=f"Rolling Mean ({window} ep.)")
ax1.axhline(y=rew_best, color="green", linestyle="--", linewidth=2,
            label=f"Best: {rew_best:.2f}")
ax1.fill_between(df_clean["total_timesteps_m"], df_clean["rollout/ep_rew_mean"],
                 alpha=0.12, color="blue")
ax1.set_xlabel("Total Timesteps (Millions)", fontsize=11)
ax1.set_ylabel("Episode Reward Mean", fontsize=11)
ax1.set_title(f"[{phase_label}] Training Progress: Episode Reward", fontsize=13, fontweight="bold")
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10, loc="best")

# 2) Episode length
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(df_clean["total_timesteps_m"], df_clean["rollout/ep_len_mean"],
         "o-", color="purple", markersize=3, linewidth=1.5, alpha=0.7)
ax2.axhline(y=time_limit, color="red", linestyle="--", linewidth=1,
            label=f"Time Limit ({time_limit})")
ax2.set_xlabel("Total Timesteps (Millions)", fontsize=11)
ax2.set_ylabel("Episode Length (steps)", fontsize=11)
ax2.set_title(f"[{phase_label}] Episode Length Trend", fontsize=12, fontweight="bold")
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=9)

# 3) Success rate
ax3 = fig.add_subplot(gs[1, 1])
if has_success:
    ax3.plot(df_clean["total_timesteps_m"], df_clean["rollout/success_rate"],
             "s-", color="green", markersize=5, linewidth=2)
    ax3.set_ylim([-0.05, 1.05])
    ax3.axhline(y=best_success, color="darkgreen", linestyle="--",
                linewidth=1.5, label=f"Best: {best_success:.2%}")
    ax3.legend(fontsize=9)
else:
    ax3.text(0.5, 0.5, "No success_rate column", ha="center", va="center", transform=ax3.transAxes)
ax3.set_xlabel("Total Timesteps (Millions)", fontsize=11)
ax3.set_ylabel("Success Rate", fontsize=11)
ax3.set_title(f"[{phase_label}] Success Rate Trend", fontsize=12, fontweight="bold")
ax3.grid(True, alpha=0.3)

# 4) Reward distribution
ax4 = fig.add_subplot(gs[2, 0])
ax4.hist(df_clean["rollout/ep_rew_mean"], bins=20, color="skyblue", edgecolor="black", alpha=0.75)
ax4.axvline(rew_mean,   color="red",    linestyle="--", linewidth=2, label=f"Mean: {rew_mean:.1f}")
ax4.axvline(rew_median, color="orange", linestyle="--", linewidth=2, label=f"Median: {rew_median:.1f}")
ax4.set_xlabel("Episode Reward", fontsize=11)
ax4.set_ylabel("Frequency", fontsize=11)
ax4.set_title(f"[{phase_label}] Reward Distribution", fontsize=12, fontweight="bold")
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3, axis="y")

# 5) Reward per step
ax5 = fig.add_subplot(gs[2, 1])
ax5.plot(df_clean["total_timesteps_m"], df_clean["reward_per_step"],
         "g-", linewidth=1.2, alpha=0.7, label="Reward/step")
rolling_rps = df_clean["reward_per_step"].rolling(window=window).mean()
ax5.plot(df_clean["total_timesteps_m"], rolling_rps,
         color="orange", linewidth=2, label=f"Rolling Mean (w={window})")
ax5.set_xlabel("Total Timesteps (Millions)", fontsize=11)
ax5.set_ylabel("Reward per Step", fontsize=11)
ax5.set_title(f"[{phase_label}] Reward per Step", fontsize=12, fontweight="bold")
ax5.grid(True, alpha=0.3)
ax5.legend(fontsize=9)

# 6) Dynamic summary box
ax6 = fig.add_subplot(gs[3, :])
ax6.axis("off")
ax6.text(0.01, 0.99, summary_text, transform=ax6.transAxes, fontsize=10,
         verticalalignment="top", fontfamily="monospace",
         bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

plt.suptitle(
    f"Training Analysis Report [{phase_label}] — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
    fontsize=15, fontweight="bold", y=0.995,
)

# ─── Save ─────────────────────────────────────────────────────────────────────

out_report = os.path.join(ARTIFACTS_DIR, "reports", log_dir_name, "training_analysis_report.png")
os.makedirs(os.path.dirname(out_report), exist_ok=True)
plt.savefig(out_report, dpi=150, bbox_inches="tight")
print(f"✓ Report saved to: {out_report}")

# ─── CLI summary ──────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print(summary_text)
print("=" * 60)
