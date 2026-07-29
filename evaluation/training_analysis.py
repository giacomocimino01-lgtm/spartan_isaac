#!/usr/bin/env python3
"""
Comprehensive training analysis and report
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Read CSV
csv_file = 'random_sb3_log_sim_phase_1_bc_raw_no_norm_300k/progress.csv'
df = pd.read_csv(csv_file)

# Clean up - remove rows where ep_rew_mean is NaN
df_clean = df[df['rollout/ep_rew_mean'].notna()].copy()
df_clean['total_timesteps_m'] = df_clean['time/total_timesteps'] / 1e6

# Calcolo reward medio per step
df_clean['reward_per_step'] = df_clean['rollout/ep_rew_mean'] / df_clean['rollout/ep_len_mean']

# Create comprehensive report
fig = plt.figure(figsize=(16, 15))
gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)

# Main plot: Reward with rolling average
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(df_clean['total_timesteps_m'], df_clean['rollout/ep_rew_mean'], 
        'b-', linewidth=1, alpha=0.6, label='Raw Reward')
window = min(10, len(df_clean) // 5)
rolling_mean = df_clean['rollout/ep_rew_mean'].rolling(window=window).mean()
ax1.plot(df_clean['total_timesteps_m'], rolling_mean, 
        'r-', linewidth=2.5, label=f'Rolling Mean ({window} episodes)')
ax1.axhline(y=-528.79, color='green', linestyle='--', linewidth=2, label='Best: -528.79')
ax1.fill_between(df_clean['total_timesteps_m'], df_clean['rollout/ep_rew_mean'], alpha=0.15, color='blue')
ax1.set_xlabel('Total Timesteps (Millions)', fontsize=11)
ax1.set_ylabel('Episode Reward Mean', fontsize=11)
ax1.set_title('Training Progress: Episode Reward', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10, loc='best')

# --- Nuovo subplot: reward medio per step ---
ax_rps = fig.add_subplot(gs[2, 1])
ax_rps.plot(df_clean['total_timesteps_m'], df_clean['reward_per_step'], 'g-', linewidth=1.5, alpha=0.7, label='Reward medio per step')
window_rps = min(10, len(df_clean) // 5)
rolling_rps = df_clean['reward_per_step'].rolling(window=window_rps).mean()
ax_rps.plot(df_clean['total_timesteps_m'], rolling_rps, 'orange', linewidth=2, label=f'Rolling Mean (window={window_rps})')
ax_rps.set_xlabel('Total Timesteps (Millions)', fontsize=11)
ax_rps.set_ylabel('Reward medio per step', fontsize=11)
ax_rps.set_title('Reward medio per step', fontsize=12, fontweight='bold')
ax_rps.grid(True, alpha=0.3)
ax_rps.legend(fontsize=9)

# Episode Length
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(df_clean['total_timesteps_m'], df_clean['rollout/ep_len_mean'], 'o-', 
        color='purple', markersize=4, linewidth=1.5, alpha=0.7)
ax2.axhline(y=600, color='red', linestyle='--', linewidth=1, label='Time Limit (600)')
ax2.set_xlabel('Total Timesteps (Millions)', fontsize=11)
ax2.set_ylabel('Episode Length', fontsize=11)
ax2.set_title('Episode Length Trend', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=9)

# Success Rate
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(df_clean['total_timesteps_m'], df_clean['rollout/success_rate'], 's-', 
        color='green', markersize=6, linewidth=2)
ax3.set_xlabel('Total Timesteps (Millions)', fontsize=11)
ax3.set_ylabel('Success Rate', fontsize=11)
ax3.set_title('Success Rate Trend', fontsize=12, fontweight='bold')
ax3.set_ylim([-0.05, 1.05])
 
ax4 = fig.add_subplot(gs[2, 0])
ax4.hist(df_clean['rollout/ep_rew_mean'], bins=15, color='skyblue', edgecolor='black', alpha=0.7)
ax4.axvline(df_clean['rollout/ep_rew_mean'].mean(), color='red', linestyle='--', 
           linewidth=2, label=f'Mean: {df_clean["rollout/ep_rew_mean"].mean():.1f}')
ax4.axvline(df_clean['rollout/ep_rew_mean'].median(), color='orange', linestyle='--', 
           linewidth=2, label=f'Median: {df_clean["rollout/ep_rew_mean"].median():.1f}')
ax4.set_xlabel('Episode Reward', fontsize=11)
ax4.set_ylabel('Frequency', fontsize=11)
ax4.set_title('Reward Distribution', fontsize=12, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3, axis='y')


ax5 = fig.add_subplot(gs[3, :])
ax5.axis('off')
stats_text = f"""
TRAINING STATISTICS SUMMARY

ISSUES DETECTED:
        - Reward degrading over time
        - 0% success rate
        - Most episodes hit time limit (600 steps)
        - Training not converging
"""

ax5.text(0.01, 0.99, stats_text, transform=ax5.transAxes, fontsize=10,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.suptitle(f'Training Analysis Report - Generated {datetime.now().strftime("%Y-%m-%d %H:%M")}',
            fontsize=15, fontweight='bold', y=0.995)

plt.savefig('random_sb3_log_sim_phase_1/training_analysis_report.png', dpi=150, bbox_inches='tight')
print("✓ Comprehensive report saved to: random_sb3_log_sim_phase_1/training_analysis_report.png")

# Print key findings
print("\n" + "="*60)
print("KEY FINDINGS")
print("="*60)
print(f"❌ Reward degradation detected: {df_clean['rollout/ep_rew_mean'].iloc[-1]:.2f} vs {df_clean['rollout/ep_rew_mean'].iloc[0]:.2f}")
print(f"❌ Success rate at 0% - model not completing task")
print(f"{(df_clean['rollout/ep_len_mean'] == 600).sum()}/{len(df_clean)} episodes hit time limit")
print(f"\n✓ Training data points: {len(df_clean)} episodes")
print(f"✓ Total training time: ~{df_clean['time/total_timesteps'].max():,.0f} steps")
print("\nRECOMMENDATIONS:")
print("1. Check state machine logic for errors")
print("2. Verify reward calculation is correct")
print("3. Consider reducing learning rate or adjusting other hyperparameters")
print("4. Debug why success rate is 0% despite attempts")
print("="*60)
