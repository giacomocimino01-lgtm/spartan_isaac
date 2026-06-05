#!/usr/bin/env python3
"""
Focused reward progress plot with rolling averages
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read CSV
csv_file = 'sb3_log_sim/progress.csv'
df = pd.read_csv(csv_file)

# Clean up - remove rows where ep_rew_mean is NaN
df_clean = df[df['rollout/ep_rew_mean'].notna()].copy()
df_clean['total_timesteps_m'] = df_clean['time/total_timesteps'] / 1e6  # Convert to millions

print(f"Total training steps: {df_clean['time/total_timesteps'].max():,.0f}")
print(f"Episodes with reward data: {len(df_clean)}")
if len(df_clean) > 0:
    print(f"Mean reward range: [{df_clean['rollout/ep_rew_mean'].min():.2f}, {df_clean['rollout/ep_rew_mean'].max():.2f}]")
    print(f"Latest mean reward: {df_clean['rollout/ep_rew_mean'].iloc[-1]:.2f}")
    print(f"Success rate (latest): {df_clean['rollout/success_rate'].iloc[-1]:.4f}")

# Calcolo reward medio per step
df_clean['reward_per_step'] = df_clean['rollout/ep_rew_mean'] / df_clean['rollout/ep_len_mean']
print(f"Reward medio per step (inizio): {df_clean['reward_per_step'].iloc[0]:.4f}")
print(f"Reward medio per step (fine): {df_clean['reward_per_step'].iloc[-1]:.4f}")
print(f"Reward medio per step (min, max): [{df_clean['reward_per_step'].min():.4f}, {df_clean['reward_per_step'].max():.4f}]")

# Create focused reward plot
fig, ax = plt.subplots(figsize=(12, 6))

# Plot raw rewards
ax.plot(df_clean['total_timesteps_m'], df_clean['rollout/ep_rew_mean'], 
        'b-', linewidth=0.8, alpha=0.5, label='Raw Episode Reward Mean')

# Add rolling mean (smooth version)
window = min(30, max(5, len(df_clean) // 10))
rolling_mean = df_clean['rollout/ep_rew_mean'].rolling(window=window).mean()
ax.plot(df_clean['total_timesteps_m'], rolling_mean, 
        'r-', linewidth=2.5, label=f'Rolling Mean (window={window})')

# Fill between for visualization
ax.fill_between(df_clean['total_timesteps_m'], df_clean['rollout/ep_rew_mean'], 
                alpha=0.2, color='blue')

ax.set_xlabel('Total Timesteps (Millions)', fontsize=12)
ax.set_ylabel('Episode Reward Mean', fontsize=12)
ax.set_title('Training Progress: Episode Reward Over Time', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11, loc='best')
plt.tight_layout()
plt.savefig('sb3_log_sim/reward_progress.png', dpi=150, bbox_inches='tight')
print(f"\nReward plot saved to: sb3_log_sim/reward_progress.png")

# --- Nuovo plot: reward medio per step ---
fig2, ax2 = plt.subplots(figsize=(12, 6))
ax2.plot(df_clean['total_timesteps_m'], df_clean['reward_per_step'], 'g-', linewidth=1.5, alpha=0.7, label='Reward medio per step')
window2 = min(30, max(5, len(df_clean) // 10))
rolling_rps = df_clean['reward_per_step'].rolling(window=window2).mean()
ax2.plot(df_clean['total_timesteps_m'], rolling_rps, 'orange', linewidth=2, label=f'Rolling Mean (window={window2})')
ax2.set_xlabel('Total Timesteps (Millions)', fontsize=12)
ax2.set_ylabel('Mean reward per step', fontsize=12)
ax2.set_title('Mean reward per step', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=11, loc='best')
plt.tight_layout()
plt.savefig('sb3_log_sim/reward_per_step_progress.png', dpi=150, bbox_inches='tight')
print(f"Plot reward medio per step salvato in: sb3_log_sim/reward_per_step_progress.png")

# Print statistics
print(f"\nStatistics:")
print(f"  Initial mean reward: {df_clean['rollout/ep_rew_mean'].iloc[0]:.2f}")
print(f"  Final mean reward: {df_clean['rollout/ep_rew_mean'].iloc[-1]:.2f}")
print(f"  Best mean reward: {df_clean['rollout/ep_rew_mean'].max():.2f}")
print(f"  Worst mean reward: {df_clean['rollout/ep_rew_mean'].min():.2f}")

if len(df_clean) > 1:
    improvement = df_clean['rollout/ep_rew_mean'].iloc[-1] - df_clean['rollout/ep_rew_mean'].iloc[0]
    print(f"  Total improvement: {improvement:.2f}")

plt.close()
