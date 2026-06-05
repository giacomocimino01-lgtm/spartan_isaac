import pandas as pd
import matplotlib.pyplot as plt
import os

# Load progress data
csv_path = '/home/aiprah/Documents/m_dVrk/sb3_log_sim/progress.csv'
df = pd.read_csv(csv_path)

# Create a figure with subplots
fig, axs = plt.subplots(3, 2, figsize=(15, 15))
fig.suptitle('Training Metrics', fontsize=16)

# 1. Episode Reward
axs[0, 0].plot(df['time/total_timesteps'], df['rollout/ep_rew_mean'], label='Reward', color='blue')
axs[0, 0].set_title('Mean Episode Reward (rollout/ep_rew_mean)')
axs[0, 0].set_xlabel('Timesteps')
axs[0, 0].set_ylabel('Reward')
axs[0, 0].grid(True)

# 2. Episode Length
axs[0, 1].plot(df['time/total_timesteps'], df['rollout/ep_len_mean'], label='Length', color='orange')
axs[0, 1].set_title('Mean Episode Length (rollout/ep_len_mean)')
axs[0, 1].set_xlabel('Timesteps')
axs[0, 1].set_ylabel('Length')
axs[0, 1].grid(True)

# 3. Value Loss
axs[1, 0].plot(df['time/total_timesteps'], df['train/value_loss'], label='Value Loss', color='green')
axs[1, 0].set_title('Value Loss (train/value_loss)')
axs[1, 0].set_xlabel('Timesteps')
axs[1, 0].set_ylabel('Loss')
axs[1, 0].grid(True)

# 4. Policy Loss
axs[1, 1].plot(df['time/total_timesteps'], df['train/policy_loss'], label='Policy Loss', color='red')
axs[1, 1].set_title('Policy Loss (train/policy_loss)')
axs[1, 1].set_xlabel('Timesteps')
axs[1, 1].set_ylabel('Loss')
axs[1, 1].grid(True)

# 5. Entropy Loss
axs[2, 0].plot(df['time/total_timesteps'], df['train/entropy_loss'], label='Entropy Loss', color='purple')
axs[2, 0].set_title('Entropy Loss (train/entropy_loss)')
axs[2, 0].set_xlabel('Timesteps')
axs[2, 0].set_ylabel('Loss')
axs[2, 0].grid(True)

# 6. Explained Variance
axs[2, 1].plot(df['time/total_timesteps'], df['train/explained_variance'], label='Explained Variance', color='brown')
axs[2, 1].set_title('Explained Variance (train/explained_variance)')
axs[2, 1].set_xlabel('Timesteps')
axs[2, 1].set_ylabel('Variance')
axs[2, 1].grid(True)

plt.tight_layout()
plt.subplots_adjust(top=0.95)
out_path = '/home/aiprah/Documents/m_dVrk/sim_training_metrics.png'
plt.savefig(out_path)
print(f'Saved metrics plot to {out_path}')

# Now extract distances from log file
log_path = '/home/aiprah/Documents/m_dVrk/outlog.log'
distances = []
with open(log_path, 'r') as f:
    for line in f:
        if "Episode done. Distance=" in line:
            parts = line.strip().split("Distance=")
            if len(parts) == 2:
                try:
                    dist = float(parts[1])
                    distances.append(dist)
                except ValueError:
                    pass

if distances:
    plt.figure(figsize=(8, 5))
    plt.plot(distances, marker='o', linestyle='-', color='teal')
    plt.title('Final Distance per Episode (from log)')
    plt.xlabel('Logged Episode Index (ENV 0)')
    plt.ylabel('Distance')
    plt.grid(True)
    dist_out_path = '/home/aiprah/Documents/m_dVrk/sim_distance_metrics.png'
    plt.savefig(dist_out_path)
    print(f'Saved distance plot to {dist_out_path}')
