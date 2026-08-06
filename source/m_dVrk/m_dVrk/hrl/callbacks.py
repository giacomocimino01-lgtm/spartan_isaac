"""SB3 training callbacks for the m_dVrk PPO training loop.

All callbacks are defined here so they can be shared by training scripts
without being redefined inline.
"""

from __future__ import annotations

import csv
import math
import os
from collections import deque

from stable_baselines3.common.callbacks import BaseCallback


class EpisodeCsvLoggerCallback(BaseCallback):
    """Log per-episode metrics to a CSV file during PPO training.

    Writes one row per completed episode with timing, reward, success, and
    task-specific ring placement counts.

    Args:
        file_path: Absolute path to the output CSV file (parent directory
            will be created if it does not exist).
        verbose: SB3 verbosity level (0 = silent).
    """

    def __init__(self, file_path: str, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.file_path = file_path
        self._file = None
        self._writer = None

    def _on_training_start(self) -> None:
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        self._file = open(self.file_path, "w", newline="")
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=[
                "total_timesteps",
                "env_id",
                "episode_reward",
                "episode_length",
                "episode_time_seconds",
                "is_success",
                "rings_complete",
                "terminal_distance",
                "green_peg_ring_count",
                "red_peg_ring_count",
                "blue_peg_ring_count",
                "time_limit_truncated",
            ],
        )
        self._writer.writeheader()
        self._file.flush()
        print(f"[PPO] Episode CSV logger active: {self.file_path}", flush=True)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if infos is None or self._writer is None:
            return True

        wrote_rows = False
        for env_id, info in enumerate(infos):
            episode = info.get("episode")
            if episode is None:
                continue

            self._writer.writerow(
                {
                    "total_timesteps": int(self.num_timesteps),
                    "env_id": env_id,
                    "episode_reward": float(episode.get("r", 0.0)),
                    "episode_length": int(episode.get("l", 0)),
                    "episode_time_seconds": float(episode.get("t", 0.0)),
                    "is_success": bool(info.get("is_success", False)),
                    "rings_complete": bool(info.get("rings_complete", False)),
                    "terminal_distance": float(info.get("terminal_distance", float("nan"))),
                    "green_peg_ring_count": int(info.get("green_peg_ring_count", 0)),
                    "red_peg_ring_count": int(info.get("red_peg_ring_count", 0)),
                    "blue_peg_ring_count": int(info.get("blue_peg_ring_count", 0)),
                    "time_limit_truncated": bool(info.get("TimeLimit.truncated", False)),
                }
            )
            wrote_rows = True

        if wrote_rows and self._file is not None:
            self._file.flush()

        return True

    def _on_training_end(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None


class BestTerminalDistanceCheckpointCallback(BaseCallback):
    """Save the model whenever the rolling mean terminal TCC distance improves.

    Args:
        save_dir: Directory where checkpoints are saved.
        file_prefix: Filename prefix for the saved model and metadata.
        window_size: Number of recent terminal distances used for the rolling mean.
        verbose: SB3 verbosity level.
    """

    def __init__(
        self,
        save_dir: str,
        file_prefix: str,
        window_size: int,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.save_dir = save_dir
        self.file_prefix = file_prefix
        self.window_size = max(1, int(window_size))
        self.best_mean_terminal_distance = float("inf")
        self.recent_terminal_distances: deque[float] = deque(maxlen=self.window_size)

    def _on_training_start(self) -> None:
        os.makedirs(self.save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if infos is None:
            return True

        for info in infos:
            episode = info.get("episode")
            terminal_distance = info.get("terminal_distance")
            if episode is None or terminal_distance is None:
                continue
            terminal_distance = float(terminal_distance)
            if not math.isfinite(terminal_distance):
                continue
            self.recent_terminal_distances.append(terminal_distance)

        if len(self.recent_terminal_distances) < self.window_size:
            return True

        mean_terminal_distance = sum(self.recent_terminal_distances) / len(
            self.recent_terminal_distances
        )
        if mean_terminal_distance >= self.best_mean_terminal_distance:
            return True

        self.best_mean_terminal_distance = mean_terminal_distance

        model_path        = os.path.join(self.save_dir, f"{self.file_prefix}.zip")
        vec_normalize_path = os.path.join(self.save_dir, f"{self.file_prefix}_vecnormalize.pkl")
        metric_path       = os.path.join(self.save_dir, f"{self.file_prefix}_metrics.txt")

        self.model.save(model_path)

        vec_normalize_env = self.model.get_vec_normalize_env()
        if vec_normalize_env is not None:
            vec_normalize_env.save(vec_normalize_path)

        with open(metric_path, "w", encoding="ascii") as f:
            f.write(f"num_timesteps={int(self.num_timesteps)}\n")
            f.write(f"window_size={len(self.recent_terminal_distances)}\n")
            f.write(f"mean_terminal_distance={mean_terminal_distance:.8f}\n")

        if self.verbose > 0:
            print(
                "[Checkpoint] New best terminal distance: "
                f"mean={mean_terminal_distance:.6f} "
                f"over last {len(self.recent_terminal_distances)} episodes"
            )

        return True


class FreezePolicyCallback(BaseCallback):
    """Freeze / unfreeze PPO actor weights around a BC-pretrained checkpoint.

    The actor (``policy_net`` + ``action_net``) is frozen at training start
    and unfrozen after *freeze_timesteps* steps, allowing the critic to warm
    up before the full joint update begins.

    Args:
        freeze_timesteps: Number of timesteps to keep the actor frozen.
            Set to 0 to disable freezing entirely.
        verbose: SB3 verbosity level.
    """

    def __init__(self, freeze_timesteps: int, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.freeze_timesteps = freeze_timesteps
        self.policy_frozen = False

    def _on_training_start(self) -> None:
        if self.freeze_timesteps > 0:
            self.freeze_policy()

    def _on_step(self) -> bool:
        if self.policy_frozen and self.num_timesteps >= self.freeze_timesteps:
            self.unfreeze_policy()
        return True

    def freeze_policy(self) -> None:
        """Freeze actor parameters (``policy_net`` and ``action_net``)."""
        print(
            f"[PPO] Freezing policy (actor) parameters for the first "
            f"{self.freeze_timesteps} timesteps.",
            flush=True,
        )
        freeze_count = 0
        frozen_names: list[str] = []
        for name, param in self.model.policy.named_parameters():
            if "policy_net" in name or "action_net" in name:
                param.requires_grad = False
                param.grad = None
                freeze_count += 1
                frozen_names.append(name)
        print(f"[PPO] Froze {freeze_count} parameters:", flush=True)
        for pname in frozen_names:
            print(f"      - {pname}", flush=True)
        self.policy_frozen = True

    def unfreeze_policy(self) -> None:
        """Re-enable gradient computation for the actor parameters."""
        print(
            f"[PPO] Unfreezing policy (actor) parameters after "
            f"{self.num_timesteps} timesteps.",
            flush=True,
        )
        unfreeze_count = 0
        for name, param in self.model.policy.named_parameters():
            if "policy_net" in name or "action_net" in name:
                param.requires_grad = True
                unfreeze_count += 1
        print(f"[PPO] Unfroze {unfreeze_count} parameters.", flush=True)
        self.policy_frozen = False
