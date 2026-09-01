"""DVRKVisionHRLWrapper — the single VecEnv used by all m_dVrk scripts.

This module contains the one canonical implementation of the HRL wrapper.
Training (parallel_run.py), demo collection (collect_ppo_demos.py), and
evaluation (eval_run.py) must ALL import from here to guarantee identical
observation spaces, action sanitisation, and reward computation.

Key responsibilities:
  - Wraps IsaacLab ManagerBasedRLEnv into an SB3-compatible VecEnv.
  - Maintains the TCC embedding ring buffer.
  - Builds the 244-dim observation (emb_stack + aux + geom) via hrl.observations.
  - Sanitises discrete HRL commands and applies them to the SPARTAN state machine.
  - Computes the TCC distance reward + invalid-command penalty.
  - Manages episode resets, settle steps, success detection.
  - Optionally dumps success snapshots and videos.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch
import torchvision.io as io
import torchvision.transforms as T
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from m_dVrk.hrl.constants import (
    VERB_MAP,
    TARGET_MAP,
    VERB_TO_ID,
    TARGET_TO_ID,
    RING_NAMES,
    PEG_NAMES,
    RING_TARGETS,
    IDLE_ACTION,
    STACK_SIZE,
    EMB_DIM,
    AUX_DIM,
    GEOM_DIM,
    OBS_DIM,
    INVALID_COMMAND_PENALTY,
    REWARD_DEBUG_DUMP_INTERVAL,
    REWARD_DEBUG_DUMP_ENV_IDS,
    REWARD_DEBUG_DUMP_DIR,
    REWARD_DEBUG_DUMP_MAX_PER_ENV,
    SUCCESS_SNAPSHOT_PROB,
    SUCCESS_VIDEO_PROB,
    SUCCESS_DUMP_DIR,
    SUCCESS_VIDEO_DIR,
    SUCCESS_VIDEO_FRAME_SKIP,
    PEG_GREEN,
    PEG_RED,
    PEG_BLUE,
)
from m_dVrk.hrl.observations import get_aux_obs, get_geom_obs, build_obs
from m_dVrk.hrl.success import peg_ring_count, task_success
from m_dVrk.hrl.rewards import compute_tcc_reward, tcc_distance
from m_dVrk.hrl.tcc import compute_average_goal_embedding
from m_dVrk.controllers.ring_sync import (
    get_scene_entity_positions_w,
    sync_attached_and_frozen_rings,
)


class DVRKVisionHRLWrapper(VecEnv):
    """Vectorised HRL wrapper for the dVRK Peg-and-Ring task.

    Inherits from ``stable_baselines3.common.vec_env.VecEnv`` and wraps
    an IsaacLab ``ManagerBasedRLEnv``.

    Args:
        isaac_env: The IsaacLab environment.
        state_machine: ``SPARTANStateMachine`` initialised on the same env.
        tcc_model: Loaded ``XIRLResnet18`` in eval mode. Pass ``None`` to
            disable the TCC embedding (obs will be zeros in that region).
        task_phase: ``"phase_0"`` or ``"phase_1"``.
        reward_debug_dump_interval: How often (in steps) to dump reward
            debug images. Set to 0 to disable.
        reward_debug_dump_env_ids: Environment indices to dump images for.
        reward_debug_dump_dir: Output directory for reward debug images.
        reward_debug_dump_max_per_env: Maximum number of dumps per env.
        success_snapshot_prob: Probability of saving a PNG on each success.
        success_video_prob: Probability of saving an MP4 on each success.
        success_dump_dir: Directory for success snapshots.
        success_video_dir: Directory for success videos.
        success_video_frame_skip: Accumulate 1 frame every N steps.
    """

    def __init__(
        self,
        isaac_env,
        state_machine,
        tcc_model=None,
        task_phase: str = "phase_0",
        reward_debug_dump_interval: int = REWARD_DEBUG_DUMP_INTERVAL,
        reward_debug_dump_env_ids: tuple = REWARD_DEBUG_DUMP_ENV_IDS,
        reward_debug_dump_dir: str = REWARD_DEBUG_DUMP_DIR,
        reward_debug_dump_max_per_env: int = REWARD_DEBUG_DUMP_MAX_PER_ENV,
        success_snapshot_prob: float = SUCCESS_SNAPSHOT_PROB,
        success_video_prob: float = SUCCESS_VIDEO_PROB,
        success_dump_dir: str = SUCCESS_DUMP_DIR,
        success_video_dir: str = SUCCESS_VIDEO_DIR,
        success_video_frame_skip: int = SUCCESS_VIDEO_FRAME_SKIP,
    ) -> None:
        self.env = isaac_env
        self.sm = state_machine
        self.tcc_model = tcc_model
        self.task_phase = task_phase

        self.num_envs = isaac_env.num_envs
        self.stack_size = STACK_SIZE
        self.emb_dim = EMB_DIM

        # --- Spaces ---
        act_space = spaces.MultiDiscrete(
            [len(VERB_MAP), len(TARGET_MAP), len(VERB_MAP), len(TARGET_MAP)]
        )
        obs_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(OBS_DIM,),
            dtype=np.float32,
        )
        self.render_mode = None
        super().__init__(self.num_envs, obs_space, act_space)

        # --- Reward debug dump ---
        self.reward_debug_dump_interval = max(int(reward_debug_dump_interval), 0)
        self.reward_debug_dump_dir = reward_debug_dump_dir
        self.reward_debug_dump_max_per_env = max(int(reward_debug_dump_max_per_env), 0)
        self.reward_debug_dump_env_ids = self._sanitize_dump_env_ids(reward_debug_dump_env_ids)
        self.reward_debug_dump_counts = [0] * self.num_envs
        self.reward_debug_last_dump_step = [-1] * self.num_envs

        if (self.reward_debug_dump_interval > 0
                and self.reward_debug_dump_env_ids
                and self.reward_debug_dump_max_per_env > 0):
            os.makedirs(self.reward_debug_dump_dir, exist_ok=True)
            print(
                f"[RewardDebug] Dump active: every={self.reward_debug_dump_interval} steps, "
                f"envs={self.reward_debug_dump_env_ids}, dir={self.reward_debug_dump_dir}"
            )

        # --- Success snapshot / video ---
        self.success_snapshot_prob    = success_snapshot_prob
        self.success_video_prob       = success_video_prob
        self.success_dump_dir         = success_dump_dir
        self.success_video_dir        = success_video_dir
        self.success_video_frame_skip = success_video_frame_skip
        self._success_frame_buffers   = [[] for _ in range(self.num_envs)]
        self._ep_step_for_video       = [0] * self.num_envs
        os.makedirs(self.success_dump_dir, exist_ok=True)
        os.makedirs(self.success_video_dir, exist_ok=True)
        print(
            f"[SuccessDump] snapshot_prob={self.success_snapshot_prob:.0%}, "
            f"video_prob={self.success_video_prob:.0%}, "
            f"frame_skip={self.success_video_frame_skip}"
        )

        # --- Settle steps (allow rendering to catch up after ring placement) ---
        self.SETTLE_STEPS = 5
        self.settle_steps_remaining = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.env.device
        )

        # --- Action / embedding state ---
        self.prev_actions = IDLE_ACTION.to(self.env.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.last_override_flags = torch.zeros(
            (self.num_envs, 2), dtype=torch.float32, device=self.env.device
        )
        self.emb_buffer = torch.zeros(
            (self.num_envs, self.stack_size, self.emb_dim),
            dtype=torch.float32,
            device=self.env.device,
        )
        self.preprocess = T.Compose([T.Resize((112, 112), antialias=True)])
        self.current_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.env.device)
        self.goal_embedding = None
        self.actions = None

        # --- Robot tip indices ---
        _robot_r = self.env.scene["robot_right"]
        _robot_l = self.env.scene["robot_left"]
        self._tip_idx_r = _robot_r.find_bodies("psm_tool_tip_link")[0][0]
        self._tip_idx_l = _robot_l.find_bodies("psm_tool_tip_link")[0][0]

        # --- Static peg positions (local frame, cached once) ---
        self._ring_names = RING_NAMES
        self._peg_names  = PEG_NAMES
        self._cached_peg_pos_local = self._cache_static_peg_positions_local()

    # -----------------------------------------------------------------------
    # Initialisation helpers
    # -----------------------------------------------------------------------

    def _cache_static_peg_positions_local(self) -> torch.Tensor:
        origins = self.env.scene.env_origins
        peg_pos_local = [
            get_scene_entity_positions_w(self.env, name) - origins
            for name in self._peg_names
        ]
        return torch.stack(peg_pos_local, dim=1)  # (N, 4, 3)

    def _sanitize_dump_env_ids(self, env_ids_spec) -> list[int]:
        env_ids = []
        for raw in env_ids_spec:
            try:
                env_id = int(raw)
            except (ValueError, TypeError):
                print(f"[RewardDebug] Ignoring invalid env id: {raw}")
                continue
            if 0 <= env_id < self.num_envs:
                env_ids.append(env_id)
            else:
                print(f"[RewardDebug] Ignoring out-of-range env id: {env_id}")
        return sorted(set(env_ids))

    # -----------------------------------------------------------------------
    # Observation
    # -----------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        emb_obs = self.emb_buffer  # delegate to build_obs for the view
        aux_obs  = get_aux_obs(
            self.sm,
            self.prev_actions,
            self.last_override_flags,
            self.num_envs,
            self.env.device,
        )
        geom_obs = get_geom_obs(
            self.env,
            self.sm,
            self._ring_names,
            self._peg_names,
            self._tip_idx_r,
            self._tip_idx_l,
            self._cached_peg_pos_local,
            self.task_phase,
            self.num_envs,
        )
        return build_obs(emb_obs, aux_obs, geom_obs, self.num_envs)

    # -----------------------------------------------------------------------
    # TCC embeddings
    # -----------------------------------------------------------------------

    def _get_batched_embeddings(self, dump_debug: bool = False) -> torch.Tensor:
        rgb_data = self.env.scene.sensors["camera"].data.output["rgb"]
        model_device = next(self.tcc_model.parameters()).device

        raw_img_batch = (
            rgb_data[:, :, :, :3]
            .permute(0, 3, 1, 2)
            .float()
            .to(model_device)
            / 255.0
        )
        img_batch = self.preprocess(raw_img_batch)

        if dump_debug:
            self._maybe_dump_reward_images(raw_img_batch, img_batch)

        with torch.no_grad():
            emb = self.tcc_model(img_batch)

        return emb.to(self.env.device)

    def compute_and_set_goal_embedding(self, tcc_model, proc_transform, dataset_path: str, raise_on_error: bool = True) -> None:
        """Compute a mean goal embedding from an XIRL/TCC dataset and assign it.

        Parameters
        - tcc_model: the XIRLResnet18 model used for embedding computation
        - proc_transform: torchvision transform to apply to images
        - dataset_path: path to the dataset root
        - raise_on_error: if True, re-raise exceptions from the computation
        """
        try:
            goal_emb = compute_average_goal_embedding(tcc_model, proc_transform, dataset_path, self.env.device)
            self.goal_embedding = goal_emb
            print(f"[Wrapper] Goal embedding set from dataset: {dataset_path}")
        except Exception as e:
            print(f"[Wrapper] WARNING: failed to compute goal embedding from {dataset_path}: {e}")
            self.goal_embedding = torch.zeros(self.emb_dim, device=self.env.device)
            if raise_on_error:
                raise

    # -----------------------------------------------------------------------
    # Action sanitisation
    # -----------------------------------------------------------------------

    def _sanitize_arm_command(
        self,
        env_id: int,
        subject: str,
        verb: str,
        target: str,
    ) -> tuple[str, str, bool]:
        """Validate a single-arm command, returning (verb, target, is_invalid)."""
        if verb == "idle":
            return "idle", "None", False
        if target == "None":
            return "idle", "None", True

        last_target = (
            self.sm.last_target_r[env_id]
            if subject == "right_arm"
            else self.sm.last_target_l[env_id]
        )
        attached_target = (
            self.sm.attached_target_r[env_id]
            if subject == "right_arm"
            else self.sm.attached_target_l[env_id]
        )

        if verb == "reach":
            return verb, target, False

        if verb == "grasp":
            if target not in RING_TARGETS:
                return "idle", "None", True
            if last_target != target:
                return "idle", "None", True
            current_peg = self.sm.ring_support_peg.get(target, [None] * self.num_envs)[env_id]
            if current_peg is not None and current_peg != "None":
                top_rings = self.sm.get_top_rings_on_peg(env_id, current_peg, num_rings=1)
                top_ring = top_rings[0] if top_rings else None
                if top_ring != target:
                    return "idle", "None", True
            return verb, target, False

        if verb == "release":
            if target not in {t for t in TARGET_MAP.values() if t.startswith("peg_")}:
                return "idle", "None", True
            if attached_target is None:
                return "idle", "None", True
            if last_target != target:
                return "idle", "None", True
            return verb, target, False

        return verb, target, False

    # -----------------------------------------------------------------------
    # VecEnv interface
    # -----------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        self.env.reset()
        self.current_step[:] = 0

        for i in range(self.num_envs):
            self.sm.reset_env(i, phase=self.task_phase)

        sync_attached_and_frozen_rings(
            self.env,
            self.sm,
            torch.ones(self.num_envs, dtype=torch.bool, device=self.env.device),
        )

        self.prev_actions[:] = IDLE_ACTION.to(self.env.device).unsqueeze(0)
        self.last_override_flags.zero_()

        if self.tcc_model is not None:
            init_emb = self._get_batched_embeddings()
            self.emb_buffer[:] = init_emb.unsqueeze(1).repeat(1, self.stack_size, 1)
        else:
            self.emb_buffer.zero_()

        return self._get_obs()

    def step_async(self, actions) -> None:
        self.actions = actions

    def step_wait(self):
        actions = self.actions

        sanitized_actions = torch.as_tensor(
            actions, dtype=torch.long, device=self.env.device
        ).clone()

        override_flags = torch.zeros(
            (self.num_envs, 2), dtype=torch.float32, device=self.env.device
        )
        invalid_command_counts = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.env.device
        )

        for i in range(self.num_envs):
            was_busy_l = self.sm.sub_state_l[i] != "IDLE"
            was_busy_r = self.sm.sub_state_r[i] != "IDLE"

            old_cmd_l = self.sm.current_triplet_l[i]
            old_cmd_r = self.sm.current_triplet_r[i]

            verb_l = VERB_MAP[int(actions[i, 0])]
            tgt_l  = TARGET_MAP[int(actions[i, 1])]
            verb_r = VERB_MAP[int(actions[i, 2])]
            tgt_r  = TARGET_MAP[int(actions[i, 3])]

            verb_l, tgt_l, invalid_l = self._sanitize_arm_command(i, "left_arm", verb_l, tgt_l)
            verb_r, tgt_r, invalid_r = self._sanitize_arm_command(i, "right_arm", verb_r, tgt_r)

            sanitized_actions[i, 0] = VERB_TO_ID[verb_l]
            sanitized_actions[i, 1] = TARGET_TO_ID[tgt_l]
            sanitized_actions[i, 2] = VERB_TO_ID[verb_r]
            sanitized_actions[i, 3] = TARGET_TO_ID[tgt_r]

            new_cmd_l = {"verb": verb_l, "subject": "left_arm",  "target": tgt_l}
            new_cmd_r = {"verb": verb_r, "subject": "right_arm", "target": tgt_r}

            override_flags[i, 0] = 1.0 if was_busy_l and old_cmd_l != new_cmd_l else 0.0
            override_flags[i, 1] = 1.0 if was_busy_r and old_cmd_r != new_cmd_r else 0.0

            invalid_command_counts[i] = (
                float(invalid_l and not was_busy_l)
                + float(invalid_r and not was_busy_r)
            )

            # Keep prev_actions aligned with the command actually executing
            if was_busy_l and old_cmd_l is not None:
                sanitized_actions[i, 0] = VERB_TO_ID.get(old_cmd_l["verb"], VERB_TO_ID["idle"])
                sanitized_actions[i, 1] = TARGET_TO_ID.get(old_cmd_l["target"], TARGET_TO_ID["None"])
            if was_busy_r and old_cmd_r is not None:
                sanitized_actions[i, 2] = VERB_TO_ID.get(old_cmd_r["verb"], VERB_TO_ID["idle"])
                sanitized_actions[i, 3] = TARGET_TO_ID.get(old_cmd_r["target"], TARGET_TO_ID["None"])

            if not was_busy_l:
                self.sm.set_new_triplet(verb_l, "left_arm",  tgt_l, i)
            if not was_busy_r:
                self.sm.set_new_triplet(verb_r, "right_arm", tgt_r, i)

        self.prev_actions = sanitized_actions
        self.last_override_flags = override_flags

        # --- Physics step ---
        robot_r = self.env.scene["robot_right"]
        robot_l = self.env.scene["robot_left"]
        _snap_offset_r = torch.tensor([-0.005, 0.0, 0.0], device=self.env.device)
        _snap_offset_l = torch.tensor([+0.005, 0.0, 0.0], device=self.env.device)
        snap_quat      = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.env.device)

        continuous_actions = self.sm.get_action()
        _, _, terminated_il, truncated_il, _ = self.env.step(continuous_actions)
        terminated_il = terminated_il.to(device=self.env.device, dtype=torch.bool)
        truncated_il  = truncated_il.to(device=self.env.device, dtype=torch.bool)
        isaac_done    = terminated_il | truncated_il
        active_env_mask = ~isaac_done

        tip_r = robot_r.data.body_pos_w[:, self._tip_idx_r]
        tip_l = robot_l.data.body_pos_w[:, self._tip_idx_l]

        # --- Snap attached / frozen rings ---
        for ring_name in self._ring_names:
            ring  = self.env.scene[ring_name]
            new_s = ring.data.root_state_w.clone()

            mask_r = torch.tensor(
                [self.sm.attached_target_r[i] == ring_name for i in range(self.num_envs)],
                dtype=torch.bool, device=self.env.device,
            ) & active_env_mask
            mask_l = torch.tensor(
                [self.sm.attached_target_l[i] == ring_name for i in range(self.num_envs)],
                dtype=torch.bool, device=self.env.device,
            ) & active_env_mask
            mask_frozen = self.sm.frozen_rings_mask[ring_name] & active_env_mask

            write_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.env.device)

            if mask_r.any():
                new_s[mask_r, 0:3] = tip_r[mask_r] + _snap_offset_r
                new_s[mask_r, 3:7] = snap_quat
                new_s[mask_r, 7:13] = 0.0
                write_mask |= mask_r
            if mask_l.any():
                new_s[mask_l, 0:3] = tip_l[mask_l] + _snap_offset_l
                new_s[mask_l, 3:7] = snap_quat
                new_s[mask_l, 7:13] = 0.0
                write_mask |= mask_l
            if mask_frozen.any():
                new_s[mask_frozen, 0:7] = self.sm.frozen_rings_pose[ring_name][mask_frozen]
                new_s[mask_frozen, 7:13] = 0.0
                write_mask |= mask_frozen

            if write_mask.any():
                env_ids_to_write = write_mask.nonzero(as_tuple=False).flatten()
                ring.write_root_state_to_sim(new_s[env_ids_to_write], env_ids=env_ids_to_write)

        # --- TCC embeddings ---
        new_embs = self._get_batched_embeddings(dump_debug=True)
        self.emb_buffer = torch.roll(self.emb_buffer, shifts=-1, dims=1)
        self.emb_buffer[:, -1, :] = new_embs

        # --- Success frames accumulation ---
        rgb_data = self.env.scene.sensors["camera"].data.output["rgb"]
        self._accumulate_success_frames(rgb_data)

        # --- Debug logging (env 0, every 10 steps) ---
        step0 = self.current_step[0].item()
        if step0 % 10 == 0:
            cmd_r = self.sm.current_triplet_r[0]
            cmd_l = self.sm.current_triplet_l[0]
            exec_r = continuous_actions[0, 0:3].detach().cpu().tolist()
            grip_r = continuous_actions[0, 7:9].detach().cpu().tolist()
            exec_l = continuous_actions[0, 9:12].detach().cpu().tolist()
            grip_l = continuous_actions[0, 16:18].detach().cpu().tolist()
            tip_r0 = tip_r[0].detach().cpu().tolist()
            tip_l0 = tip_l[0].detach().cpu().tolist()
            print(
                f"[ENV0 Step {step0}] "
                f"R: {cmd_r['verb'] if cmd_r else 'idle'}->{cmd_r['target'] if cmd_r else 'None'} "
                f"({self.sm.sub_state_r[0]}) | "
                f"L: {cmd_l['verb'] if cmd_l else 'idle'}->{cmd_l['target'] if cmd_l else 'None'} "
                f"({self.sm.sub_state_l[0]}) | "
                f"exec_r={exec_r} grip_r={grip_r} tip_r={tip_r0} "
                f"exec_l={exec_l} grip_l={grip_l} tip_l={tip_l0}"
            )

        self.current_step += 1

        # --- Reward ---
        rewards = compute_tcc_reward(new_embs, self.goal_embedding, invalid_command_counts)
        rew_distance_raw = tcc_distance(new_embs, self.goal_embedding)

        # --- Success / settle ---
        green_peg_ring_count = peg_ring_count(
            self.sm, PEG_GREEN, self._ring_names, self.num_envs, self.env.device
        )
        rings_complete = task_success(
            self.sm, self.task_phase, self._ring_names, self.num_envs, self.env.device
        )
        just_completed = rings_complete & (self.settle_steps_remaining == 0)
        self.settle_steps_remaining[just_completed] = self.SETTLE_STEPS
        settling = self.settle_steps_remaining > 0
        self.settle_steps_remaining[settling] -= 1
        settled_success = rings_complete & (self.settle_steps_remaining == 0)
        task_success_flag = settled_success | (rings_complete & isaac_done)

        if task_success_flag.any():
            self._maybe_dump_success_assets(task_success_flag, rgb_data)

        # --- Done flags ---
        dones = isaac_done | task_success_flag
        dones_numpy = dones.cpu().numpy()

        terminal_obs = self._get_obs()

        infos: list[dict] = [{} for _ in range(self.num_envs)]

        for idx, invalid_count in enumerate(invalid_command_counts.cpu().tolist()):
            if invalid_count:
                infos[idx]["invalid_command_count"] = int(invalid_count)

        terminated_np = terminated_il.cpu().numpy()
        truncated_np  = truncated_il.cpu().numpy()

        for idx in dones.nonzero(as_tuple=False).flatten().tolist():
            infos[idx]["terminal_observation"] = terminal_obs[idx].copy()
            infos[idx]["TimeLimit.truncated"] = bool(truncated_np[idx] and not terminated_np[idx])
            infos[idx]["is_success"]          = bool(task_success_flag[idx].item())
            infos[idx]["rings_complete"]       = bool(rings_complete[idx].item())
            infos[idx]["terminal_distance"]    = float(rew_distance_raw[idx].item())
            infos[idx]["green_peg_ring_count"] = int(green_peg_ring_count[idx].item())
            if self.task_phase == "phase_1":
                infos[idx]["red_peg_ring_count"]  = int(
                    peg_ring_count(self.sm, PEG_RED,  self._ring_names, self.num_envs, self.env.device)[idx].item()
                )
                infos[idx]["blue_peg_ring_count"] = int(
                    peg_ring_count(self.sm, PEG_BLUE, self._ring_names, self.num_envs, self.env.device)[idx].item()
                )

        # --- Manual reset for non-Isaac terminal conditions ---
        manual_reset = dones & (~isaac_done)
        if manual_reset.any():
            reset_ids = manual_reset.nonzero(as_tuple=False).flatten()
            self.env.reset(env_ids=reset_ids)

        # --- Wrapper / state-machine reset ---
        if dones.any():
            done_ids = dones.nonzero(as_tuple=False).flatten().tolist()
            for idx in done_ids:
                print(
                    f"[ENV {idx}] Episode done. "
                    f"Distance={rew_distance_raw[idx].item():.4f}. Resetting."
                )
                self.sm.reset_env(idx, phase=self.task_phase)
                self.current_step[idx] = 0
                self.settle_steps_remaining[idx] = 0
                self.prev_actions[idx] = IDLE_ACTION.to(self.env.device)
                self.last_override_flags[idx] = 0.0
                self._success_frame_buffers[idx].clear()
                self._ep_step_for_video[idx] = 0

            sync_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.env.device)
            for idx in done_ids:
                sync_mask[idx] = True
            sync_attached_and_frozen_rings(self.env, self.sm, sync_mask)

            reset_embs = self._get_batched_embeddings()
            for idx in done_ids:
                self.emb_buffer[idx] = reset_embs[idx].unsqueeze(0).repeat(self.stack_size, 1)

        obs_after_reset = self._get_obs()
        return obs_after_reset, rewards.cpu().numpy(), dones_numpy, infos

    # -----------------------------------------------------------------------
    # SB3 VecEnv required stubs
    # -----------------------------------------------------------------------

    def get_attr(self, attr_name, indices=None):
        n = self.num_envs if indices is None else (1 if isinstance(indices, int) else len(indices))
        return [getattr(self, attr_name, None)] * n

    def set_attr(self, attr_name, value, indices=None): pass

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        n = self.num_envs if indices is None else (1 if isinstance(indices, int) else len(indices))
        return [None] * n

    def env_is_wrapped(self, wrapper_class, indices=None):
        n = self.num_envs if indices is None else (1 if isinstance(indices, int) else len(indices))
        return [False] * n

    def close(self): pass

    # -----------------------------------------------------------------------
    # Reward debug helpers
    # -----------------------------------------------------------------------

    def _write_debug_png(self, img_tensor: torch.Tensor, out_path: str) -> None:
        img_u8 = torch.clamp(img_tensor.detach().cpu(), 0.0, 1.0)
        img_u8 = (img_u8 * 255.0).round().to(torch.uint8)
        io.write_png(img_u8, out_path)

    def _maybe_dump_reward_images(
        self,
        raw_img_batch: torch.Tensor,
        model_img_batch: torch.Tensor,
    ) -> None:
        if self.reward_debug_dump_interval <= 0:
            return
        if not self.reward_debug_dump_env_ids:
            return
        if self.reward_debug_dump_max_per_env <= 0:
            return

        for env_id in self.reward_debug_dump_env_ids:
            if self.reward_debug_dump_counts[env_id] >= self.reward_debug_dump_max_per_env:
                continue
            step = int(self.current_step[env_id].item())
            if step <= 0 or step % self.reward_debug_dump_interval != 0:
                continue
            if self.reward_debug_last_dump_step[env_id] == step:
                continue

            dump_idx = self.reward_debug_dump_counts[env_id]
            base_path = os.path.join(
                self.reward_debug_dump_dir,
                f"env_{env_id:03d}_dump_{dump_idx:03d}_step_{step:07d}",
            )
            self._write_debug_png(raw_img_batch[env_id], f"{base_path}_reward_raw.png")
            self._write_debug_png(model_img_batch[env_id], f"{base_path}_reward_input.png")
            self.reward_debug_dump_counts[env_id] += 1
            self.reward_debug_last_dump_step[env_id] = step

    # -----------------------------------------------------------------------
    # Success snapshot / video helpers
    # -----------------------------------------------------------------------

    def _accumulate_success_frames(self, rgb_data: torch.Tensor) -> None:
        for idx in range(self.num_envs):
            self._ep_step_for_video[idx] += 1
            if self._ep_step_for_video[idx] % self.success_video_frame_skip != 0:
                continue
            # .clone() is ESSENTIAL: the rgb buffer is updated in-place by IsaacLab
            frame = rgb_data[idx, :, :, :3].clone().cpu().to(torch.uint8)
            self._success_frame_buffers[idx].append(frame)

    def _maybe_dump_success_assets(
        self,
        task_success_tensor: torch.Tensor,
        rgb_data: torch.Tensor,
    ) -> None:
        for idx in task_success_tensor.nonzero(as_tuple=False).flatten().tolist():
            ts  = int(self.current_step[idx].item())
            tag = f"env{idx:03d}_step{ts:06d}"

            if random.random() < self.success_snapshot_prob:
                try:
                    frame = (
                        rgb_data[idx, :, :, :3]
                        .clone().cpu().to(torch.uint8)
                        .permute(2, 0, 1)  # (3, H, W)
                    )
                    png_path = os.path.join(self.success_dump_dir, f"success_{tag}.png")
                    io.write_png(frame, png_path)
                    print(f"[SuccessDump] Screenshot saved: {png_path}", flush=True)
                except Exception as exc:
                    print(f"[SuccessDump] Screenshot error env {idx}: {exc}", flush=True)

            if random.random() < self.success_video_prob:
                frames = self._success_frame_buffers[idx]
                if len(frames) >= 2:
                    try:
                        video_tensor = torch.stack(frames, dim=0)  # (T, H, W, 3)
                        mp4_path = os.path.join(self.success_video_dir, f"success_{tag}.mp4")
                        io.write_video(mp4_path, video_tensor, fps=10)
                        print(
                            f"[SuccessDump] Video saved: {mp4_path} ({len(frames)} frames)",
                            flush=True,
                        )
                    except Exception as exc:
                        print(f"[SuccessDump] Video error env {idx}: {exc}", flush=True)
