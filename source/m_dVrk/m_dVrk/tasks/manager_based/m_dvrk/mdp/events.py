from __future__ import annotations

import math

import numpy as np
import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedEnv

from m_dVrk.hrl.curriculum import (
    phase_entry_pose_sampling_bounds,
    scaled_interval_around_anchor,
)


def reset_joints_from_empirical_bank(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    bank_path: str,
    bank_side: str,
    fallback_joint_positions: dict[str, float],
    empirical_probability: float = 1.0,
    empirical_sample_mode: str = "random",
    sample_owner: bool = True,
    use_recorded_velocity: bool = False,
    expected_task_phase: str = "phase_4",
    asset_cfg=None,
):
    """Reset an arm from paired empirical robot/ring configurations.

    The right-arm reset samples one bank row and the exact/empirical choice;
    the left-arm reset reuses both so the two arms always come from the same
    physical handoff.  Episodes not assigned to the empirical component use
    the calibrated deterministic parked pose.
    """

    if asset_cfg is None:
        raise ValueError("asset_cfg is required")
    if bank_side not in {"right", "left"}:
        raise ValueError("bank_side must be 'right' or 'left'")
    if not 0.0 <= empirical_probability <= 1.0:
        raise ValueError("empirical_probability must be in [0, 1]")
    if empirical_sample_mode not in {"random", "sequential"}:
        raise ValueError("empirical_sample_mode must be 'random' or 'sequential'")

    cache = getattr(env, "_empirical_entry_bank", None)
    if cache is None or cache["path"] != bank_path:
        with np.load(bank_path, allow_pickle=False) as data:
            full_state_keys = {
                "ring_root_state_local",
                "ring_frozen_mask",
                "ring_support_peg",
                "ring_stack_level",
                "ring_names",
            }
            attachment_keys = {"attached_target_right", "attached_target_left"}
            present_full_state_keys = full_state_keys.intersection(data.files)
            if present_full_state_keys and present_full_state_keys != full_state_keys:
                missing = sorted(full_state_keys - present_full_state_keys)
                raise ValueError(
                    "Incomplete empirical Phase-4 full-state bank; missing: "
                    + ", ".join(missing)
                )
            present_attachment_keys = attachment_keys.intersection(data.files)
            if present_attachment_keys and present_attachment_keys != attachment_keys:
                missing = sorted(attachment_keys - present_attachment_keys)
                raise ValueError(
                    "Incomplete empirical attachment state; missing: "
                    + ", ".join(missing)
                )
            bank_task_phase = (
                str(np.asarray(data["task_phase"]).item())
                if "task_phase" in data.files
                else "phase_4"
            )
            if bank_task_phase != expected_task_phase:
                raise ValueError(
                    f"Empirical bank task_phase={bank_task_phase!r} does not "
                    f"match requested {expected_task_phase!r}"
                )
            cache = {
                "path": bank_path,
                "task_phase": bank_task_phase,
                "joint_pos_right": torch.as_tensor(
                    data["joint_pos_right"], dtype=torch.float32, device=env.device
                ),
                "joint_vel_right": torch.as_tensor(
                    data["joint_vel_right"], dtype=torch.float32, device=env.device
                ),
                "joint_pos_left": torch.as_tensor(
                    data["joint_pos_left"], dtype=torch.float32, device=env.device
                ),
                "joint_vel_left": torch.as_tensor(
                    data["joint_vel_left"], dtype=torch.float32, device=env.device
                ),
                "joint_names_right": tuple(str(x) for x in data["joint_names_right"]),
                "joint_names_left": tuple(str(x) for x in data["joint_names_left"]),
                "has_full_state": present_full_state_keys == full_state_keys,
                "has_attachment_state": present_attachment_keys == attachment_keys,
            }
            if cache["has_full_state"]:
                cache.update(
                    {
                        "ring_root_state_local": torch.as_tensor(
                            data["ring_root_state_local"],
                            dtype=torch.float32,
                            device=env.device,
                        ),
                        "ring_frozen_mask": torch.as_tensor(
                            data["ring_frozen_mask"],
                            dtype=torch.bool,
                            device=env.device,
                        ),
                        "ring_support_peg": np.asarray(
                            data["ring_support_peg"], dtype=str
                        ).copy(),
                        "ring_stack_level": torch.as_tensor(
                            data["ring_stack_level"],
                            dtype=torch.long,
                            device=env.device,
                        ),
                        "ring_names": tuple(str(x) for x in data["ring_names"]),
                    }
                )
            if cache["has_attachment_state"]:
                cache.update(
                    {
                        "attached_target_right": np.asarray(
                            data["attached_target_right"], dtype=str
                        ).copy(),
                        "attached_target_left": np.asarray(
                            data["attached_target_left"], dtype=str
                        ).copy(),
                    }
                )
        if cache["joint_pos_right"].shape[0] == 0:
            raise ValueError(f"Empirical entry bank is empty: {bank_path}")
        if cache["joint_pos_right"].shape[0] != cache["joint_pos_left"].shape[0]:
            raise ValueError("Right/left empirical entry banks have different sizes")
        if cache["has_attachment_state"]:
            bank_size = cache["joint_pos_right"].shape[0]
            for key in ("attached_target_right", "attached_target_left"):
                if tuple(cache[key].shape) != (bank_size,):
                    raise ValueError(
                        f"{key} must have shape ({bank_size},), got "
                        f"{tuple(cache[key].shape)}"
                    )
        if cache["has_full_state"]:
            bank_size = cache["joint_pos_right"].shape[0]
            ring_state = cache["ring_root_state_local"]
            if ring_state.ndim != 3 or ring_state.shape[0] != bank_size:
                raise ValueError(
                    "ring_root_state_local must have shape (states, rings, 13)"
                )
            if ring_state.shape[2] != 13:
                raise ValueError(
                    "ring_root_state_local must contain pose and velocity (13 values)"
                )
            expected_shape = ring_state.shape[:2]
            for key in (
                "ring_frozen_mask",
                "ring_support_peg",
                "ring_stack_level",
            ):
                if tuple(cache[key].shape) != tuple(expected_shape):
                    raise ValueError(
                        f"{key} shape {tuple(cache[key].shape)} does not match "
                        f"{tuple(expected_shape)}"
                    )
            if len(cache["ring_names"]) != expected_shape[1]:
                raise ValueError("ring_names length does not match ring state columns")
        setattr(env, "_empirical_entry_bank", cache)
        # Compatibility with old Phase-4 diagnostics.
        setattr(env, "_phase4_empirical_entry_bank", cache)

    sample_indices = getattr(env, "_empirical_entry_indices", None)
    sample_enabled = getattr(env, "_empirical_entry_enabled", None)
    if sample_indices is None or sample_indices.shape[0] != env.num_envs:
        sample_indices = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )
        sample_enabled = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
        setattr(env, "_empirical_entry_indices", sample_indices)
        setattr(env, "_empirical_entry_enabled", sample_enabled)
        setattr(env, "_phase4_empirical_entry_indices", sample_indices)
        setattr(env, "_phase4_empirical_entry_enabled", sample_enabled)

    bank_size = int(cache["joint_pos_right"].shape[0])
    if sample_owner:
        if empirical_sample_mode == "sequential":
            counter = int(getattr(env, "_empirical_entry_counter", 0))
            sample_indices[env_ids] = (
                torch.arange(len(env_ids), device=env.device, dtype=torch.long)
                + counter
            ) % bank_size
            setattr(
                env,
                "_empirical_entry_counter",
                counter + len(env_ids),
            )
        else:
            sample_indices[env_ids] = torch.randint(
                bank_size, (len(env_ids),), device=env.device
            )
        sample_enabled[env_ids] = (
            torch.rand(len(env_ids), device=env.device) < empirical_probability
        )

    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()
    joint_name_to_id = {name: idx for idx, name in enumerate(asset.joint_names)}

    unknown_fallback = sorted(set(fallback_joint_positions) - set(joint_name_to_id))
    if unknown_fallback:
        raise ValueError(
            f"Unknown fallback joints for {asset_cfg.name!r}: "
            + ", ".join(unknown_fallback)
        )
    for joint_name, value in fallback_joint_positions.items():
        joint_pos[:, joint_name_to_id[joint_name]] = float(value)
    joint_vel.zero_()

    names = cache[f"joint_names_{bank_side}"]
    unknown_bank = sorted(set(names) - set(joint_name_to_id))
    if unknown_bank:
        raise ValueError(
            f"Unknown bank joints for {asset_cfg.name!r}: " + ", ".join(unknown_bank)
        )
    enabled = sample_enabled[env_ids]
    if bool(enabled.any().item()):
        bank_rows = sample_indices[env_ids][enabled]
        bank_pos = cache[f"joint_pos_{bank_side}"][bank_rows]
        bank_vel = cache[f"joint_vel_{bank_side}"][bank_rows]
        enabled_rows = enabled.nonzero(as_tuple=False).flatten()
        for bank_col, joint_name in enumerate(names):
            asset_col = joint_name_to_id[joint_name]
            joint_pos[enabled_rows, asset_col] = bank_pos[:, bank_col]
            if use_recorded_velocity:
                joint_vel[enabled_rows, asset_col] = bank_vel[:, bank_col]

    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def reset_joints_to_interpolated_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    target_joint_positions: dict[str, float],
    scale: float = 1.0,
    sample_uniform_scale: bool = False,
    sample_frontier_width: float | None = None,
    sample_owner: bool = True,
    neighborhood_joint_scales: dict[str, float] | None = None,
    neighborhood_probability: float = 0.0,
    neighborhood_scale: float = 1.0,
    neighborhood_owner: bool = True,
    scale_attribute: str = "phase_entry_pose_scale",
    asset_cfg=None,
):
    """Reset joints between their canonical default and a phase-entry pose.

    Unlike changing ``ArticulationCfg.init_state``, this keeps
    ``asset.data.default_joint_pos`` canonical.  Consequently Isaac Lab's
    ``joint_pos_rel`` observation uses the same reference frame during
    standalone training and full-task inference.

    When ``sample_uniform_scale`` is enabled, the owner arm samples one
    interpolation factor per environment and the second arm reuses it.  With
    no frontier width the interval is ``[0, current_scale]``; with a width it
    is ``[current_scale - width, current_scale]`` (clamped at zero).
    """

    if asset_cfg is None:
        raise ValueError("asset_cfg is required")

    current_scale = float(getattr(env, scale_attribute, scale))
    if not 0.0 <= current_scale <= 1.0:
        raise ValueError(f"{scale_attribute} must be in [0, 1]")

    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()
    target_pos = joint_pos.clone()

    joint_name_to_id = {name: idx for idx, name in enumerate(asset.joint_names)}
    unknown = sorted(set(target_joint_positions) - set(joint_name_to_id))
    if unknown:
        raise ValueError(
            f"Unknown joints for asset {asset_cfg.name!r}: {', '.join(unknown)}"
        )
    for joint_name, value in target_joint_positions.items():
        target_pos[:, joint_name_to_id[joint_name]] = float(value)

    cache_name = "_phase_entry_pose_sampled_scale"
    sampled_scale = getattr(env, cache_name, None)
    if sampled_scale is None or sampled_scale.shape[0] != env.num_envs:
        sampled_scale = torch.zeros(env.num_envs, device=joint_pos.device)
        setattr(env, cache_name, sampled_scale)

    if sample_owner:
        if sample_uniform_scale and current_scale > 0.0:
            frontier_width = getattr(
                env,
                "phase_entry_pose_frontier_width",
                sample_frontier_width,
            )
            lower, upper = phase_entry_pose_sampling_bounds(
                current_scale,
                frontier_width,
            )
            alpha = lower + torch.rand(
                len(env_ids), device=joint_pos.device
            ) * (upper - lower)
        else:
            alpha = torch.full(
                (len(env_ids),), current_scale, device=joint_pos.device
            )
        sampled_scale[env_ids] = alpha
    else:
        alpha = sampled_scale[env_ids]

    joint_pos.lerp_(target_pos, alpha.unsqueeze(1))

    if neighborhood_joint_scales and neighborhood_probability > 0.0:
        if not 0.0 <= neighborhood_probability <= 1.0:
            raise ValueError("neighborhood_probability must be in [0, 1]")
        if not 0.0 <= neighborhood_scale <= 1.0:
            raise ValueError("neighborhood_scale must be in [0, 1]")

        noise_joint_ids: list[int] = []
        noise_scales: list[float] = []
        unknown_noise_joints = sorted(
            set(neighborhood_joint_scales) - set(joint_name_to_id)
        )
        if unknown_noise_joints:
            raise ValueError(
                f"Unknown neighborhood joints for asset {asset_cfg.name!r}: "
                + ", ".join(unknown_noise_joints)
            )
        for joint_name, joint_scale in neighborhood_joint_scales.items():
            if float(joint_scale) < 0.0:
                raise ValueError("neighborhood joint scales must be non-negative")
            noise_joint_ids.append(joint_name_to_id[joint_name])
            noise_scales.append(float(joint_scale) * float(neighborhood_scale))

        enabled_cache_name = "_phase_entry_park_neighborhood_enabled"
        radius_cache_name = "_phase_entry_park_neighborhood_radius"
        noise_enabled = getattr(env, enabled_cache_name, None)
        radius = getattr(env, radius_cache_name, None)
        if noise_enabled is None or noise_enabled.shape[0] != env.num_envs:
            noise_enabled = torch.zeros(
                env.num_envs, dtype=torch.bool, device=joint_pos.device
            )
            setattr(env, enabled_cache_name, noise_enabled)
        if radius is None or radius.shape[0] != env.num_envs:
            radius = torch.zeros(env.num_envs, device=joint_pos.device)
            setattr(env, radius_cache_name, radius)

        dimensions = len(noise_joint_ids)
        if neighborhood_owner:
            episode_enabled = (
                torch.rand(len(env_ids), device=joint_pos.device)
                < neighborhood_probability
            )
            # r = U^(1/d) makes samples uniform by volume inside the unit ball.
            episode_radius = torch.rand(
                len(env_ids), device=joint_pos.device
            ).pow(1.0 / max(dimensions, 1))
            episode_radius *= episode_enabled.to(dtype=episode_radius.dtype)
            noise_enabled[env_ids] = episode_enabled
            radius[env_ids] = episode_radius
        else:
            episode_enabled = noise_enabled[env_ids]
            episode_radius = radius[env_ids]

        if dimensions:
            direction = torch.randn(
                (len(env_ids), dimensions), device=joint_pos.device
            )
            direction /= torch.linalg.vector_norm(
                direction, dim=1, keepdim=True
            ).clamp_min(1e-8)
            semi_axes = torch.tensor(
                noise_scales,
                dtype=joint_pos.dtype,
                device=joint_pos.device,
            )
            offsets = direction * episode_radius.unsqueeze(1) * semi_axes
            joint_pos[:, noise_joint_ids] += offsets

    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    joint_vel.zero_()
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def _resolve_asset_prim_path(env: ManagerBasedEnv, asset_name: str, env_id: int) -> str:
    prim_path = env.scene[asset_name].cfg.prim_path
    env_regex_ns = env.scene.env_regex_ns

    if env_regex_ns in prim_path:
        return prim_path.replace(env_regex_ns, env.scene.env_prim_paths[env_id])

    return prim_path


def disable_collisions_between_assets(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_name_a: str,
    asset_name_b: str,
):
    """Disable collisions between two scene assets inside each environment."""

    from pxr import Sdf, UsdPhysics

    if env_ids is None:
        env_id_list = range(env.num_envs)
    elif isinstance(env_ids, slice):
        env_id_list = range(env.num_envs)[env_ids]
    else:
        env_id_list = env_ids.tolist()

    stage = env.scene.stage
    filtered_count = 0

    for env_id in env_id_list:
        path_a = _resolve_asset_prim_path(env, asset_name_a, env_id)
        path_b = _resolve_asset_prim_path(env, asset_name_b, env_id)

        prim_a = stage.GetPrimAtPath(path_a)
        prim_b = stage.GetPrimAtPath(path_b)
        if not prim_a.IsValid() or not prim_b.IsValid():
            print(f"[collision-filter] Cannot filter invalid pair: '{path_a}' <-> '{path_b}'.")
            continue

        for prim, other_path in ((prim_a, path_b), (prim_b, path_a)):
            rel = UsdPhysics.FilteredPairsAPI.Apply(prim).CreateFilteredPairsRel()
            target = Sdf.Path(other_path)
            if target not in rel.GetTargets():
                rel.AddTarget(target)

        filtered_count += 1

    print(
        f"[collision-filter] Disabled collisions between '{asset_name_a}' and "
        f"'{asset_name_b}' in {filtered_count} envs."
    )


def _get_asset_positions_local(env: ManagerBasedEnv, asset_name: str, env_ids: torch.Tensor) -> torch.Tensor:
    asset = env.scene[asset_name]
    env_origins = env.scene.env_origins[env_ids]
    env_id_list = env_ids.tolist()

    if hasattr(asset, "data") and hasattr(asset.data, "root_pos_w"):
        pos_w_all = asset.data.root_pos_w
        if pos_w_all.shape[0] == env.num_envs:
            return pos_w_all[env_ids] - env_origins
        if pos_w_all.shape[0] == len(env_id_list):
            return pos_w_all - env_origins
        if pos_w_all.shape[0] == 1:
            local_pos = pos_w_all[0] - env.scene.env_origins[0]
            return local_pos.unsqueeze(0).expand(len(env_id_list), -1)
        raise RuntimeError(
            f"Asset '{asset_name}' exposes {pos_w_all.shape[0]} poses, expected 1, {len(env_id_list)}, or {env.num_envs}."
        )

    if hasattr(asset, "get_world_poses"):
        view_count = getattr(asset, "count", None)

        if view_count == env.num_envs:
            pos_w, _ = asset.get_world_poses(indices=env_id_list)
            return pos_w - env_origins

        pos_w, _ = asset.get_world_poses()
        if pos_w.shape[0] == len(env_id_list):
            return pos_w - env_origins
        if pos_w.shape[0] == 1:
            local_pos = pos_w[0] - env.scene.env_origins[0]
            return local_pos.unsqueeze(0).expand(len(env_id_list), -1)
        if pos_w.shape[0] == env.num_envs:
            return pos_w[env_ids] - env_origins
        raise RuntimeError(
            f"Asset '{asset_name}' view returned {pos_w.shape[0]} poses, expected 1, {len(env_id_list)}, or {env.num_envs}."
        )

    raise AttributeError(f"Asset '{asset_name}' does not expose world poses for reset sampling.")


def _sample_ring_candidate(
    device: torch.device,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_height: float,
) -> torch.Tensor:
    x = torch.empty(1, device=device).uniform_(x_range[0], x_range[1]).item()
    y = torch.empty(1, device=device).uniform_(y_range[0], y_range[1]).item()
    return torch.tensor([x, y, z_height], device=device)


def _candidate_margin(
    candidate_xy: torch.Tensor,
    peg_positions_xy: torch.Tensor,
    placed_positions_xy: list[torch.Tensor],
    min_peg_clearance: float,
    min_ring_clearance: float,
) -> tuple[bool, float]:
    peg_distance = torch.full((), float("inf"), device=candidate_xy.device)
    if peg_positions_xy.numel() > 0:
        peg_distance = torch.min(torch.norm(peg_positions_xy - candidate_xy.unsqueeze(0), dim=1))

    ring_distance = torch.full((), float("inf"), device=candidate_xy.device)
    if placed_positions_xy:
        placed_xy = torch.stack(placed_positions_xy)
        ring_distance = torch.min(torch.norm(placed_xy - candidate_xy.unsqueeze(0), dim=1))

    margin = torch.minimum(peg_distance - min_peg_clearance, ring_distance - min_ring_clearance).item()
    is_valid = peg_distance.item() >= min_peg_clearance and ring_distance.item() >= min_ring_clearance
    return is_valid, margin


def _select_fallback_candidate(
    device: torch.device,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_height: float,
    peg_positions_xy: torch.Tensor,
    placed_positions_xy: list[torch.Tensor],
    min_peg_clearance: float,
    min_ring_clearance: float,
    grid_size: int = 9,
) -> torch.Tensor:
    xs = torch.linspace(x_range[0], x_range[1], steps=grid_size, device=device)
    ys = torch.linspace(y_range[0], y_range[1], steps=grid_size, device=device)

    best_candidate = torch.tensor([(x_range[0] + x_range[1]) * 0.5, (y_range[0] + y_range[1]) * 0.5, z_height], device=device)
    best_margin = -float("inf")

    for x in xs:
        for y in ys:
            candidate_xy = torch.stack((x, y))
            is_valid, margin = _candidate_margin(
                candidate_xy,
                peg_positions_xy,
                placed_positions_xy,
                min_peg_clearance,
                min_ring_clearance,
            )
            candidate = torch.tensor([x.item(), y.item(), z_height], device=device)
            if is_valid:
                return candidate
            if margin > best_margin:
                best_margin = margin
                best_candidate = candidate

    return best_candidate


def reset_rings_on_board(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    ring_names: list[str],
    peg_names: list[str],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_height: float,
    min_ring_clearance: float,
    min_peg_clearance: float,
    yaw_range: tuple[float, float] = (-math.pi, math.pi),
    max_sample_attempts: int = 64,
    randomize: bool = True,
    randomization_scale: float = 1.0,
    fixed_ring_poses: dict[str, tuple[float, float, float, float]] | None = None,
):
    """Reset all rings jointly on the board while avoiding pegs and mutual overlaps.

    Ring x/y positions are expressed in the local frame of each environment.
    Phase 0 requires randomized placements on the board; fixed poses are only
    intended for debugging or specialized non-random initial conditions.
    """

    device = env.device
    scale = float(getattr(env, "ring_randomization_scale", randomization_scale))
    if not 0.0 <= scale <= 1.0:
        raise ValueError("ring randomization scale must be in [0, 1]")
    env_origins = env.scene.env_origins[env_ids]

    peg_positions_local = []
    for peg_name in peg_names:
        peg_positions_local.append(_get_asset_positions_local(env, peg_name, env_ids))
    peg_positions_local = torch.stack(peg_positions_local, dim=1) if peg_positions_local else torch.empty((len(env_ids), 0, 3), device=device)

    for local_env_idx, env_id in enumerate(env_ids.tolist()):
        peg_positions_xy = peg_positions_local[local_env_idx, :, :2] if peg_positions_local.numel() > 0 else torch.empty((0, 2), device=device)
        placed_positions_xy: list[torch.Tensor] = []

        sampled_positions = []
        sampled_orientations = []

        for ring_name in ring_names:
            fixed_pose = None if fixed_ring_poses is None else fixed_ring_poses.get(ring_name)

            if (not randomize or scale == 0.0) and fixed_pose is not None:
                best_candidate = torch.tensor(fixed_pose[:3], device=device)
                yaw = float(fixed_pose[3])
                is_valid, margin = _candidate_margin(
                    best_candidate[:2],
                    peg_positions_xy,
                    placed_positions_xy,
                    min_peg_clearance,
                    min_ring_clearance,
                )
                if not is_valid:
                    print(
                        f"[ring-reset] Warning: fixed pose for {ring_name} has "
                        f"clearance margin {margin:.4f} in env {env_id}."
                    )
            elif randomize:
                ring_x_range = x_range
                ring_y_range = y_range
                ring_yaw_range = yaw_range
                if fixed_pose is not None and scale < 1.0:
                    ring_x_range = scaled_interval_around_anchor(
                        float(fixed_pose[0]), x_range, scale
                    )
                    ring_y_range = scaled_interval_around_anchor(
                        float(fixed_pose[1]), y_range, scale
                    )
                    ring_yaw_range = scaled_interval_around_anchor(
                        float(fixed_pose[3]), yaw_range, scale
                    )

                best_candidate = None
                best_margin = -float("inf")
                found_valid_candidate = False

                for _ in range(max_sample_attempts):
                    candidate = _sample_ring_candidate(
                        device, ring_x_range, ring_y_range, z_height
                    )
                    is_valid, margin = _candidate_margin(
                        candidate[:2],
                        peg_positions_xy,
                        placed_positions_xy,
                        min_peg_clearance,
                        min_ring_clearance,
                    )

                    if is_valid:
                        best_candidate = candidate
                        found_valid_candidate = True
                        break

                    if margin > best_margin:
                        best_margin = margin
                        best_candidate = candidate

                if best_candidate is None:
                    best_candidate = _select_fallback_candidate(
                        device,
                        ring_x_range,
                        ring_y_range,
                        z_height,
                        peg_positions_xy,
                        placed_positions_xy,
                        min_peg_clearance,
                        min_ring_clearance,
                    )

                if not found_valid_candidate and best_margin < 0.0:
                    best_candidate = _select_fallback_candidate(
                        device,
                        ring_x_range,
                        ring_y_range,
                        z_height,
                        peg_positions_xy,
                        placed_positions_xy,
                        min_peg_clearance,
                        min_ring_clearance,
                    )

                yaw = torch.empty(1, device=device).uniform_(
                    ring_yaw_range[0], ring_yaw_range[1]
                ).item()
            else:
                best_candidate = _select_fallback_candidate(
                    device,
                    x_range,
                    y_range,
                    z_height,
                    peg_positions_xy,
                    placed_positions_xy,
                    min_peg_clearance,
                    min_ring_clearance,
                    grid_size=11,
                )
                yaw = 0.0

            yaw_quat = math_utils.quat_from_euler_xyz(
                torch.zeros(1, device=device),
                torch.zeros(1, device=device),
                torch.tensor([yaw], device=device),
            )[0]

            sampled_positions.append(best_candidate + env_origins[local_env_idx])
            sampled_orientations.append(yaw_quat)
            placed_positions_xy.append(best_candidate[:2])

        for ring_name, position_w, orientation_w in zip(ring_names, sampled_positions, sampled_orientations, strict=True):
            ring_asset: RigidObject | Articulation = env.scene[ring_name]
            pose = torch.cat([position_w, orientation_w], dim=0).unsqueeze(0)
            velocity = torch.zeros((1, 6), device=device)
            ring_asset.write_root_pose_to_sim(pose, env_ids=torch.tensor([env_id], device=device, dtype=torch.long))
            ring_asset.write_root_velocity_to_sim(velocity, env_ids=torch.tensor([env_id], device=device, dtype=torch.long))
