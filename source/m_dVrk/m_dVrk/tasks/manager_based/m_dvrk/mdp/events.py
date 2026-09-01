from __future__ import annotations

import math

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedEnv


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
    fixed_ring_poses: dict[str, tuple[float, float, float, float]] | None = None,
):
    """Reset all rings jointly on the board while avoiding pegs and mutual overlaps.

    Ring x/y positions are expressed in the local frame of each environment.
    Phase 0 requires randomized placements on the board; fixed poses are only
    intended for debugging or specialized non-random initial conditions.
    """

    device = env.device
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

            if not randomize and fixed_pose is not None:
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
                best_candidate = None
                best_margin = -float("inf")
                found_valid_candidate = False

                for _ in range(max_sample_attempts):
                    candidate = _sample_ring_candidate(device, x_range, y_range, z_height)
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
                        x_range,
                        y_range,
                        z_height,
                        peg_positions_xy,
                        placed_positions_xy,
                        min_peg_clearance,
                        min_ring_clearance,
                    )

                if not found_valid_candidate and best_margin < 0.0:
                    best_candidate = _select_fallback_candidate(
                        device,
                        x_range,
                        y_range,
                        z_height,
                        peg_positions_xy,
                        placed_positions_xy,
                        min_peg_clearance,
                        min_ring_clearance,
                    )

                yaw = torch.empty(1, device=device).uniform_(yaw_range[0], yaw_range[1]).item()
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
