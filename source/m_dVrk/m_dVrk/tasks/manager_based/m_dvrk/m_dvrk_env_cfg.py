# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.sensors import CameraCfg


from . import mdp

MODULE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
USD_DATA_DIR = os.path.join(MODULE_DIR, "data", "usd")

##
# Pre-defined configs
##

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.controllers.operational_space_cfg import OperationalSpaceControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.envs.mdp.actions.actions_cfg import OperationalSpaceControllerActionCfg
from isaaclab_assets.robots.cartpole import CARTPOLE_CFG  # isort:skip
from m_dVrk.assets import DAVINCI_CFG

import math
import isaaclab.utils.math as math_utils
import torch

##
# Scene definition
##

@configclass
class PegRingSceneCfg(InteractiveSceneCfg):
    """1. Configurazione della Scena (Gli Attori)"""
    
    # ground and lights
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table", 
        spawn=sim_utils.UsdFileCfg(usd_path=f"{USD_DATA_DIR}/table.usd"),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.4, 0.0, 0.25)),
    )

    # Peg and Ring base
    peg_board = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/PegBoard",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{USD_DATA_DIR}/pn_base2.usd"),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.375, -0.05, 0.71), rot=(0.0, 0.0, 0.0, 1.0)), 
    )

    # ==========================================
    # PEGS XFORM
    # ==========================================
    peg_red = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/PegBoard/peg_red",
        spawn=None,
    )
    peg_yellow = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/PegBoard/peg_yellow",
        spawn=None,
    )
    peg_green = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/PegBoard/peg_green",
        spawn=None,
    )
    peg_blue = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/PegBoard/peg_blue",
        spawn=None,
    )

    peg_gray = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/PegBoard/peg_gray",
        spawn=None,
    )

    peg_gray1 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/PegBoard/peg_gray1",
        spawn=None,
    )

    peg_gray2 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/PegBoard/peg_gray2",
        spawn=None,
    )

    peg_gray3 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/PegBoard/peg_gray3",
        spawn=None,
    )

    # Red Ring
    ring_red = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/RingRed",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{USD_DATA_DIR}/red_ring.usd",
            scale=(1.0, 1.0, 0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=True,
                linear_damping=0.2,
                angular_damping=0.5,
                max_linear_velocity=0.5,
                max_angular_velocity=30.0,
                max_depenetration_velocity=0.05,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.01),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.335, -0.055, 0.717)), 
    )

    # Yellow Ring
    ring_yellow = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/RingYellow",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{USD_DATA_DIR}/yellow_ring.usd",
            scale=(1.0, 1.0, 0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=True,
                linear_damping=0.2,
                angular_damping=0.5,
                max_linear_velocity=0.5,
                max_angular_velocity=30.0,
                max_depenetration_velocity=0.05,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.01),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.362, -0.055, 0.717)),
    )

    # Green Ring
    ring_green = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/RingGreen",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{USD_DATA_DIR}/green_ring.usd",
            scale=(1.0, 1.0, 0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=True,
                linear_damping=0.2,
                angular_damping=0.5,
                max_linear_velocity=0.5,
                max_angular_velocity=30.0,
                max_depenetration_velocity=0.05,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.01),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.388, -0.055, 0.717)),
    )

    # Blue Ring
    ring_blue = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/RingBlue",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{USD_DATA_DIR}/blue_ring.usd",
            scale=(1.0, 1.0, 0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=True,
                linear_damping=0.2,
                angular_damping=0.5,
                max_linear_velocity=0.5,
                max_angular_velocity=30.0,
                max_depenetration_velocity=0.05,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.01),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.415, -0.055, 0.717)),
    )

    # daVinci Robot

    # ==========================================
    # ROBOT (Right and Left Arm)
    # ==========================================
    # Right arm
    robot_right: ArticulationCfg = DAVINCI_CFG.replace(prim_path="{ENV_REGEX_NS}/RobotRight")
    robot_right.init_state.pos = (0.25, -0.15, 0.8) 
    robot_right.init_state.joint_pos = {
        "psm_yaw_joint": 0.5,
        "psm_pitch_end_joint": -0.5,      
        "psm_main_insertion_joint": 0.08, 
        "psm_tool_roll_joint": 0.0,
        "psm_tool_pitch_joint": -0.785,
        "psm_tool_yaw_joint": 0.0,
    }

    # Left arm
    robot_left: ArticulationCfg = DAVINCI_CFG.replace(prim_path="{ENV_REGEX_NS}/RobotLeft")
    robot_left.init_state.pos = (0.5, -0.15, 0.8)
    robot_left.init_state.joint_pos = {
        "psm_yaw_joint": -0.5,
        "psm_pitch_end_joint": -0.5,
        "psm_main_insertion_joint": 0.08,
        "psm_tool_roll_joint": 0.0,
        "psm_tool_pitch_joint": -0.785,
        "psm_tool_yaw_joint": 0.0,
    }

    # ==========================================
    # CAMERA
    # ==========================================
    camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        # Render every ~3.3s of sim time (= 400 step @ 120Hz = 1 RL-step).
        update_period=1/30.0,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=23.5, 
            focus_distance=400.0, 
            horizontal_aperture=20.955, 
            clipping_range=(0.01, 1.0e5) 
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.375, 0.125, 0.85), 
            
            # ROTATION: Pitch 40°, Yaw -90°, Roll 0° ([w, x, y, z])
            rot=(0.664, 0.242, 0.242, -0.664), 
            
            convention="world"
        ),
    )

##
# MDP settings
##


@configclass
class ActionsCfg:
    
    arm_right = DifferentialInverseKinematicsActionCfg(
        asset_name="robot_right",

        joint_names=[
            "psm_yaw_joint",
            "psm_pitch_end_joint",
            "psm_main_insertion_joint",
            "psm_tool_roll_joint",
            "psm_tool_pitch_joint",
            "psm_tool_yaw_joint",
            ],

        body_name="psm_tool_tip_link",

        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="pinv",
        ),
        scale=1.0,
    )
    
    
    gripper_right = mdp.JointPositionActionCfg(
        asset_name="robot_right",
        joint_names=["psm_tool_gripper1_joint", "psm_tool_gripper2_joint"],
        scale=1.0,
        use_default_offset=False,
    )

    arm_left = DifferentialInverseKinematicsActionCfg(
        asset_name="robot_left",

        joint_names=[
            "psm_yaw_joint",
            "psm_pitch_end_joint",
            "psm_main_insertion_joint",
            "psm_tool_roll_joint",
            "psm_tool_pitch_joint",
            "psm_tool_yaw_joint",
            ],

        body_name="psm_tool_tip_link",

        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="pinv",
        ),
        scale=1.0,
    )
    
    
    gripper_left = mdp.JointPositionActionCfg(
        asset_name="robot_left",
        joint_names=["psm_tool_gripper1_joint", "psm_tool_gripper2_joint"],
        scale=1.0,
        use_default_offset=False,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos_right = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot_right")})
        joint_pos_left = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot_left")})

        joint_vel_right = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot_right")})
        joint_vel_left = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot_left")})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    disable_arm_arm_collisions = EventTerm(
        func=mdp.disable_collisions_between_assets,
        mode="startup",
        params={
            "asset_name_a": "robot_right",
            "asset_name_b": "robot_left",
        },
    )

    # reset
    reset_robot_right = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot_right"),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )
    
    reset_robot_left = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot_left"),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_rings = EventTerm(
        func=mdp.reset_rings_on_board,
        mode="reset",
        params={
            "ring_names": ["ring_red", "ring_yellow", "ring_green", "ring_blue"],
            "peg_names": [
                "peg_red",
                "peg_yellow",
                "peg_green",
                "peg_blue",
                "peg_gray",
                "peg_gray1",
                "peg_gray2",
                "peg_gray3",
            ],
            "x_range": (0.318, 0.432),
            "y_range": (-0.107, 0.007),
            "z_height": 0.717,
            "min_ring_clearance": 0.020,
            "min_peg_clearance": 0.016,
            "yaw_range": (-3.14, 3.14),
            "max_sample_attempts": 256,
            "randomize": True,
            "fixed_ring_poses": {
                "ring_red": (0.335, -0.055, 0.717, 0.0),
                "ring_yellow": (0.362, -0.055, 0.717, 0.0),
                "ring_green": (0.388, -0.055, 0.717, 0.0),
                "ring_blue": (0.415, -0.055, 0.717, 0.0),
            },
        },
    )
    # reset_pole_position = EventTerm(
    #     func=mdp.reset_joints_by_offset,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
    #         "position_range": (-0.25 * math.pi, 0.25 * math.pi),
    #         "velocity_range": (-0.25 * math.pi, 0.25 * math.pi),
    #     },
    # )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    dummy_reward = RewTerm(func=mdp.is_alive, weight=1.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # (2) Ring out of workspace (local frame for each env)
    # The PegBoard is at (0.375, -0.05, 0.71): ±10cm around the center
    ring_out_of_bounds = DoneTerm(
        func=mdp.ring_out_of_bounds,
        params={
            "ring_names": ["ring_red", "ring_yellow", "ring_green", "ring_blue"],
            "x_bounds": (0.28, 0.47),
            "y_bounds": (-0.16, 0.06),
            "z_min": 0.69,
        },
    )


##
# Environment configuration
##


@configclass
class MDvrkEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: PegRingSceneCfg = PegRingSceneCfg(num_envs=1, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 10
        self.episode_length_s = 60
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
