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


##
# Scene definition
##

@configclass
class PegRingSceneCfg(InteractiveSceneCfg):
    """1. Configurazione della Scena (Gli Attori)"""
    
    # Pavimento e Luce
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    # Tavolo Operatorio
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table", 
        spawn=sim_utils.UsdFileCfg(usd_path=f"{USD_DATA_DIR}/table.usd"),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.4, 0.0, 0.25)),
    )

    # La Basetta del Peg and Ring
    peg_board = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/PegBoard", # <--- Tolto /Table/
        spawn=sim_utils.UsdFileCfg(usd_path=f"{USD_DATA_DIR}/pn_base2.usd"),
        # Z=0.51: Se il tavolo è alto 0.5, la poggiamo proprio sopra
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.375, -0.05, 0.71), rot=(0.0, 0.0, 0.0, 1.0)), 
    )

    # ==========================================
    # TRACCIAMENTO XFORM DEI PEG
    # ==========================================
    peg_red = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/PegBoard/peg_red/peg_red", # Assicurati che questo nome corrisponda all'albero del tuo USD
        spawn=None, # FONDAMENTALE: Non spawna niente, traccia e basta l'oggetto esistente!
    )
    peg_yellow = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/PegBoard/peg_yellow/peg_yellow",
        spawn=None,
    )
    peg_green = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/PegBoard/peg_green/peg_green",
        spawn=None,
    )
    peg_blue = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/PegBoard/peg_blue/peg_blue",
        spawn=None,
    )

    peg_gray = AssetBaseCfg(
        # ATTENZIONE: Controlla che questo path corrisponda al tuo albero USD!
        prim_path="{ENV_REGEX_NS}/PegBoard/peg_gray/peg_gray", 
        spawn=None,
    )

    peg_gray1 = AssetBaseCfg(
        # ATTENZIONE: Controlla che questo path corrisponda al tuo albero USD!
        prim_path="{ENV_REGEX_NS}/PegBoard/peg_gray1/peg_gray1", 
        spawn=None,
    )

    # # Anello Rosso
    # ring_red = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/RingRed", # <--- Tolto /Table/PegBoard/
    #     spawn=sim_utils.UsdFileCfg(usd_path=f"{USD_DATA_DIR}/red_ring.usd"),
    #     # Z=0.7: 20 centimetri SOPRA il tavolo. Cadranno come pioggia.
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.1, 0.77)), 
    # )

    # Anello Rosso (In alto a destra rispetto al centro)
    ring_red = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/RingRed",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{USD_DATA_DIR}/red_ring.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.01),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.375, -0.04, 0.77)), 
    )

    # Anello Giallo (In basso a destra)
    ring_yellow = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/RingYellow",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{USD_DATA_DIR}/yellow_ring.usd"),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.375, -0.04, 0.77)),
    )

    # Anello Verde (In alto a sinistra)
    ring_green = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/RingGreen",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{USD_DATA_DIR}/green_ring.usd"),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.375, -0.04, 0.77)),
    )

    # Anello Blu (In basso a sinistra)
    ring_blue = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/RingBlue",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{USD_DATA_DIR}/blue_ring.usd"),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.375, -0.04, 0.77)),
    )

    # Il Robot daVinci

    # ==========================================
    # ROBOT (Destro e Sinistro)
    # ==========================================
    # Braccio Destro (Spostato di -25 cm sull'asse Y)
    robot_right: ArticulationCfg = DAVINCI_CFG.replace(prim_path="{ENV_REGEX_NS}/RobotRight")
    robot_right.init_state.pos = (0.25, -0.15, 0.8) 
    #robot_right.init_state.rot = (0.7071, 0.0, 0.0, -0.7071)
    #robot_right.init_state.rot = (0.9239, 0.0, 0.3827, 0.0)
    robot_right.init_state.joint_pos = {
        "psm_yaw_joint": 0.5,
        "psm_pitch_end_joint": -0.5,      
        "psm_main_insertion_joint": 0.08, 
        "psm_tool_roll_joint": 0.0,
        "psm_tool_pitch_joint": -0.785,
        "psm_tool_yaw_joint": 0.0,
    }

    # Braccio Sinistro (Spostato di +25 cm sull'asse Y)
    robot_left: ArticulationCfg = DAVINCI_CFG.replace(prim_path="{ENV_REGEX_NS}/RobotLeft")
    robot_left.init_state.pos = (0.5, -0.15, 0.8)
    #robot_left.init_state.rot = (0.7071, 0.0, 0.0, -0.7071)
    #robot_left.init_state.rot = (0.9239, 0.0, -0.3827, 0.0)

    # robot: ArticulationCfg = DAVINCI_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # # Forziamo la posizione iniziale del robot per metterlo SOPRA il tavolo (Z=0.5)
    # robot.init_state.pos = (0.25, 0.0, 0.8)

    # ==========================================
    # TELECAMERA
    # ==========================================
    camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        update_period=0.0,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            # focale più bassa = grandangolo (tipico degli endoscopi per vedere entrambi i bracci)
            focal_length=18.0, 
            focus_distance=400.0, 
            horizontal_aperture=20.955, 
            # Clipping vicino ridotto a 1 cm (0.01) così non taglia via i bracci se si avvicinano troppo alla lente
            clipping_range=(0.01, 1.0e5) 
        ),
        offset=CameraCfg.OffsetCfg(
            # POSIZIONE: La mettiamo dietro e in mezzo ai due bracci (X=0.15, Y=0.0) 
            # e a un'altezza di 90 cm (Z=0.9), poco sopra le basi dei robot
            pos=(0.375, 0.15, 0.85), 
            
            # ROTAZIONE: Guarda dritto davanti a sé (verso X=0.4) e si inclina in basso di ~30 gradi
            # (Quaternione calcolato: Pitch -30°, Yaw -90°, Roll 0°)
            rot=(0.683, 0.183, 0.183, -0.683), #(-0.7596879, 0, 0.6502878, 0),#(-0.3990808, -0.5533322, 0.3416105, -0.6464211),# [ -0.21829, -0.4355324, -0.7807314, 0.3913048 ]
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
            ], # Regex per i giunti del braccio che l'IK può muovere

        body_name="psm_tool_tip_link",

        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=True,   # False = Coordinate assolute nel mondo (meglio per la Macchina a Stati)
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
            ], # Regex per i giunti del braccio che l'IK può muovere

        body_name="psm_tool_tip_link",

        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=True,   # False = Coordinate assolute nel mondo (meglio per la Macchina a Stati)
            ik_method="pinv",
        ),
        scale=0.5,
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
        #joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # reset
    reset_robot_right = EventTerm(
        #func=mdp.reset_joints_by_scale,
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot_right"),
            "position_range": (0.0, 0.0), # o quello che avevi prima
            "velocity_range": (0.0, 0.0),
        },
    )
    
    reset_robot_left = EventTerm(
        #func=mdp.reset_joints_by_scale,
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot_left"),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_ring_red = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("ring_red"),
            # Ridotto a +/- 1 cm
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.06, -0.02), "yaw": (-3.14, 3.14)},
            "velocity_range": {},
        },
    )
    
    reset_ring_yellow = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("ring_yellow"),
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.06, -0.02), "yaw": (-3.14, 3.14)},
            "velocity_range": {},
        },
    )

    reset_ring_green = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("ring_green"),
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.06, -0.02), "yaw": (-3.14, 3.14)},
            "velocity_range": {},
        },
    )

    reset_ring_blue = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("ring_blue"),
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.06, -0.02), "yaw": (-3.14, 3.14)},
            "velocity_range": {},
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
    # (2) Cart out of bounds
    # cart_out_of_bounds = DoneTerm(
    #     func=mdp.joint_pos_out_of_manual_limit,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]), "bounds": (-3.0, 3.0)},
    # )


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
        self.decimation = 2
        self.episode_length_s = 5000
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation