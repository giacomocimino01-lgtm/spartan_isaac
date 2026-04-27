# Copyright (c) 2026, Il Tuo Nome / Progetto
# All rights reserved.

"""Configuration for the da Vinci Research Kit (dVRK) robot."""

import os
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "usd"))
DAVINCI_USD_PATH = os.path.join(ASSETS_DIR, "psm_col.usd")

##
# Configuration
##

DAVINCI_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=DAVINCI_USD_PATH,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, 
            solver_position_iteration_count=64, 
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.15),
        
        joint_pos={
            "psm_yaw_joint": 0.01,
            "psm_pitch_end_joint": 0.01,
            "psm_main_insertion_joint": 0.07,
            "psm_tool_roll_joint": 0.01,
            "psm_tool_pitch_joint": 0.01,
            "psm_tool_yaw_joint": 0.01,
            "psm_tool_gripper1_joint": -0.09,
            "psm_tool_gripper2_joint": 0.09,
        },
    ),
    actuators={
        "psm": ImplicitActuatorCfg(
            joint_names_expr=[
                "psm_yaw_joint",
                "psm_pitch_end_joint",
                "psm_main_insertion_joint",
                "psm_tool_roll_joint",
                "psm_tool_pitch_joint",
                "psm_tool_yaw_joint",
            ],
            friction=0.0,
            dynamic_friction=0.0,
            viscous_friction=0.0,
            effort_limit=None,
            velocity_limit=None,
            stiffness=8000.0,
            damping=40.0,
        ),
        "psm_tool": ImplicitActuatorCfg(
            joint_names_expr=["psm_tool_gripper.*"],
            effort_limit=0.1,
            velocity_limit=0.2,
            stiffness=500.0,
            damping=0.1,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration for the daVinci dVRK robot arm."""