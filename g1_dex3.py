"""Unitree Dex3 hands on SONIC's G1 training asset (shared by isaac_sonic_bridge.py and isaac_viewer.py).

SONIC's training URDF (gear_sonic/data/assets/robot_description/urdf/g1/main.urdf) already contains the Dex3 links
with their real masses, but the 14 finger joints are `fixed`. `dex3_urdf()` writes a copy next to it with those
joints `revolute` (axes and limits are already in the file), so the body the policy was trained on is unchanged
and only the fingers can move. Import this module after the AppLauncher has started.
"""

import os
import re

from isaaclab.actuators import ImplicitActuatorCfg

# Dex3 motor order of rt/dex3/{left,right}/cmd and SONIC's ZMQ left/right_hand_joints
# (= gear_sonic_deploy/g1/g1_29dof_with_hand.xml joint order; note index/middle swap between the hands)
DEX3_LEFT = ["left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
             "left_hand_middle_0_joint", "left_hand_middle_1_joint",
             "left_hand_index_0_joint", "left_hand_index_1_joint"]
DEX3_RIGHT = ["right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
              "right_hand_index_0_joint", "right_hand_index_1_joint",
              "right_hand_middle_0_joint", "right_hand_middle_1_joint"]
NUM_DEX3 = 7
# what the real hand gets before any command arrives (SONIC's Dex3Hands default gains)
DEX3_KP, DEX3_KD = 1.5, 0.1


def dex3_urdf(training_urdf: str) -> str:
    """Path of a copy of SONIC's training URDF with the Dex3 finger joints made revolute (written once)."""
    out = os.path.join(os.path.dirname(training_urdf), "main_dex3_fingers.urdf")
    text = open(training_urdf).read()
    pattern = r'(<joint\s+name="(?:left|right)_hand_(?:thumb|index|middle)_\d_joint"\s+type=")fixed(")'
    new, n = re.subn(pattern, r"\1revolute\2", text)
    assert n == 2 * NUM_DEX3, f"expected {2 * NUM_DEX3} Dex3 finger joints in {training_urdf}, found {n}"
    if not os.path.exists(out) or open(out).read() != new:
        with open(out, "w") as f:
            f.write(new)
    return out


def add_dex3(cfg, sonic_root: str):
    """Switch an ArticulationCfg built from G1_CYLINDER_MODEL_12_DEX_CFG to the movable-finger URDF."""
    cfg.spawn.asset_path = dex3_urdf(os.path.join(sonic_root, cfg.spawn.asset_path))
    cfg.actuators["dex3"] = ImplicitActuatorCfg(
        joint_names_expr=[".*_hand_(thumb|index|middle)_[0-2]_joint"],
        effort_limit_sim={".*_hand_thumb_0_joint": 2.45,  # URDF / Dex3 spec
                          ".*_hand_(thumb_[12]|index_[01]|middle_[01])_joint": 1.4},
        velocity_limit_sim=12.0,
        stiffness=0.0,  # gains come from rt/dex3/*/cmd
        damping=0.0,
        armature=0.0,
    )
    return cfg


def filter_hand_self_collisions(root: str = "/World/Robot") -> int:
    """Disable contacts between the links of the same hand (palm/wrist + fingers); contacts with objects stay on.

    The Dex3 collision meshes overlap at the knuckles and when the fingers close, so PhysX pushes the thumb into its
    joint limits; the real hand's fingers do not collide with each other. Call before sim.reset().
    """
    import omni.usd
    from pxr import UsdPhysics

    stage = omni.usd.get_context().get_stage()
    n = 0
    for side in ("left", "right"):
        links = [p for p in stage.Traverse()
                 if str(p.GetPath()).startswith(root + "/") and p.HasAPI(UsdPhysics.RigidBodyAPI)
                 and (p.GetName().startswith(f"{side}_hand_") or p.GetName() == f"{side}_wrist_yaw_link")]
        for a in links:
            rel = UsdPhysics.FilteredPairsAPI.Apply(a).CreateFilteredPairsRel()
            for b in links:
                if b is not a:
                    rel.AddTarget(b.GetPath())
                    n += 1
    return n
