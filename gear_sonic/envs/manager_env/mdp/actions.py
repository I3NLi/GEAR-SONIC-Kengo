from __future__ import annotations

from isaaclab.utils import configclass


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = None
