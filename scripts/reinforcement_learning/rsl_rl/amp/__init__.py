"""AMP (Adversarial Motion Priors) 模块"""

from .motion_loader import AMPLoader
from .himmy_motion_loader import HimmyAMPLoader

__all__ = ["AMPLoader", "HimmyAMPLoader"]
