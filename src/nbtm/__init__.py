"""
NBTM: Nonparametric Bayesian Topic Modeling

A research framework for comparing topic modeling algorithms.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from nbtm.models import create_model, get_available_models

__all__ = [
    "__version__",
    "create_model",
    "get_available_models",
]
