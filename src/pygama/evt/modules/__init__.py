"""
Contains submodules for evt processing
"""

from .spm import (
    get_energy,
    get_energy_dplms,
    get_etc,
    get_majority,
    get_majority_dplms,
    get_time_shift,
)

__all__ = [
    "get_energy",
    "get_majority",
    "get_energy_dplms",
    "get_majority_dplms",
    "get_etc",
    "get_time_shift",
]
