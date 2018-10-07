# -*- coding: utf-8 -*-

__version__ = "0.1.0"

#kill annoying h5py warning
import warnings
warnings.filterwarnings(action="ignore", module="h5py", category=FutureWarning)

# pygama/decoders/__init__.py
from .dataloading import get_decoders
from .dataloading import get_next_event
# from .dataloading import DataLoader
from .digitizers import get_digitizers
from .digitizers import Gretina4MDecoder
from .digitizers import SIS3302Decoder
from .pollers import MJDPreampDecoder
from .pollers import ISegHVDecoder
__all__ = [
    "get_decoders",
    "get_next_event",
    "get_digitizers",
    # "DataLoader",
    #digitizers
    "Gretina4MDecoder",
    "SIS3302Decoder",
    #pollers
    "MJDPreampDecoder",
    "ISegHVDecoder"
]

# pygama/decoders/__init__.py
from ._pygama import TierOneProcessorList
from ._processing import process_tier_0
from ._processing import process_tier_1
from .processors import Calculator
from .processors import Transformer
from .processors import DatabaseLookup
# from .processors import Tier0Passer
__all__ = [
    "process_tier_0",
    "process_tier_1",
    "Calculator",
    "Transformer",
    "DatabaseLookup",
    # "Tier0Passer",
    "TierOneProcessorList"
]
