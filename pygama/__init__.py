# -*- coding: utf-8 -*-
""" pygama init file.  currently we declare some stuff (a public API)
    and it might be better to declare more of the public functions here if it
    makes sphinx auto-generated documentation easier.
"""
__version__ = "0.1.0"

# There's a nice discussion about what to include in this file here:
# https://www.reddit.com/r/Python/comments/1bbbwk/whats_your_opinion_on_what_to_include_in_init_py/
#
# Clint's fav answer:
#
#     I like importing key functions and classes. Flat is better than nested, so
#     as a user of a library, I prefer `from library import ThingIWant` or
#     `import library` and then using `library.ThingIWant` rather than
#     `from library.things.thing_i_want import ThingIWant`.
#
#     More specifically, I would often have the contents of __init__.py be:
#         "Docstring explaining package" (use triple quotes though)
#         from thispackage.module_or_subpackage import *
#         from thispackage.module_thats_next_alphabetically import *
#         ...
#     And then have each module use __all__ to specify which names constitute
#     its public API that should be exposed by the package.
#
#     For something you are distributing publicly, I don't think your __init__.py
#     should ever by "blank" as specified by option 1: you should at least include a
#     docstring explaining what the package does. This will help users poking around
#     in ipython, etc.

# kill annoying h5py warning
import warnings
warnings.filterwarnings(action="ignore", module="h5py", category=FutureWarning)

from .decoders.dataloading import get_decoders
from .decoders.dataloading import get_next_event

from .decoders.digitizers import get_digitizers
from .decoders.digitizers import Gretina4MDecoder
from .decoders.digitizers import SIS3302Decoder

from .decoders.pollers import MJDPreampDecoder
from .decoders.pollers import ISegHVDecoder

from .processing._pygama import TierOneProcessorList
from .processing._pygama import ProcessTier0
from .processing._pygama import ProcessTier1

from .processing.processors import process_tier_0
from .processing.processors import process_tier_1
from .processing.processors import Calculator
from .processing.processors import Transformer
from .processing.processors import DatabaseLookup

# __all__ = [
#     "get_decoders",
#     "get_next_event"
# ]

# # pygama/decoders/__init__.py
# from dataloading import get_decoders
# from dataloading import get_next_event
# # from .dataloading import DataLoader
# from .digitizers import get_digitizers
# from .digitizers import Gretina4MDecoder
# from .digitizers import SIS3302Decoder
# from .pollers import MJDPreampDecoder
# from .pollers import ISegHVDecoder
# __all__ = [
#     "get_decoders",
#     "get_next_event",
#     "get_digitizers",
#     # "DataLoader",
#     #digitizers
#     "Gretina4MDecoder",
#     "SIS3302Decoder",
#     #pollers
#     "MJDPreampDecoder",
#     "ISegHVDecoder"
# ]
#
# # pygama/decoders/__init__.py
# from ._pygama import TierOneProcessorList
# from ._processing import process_tier_0
# from ._processing import process_tier_1
# from .processors import Calculator
# from .processors import Transformer
# from .processors import DatabaseLookup
# # from .processors import Tier0Passer
# __all__ = [
#     "process_tier_0",
#     "process_tier_1",
#     "Calculator",
#     "Transformer",
#     "DatabaseLookup",
#     # "Tier0Passer",
#     "TierOneProcessorList"
# ]
