# -*- coding: utf-8 -*-
""" pygama init file.
    it might be better to declare more of the public functions here if it
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

# kill annoying warnings
import warnings, pandas
warnings.filterwarnings(action="ignore", module="h5py", category=FutureWarning)
warnings.filterwarnings(
    'ignore', category=pandas.io.pytables.PerformanceWarning)

# # try to import all public functions
# import pkgutil
# __path__ = pkgutil.extend_path(__path__, __name__)
# for importer, modname, ispkg in pkgutil.walk_packages(
#         path=__path__, prefix=__name__ + '.'):
#     __import__(modname)

# bring key functions to the primary API, e.g. pygama.process_tier_0
# i don't really like how i have to specify everything here
# from .processing.processing import *
# from .processing.base_classes import *
# from .decoders.digitizers import *
# from .processing.vectorized import *
# from .waveform import *
from .dataset import DataSet

# from .decoders.dataloading import get_decoders
# from .decoders.dataloading import get_next_event
# from .decoders.digitizers import get_digitizers
# from .decoders.digitizers import Gretina4MDecoder
# from .decoders.digitizers import SIS3302Decoder
# from .decoders.pollers import MJDPreampDecoder
# from .decoders.pollers import ISegHVDecoder
# from .processing.processing import process_tier_0
# from .processing.processing import process_tier_1
# from .processing.base_classes import Calculator
# from .processing.base_classes import Transformer
# from .processing.base_classes import DatabaseLookup
# from .analysis.calibration import get_most_prominent_peaks
