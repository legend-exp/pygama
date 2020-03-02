from scimath.units.api import unit_parser
from scimath.units.time import *
from scimath.units.frequency import *

ghz = 1000000000*hz
ghz.label = 'gHz'
gigahertz = ghz

mhz = 1000000*hz
mhz.label = 'MHz'
megahertz = mhz

#import pygama.dsp.units as u
#unit_parser.parser.extend(u)
