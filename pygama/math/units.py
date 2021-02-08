from scimath.units.unit import unit
from scimath.units.api import unit_parser
from scimath.units.time import *
from scimath.units.frequency import *
from scimath.units.convert import convert, convert_str, InvalidConversion
import sys

ghz = 1000000000*hz
ghz.label = 'gHz'
gigahertz = ghz

mhz = 1000000*hz
mhz.label = 'MHz'
megahertz = mhz

unit_parser.parser.extend(sys.modules['pygama.dsp.units'])
