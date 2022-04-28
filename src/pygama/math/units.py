from pint import Measurement, Quantity, Unit, UnitRegistry, set_application_registry

unit_registry = UnitRegistry(auto_reduce_dimensions=True)
set_application_registry(unit_registry)

# Define constants useful for LEGEND below
unit_registry.define('m_76 = 75.921402729 * amu') # 10.1103/PhysRevC.81.032501
unit_registry.define('Q_bb = 2039.061 * keV') # 10.1103/PhysRevC.81.032501
