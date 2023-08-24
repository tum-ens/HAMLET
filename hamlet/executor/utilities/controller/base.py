__author__ = "MarkusDoepfert"
__credits__ = ""
__license__ = ""
__maintainer__ = "MarkusDoepfert"
__email__ = "markus.doepfert@tum.de"

import hamlet.constants as c


class ControllerBase:

    def __init__(self):
        # Mapping of components to energy types (e.g. electricity) and operation modes (e.g. generation)
        # Note: Key states which type of energy is addressed and the value states which type of operation it usually has
        self.component_mapping = {
            # Electricity
            c.P_INFLEXIBLE_LOAD: {c.ET_ELECTRICITY: c.OM_LOAD},
            c.P_FLEXIBLE_LOAD: {c.ET_ELECTRICITY: c.OM_LOAD},
            c.P_PV: {c.ET_ELECTRICITY: c.OM_GENERATION},
            c.P_WIND: {c.ET_ELECTRICITY: c.OM_GENERATION},
            c.P_FIXED_GEN: {c.ET_ELECTRICITY: c.OM_GENERATION},
            c.P_EV: {c.ET_ELECTRICITY: c.OM_STORAGE},
            c.P_BATTERY: {c.ET_ELECTRICITY: c.OM_STORAGE},
            c.P_PSH: {c.ET_ELECTRICITY: c.OM_STORAGE},
            c.P_HYDROGEN: {c.ET_ELECTRICITY: c.OM_STORAGE},

            # Heat
            c.P_HEAT: {c.ET_HEAT: c.OM_LOAD},
            c.P_DHW: {c.ET_HEAT: c.OM_LOAD},
            c.P_HEAT_STORAGE: {c.ET_ELECTRICITY: c.OM_STORAGE},

            # Hybrid
            c.P_HP: {c.ET_ELECTRICITY: c.OM_LOAD, c.ET_HEAT: c.OM_GENERATION},
        }

    def run(self):
        raise NotImplementedError()