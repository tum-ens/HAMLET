__author__ = "MarkusDoepfert"
__credits__ = ""
__license__ = ""
__maintainer__ = "MarkusDoepfert"
__email__ = "markus.doepfert@tum.de"

import logging

from hamlet import constants as c
from hamlet.executor.utilities.controller.controller_base import ControllerBase
from hamlet.executor.utilities.controller.rtc.optim.linopy.optim_linopy import Linopy
from hamlet.executor.utilities.controller.rtc.optim.poi.optim_poi import POI
from hamlet.executor.utilities.controller.rtc.rb.rule_based import RuleBased
from hamlet.executor.utilities.database.agent_db import AgentDB

# warnings.filterwarnings("ignore")
logging.getLogger('linopy').setLevel(logging.CRITICAL)


class Rtc(ControllerBase):

    def __init__(self, method: str, **kwargs):

        # Call the super class
        super().__init__()

        # Store the method and kwargs
        self.method = method
        self.kwargs = kwargs

        # Mapping from input string to class name
        self.class_mapping = {
            c.C_LINOPY: Linopy,
            c.C_POI: POI,
            c.C_RB: RuleBased,
        }

    def run(self, **kwargs) -> AgentDB:
        # Return if no method is specified
        if self.method is None:
            return

        # Use the mapping to get the class
        controller_class = self.class_mapping.get(self.method.lower())

        if controller_class is None:
            raise ValueError(f"Unsupported method: {self.method}.\n"
                             f"The available methods are: {self.class_mapping.keys()}")

        return controller_class(**kwargs, mapping=c.COMP_MAP).run()

