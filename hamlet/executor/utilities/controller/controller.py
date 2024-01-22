__author__ = "MarkusDoepfert"
__credits__ = ""
__license__ = ""
__maintainer__ = "MarkusDoepfert"
__email__ = "markus.doepfert@tum.de"

from hamlet.executor.utilities.controller.rtc.rtc import Rtc
from hamlet.executor.utilities.controller.mpc.mpc import Mpc
import hamlet.constants as c

# Instructions: For a new controller type import the class here and add it to the mapping in the Controller class


class Controller:

    def __init__(self, controller_type: str, **kwargs):
        self.kwargs = kwargs
        self.method = kwargs['method']

        # Mapping of controller types to classes
        controllers = {
            c.C_RTC: Rtc,
            c.C_MPC: Mpc,
        }

        # Lookup the class based on the controller_type
        self.controller = controllers.get(controller_type.lower())

        if self.controller is None:
            raise ValueError(f'Controller method {controller_type} not available. \n'
                             f'The available methods are: {controllers.keys()}. \n'
                             f'If you think this is an error, you might have forgotten to import the method to the '
                             f'class or named it incorrectly.')

    def create_instance(self):
        return self.controller(**self.kwargs)

