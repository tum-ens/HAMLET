# Class is supposed to be similar to the agent one where it allows to include new methods
# Currently really rough draft and needs some thinking. Should probably work like pyomo or so

from hamlet.executor.utilities.controller.rtc import Rtc
from hamlet.executor.utilities.controller.mpc import Mpc

# Instructions: For a new controller type import the class here and add it to the mapping in the Controller class

class Controller:

    def __init__(self, controller_type: str, **kwargs):
        self.kwargs = kwargs
        self.method = kwargs['method']

        # Mapping of controller types to classes
        controllers = {
            'rtc': Rtc,
            'mpc': Mpc,
        }

        # Lookup the class based on the controller_type
        self.controller = controllers.get(controller_type.lower())

        if self.controller is None:
            raise ValueError(f'Controller method {controller_type} not available. If you think this is an error, you '
                             f'might have forgotten to import the method to the class or named it incorrectly.')

    def create_instance(self):
        print(self.method)
        return self.controller(**self.kwargs)

