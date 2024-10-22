__author__ = "MarkusDoepfert"
__credits__ = ""
__license__ = ""
__maintainer__ = "MarkusDoepfert"
__email__ = "markus.doepfert@tum.de"

import logging

from hamlet import constants as c
from hamlet.executor.utilities.controller.controller_base import ControllerBase
from hamlet.executor.utilities.controller.rtc.rtc_base import RtcBase
from hamlet.executor.utilities.controller.rtc.linopy.rtc_linopy import Linopy
from hamlet.executor.utilities.controller.rtc.poi.rtc_poi import POI

# warnings.filterwarnings("ignore")
logging.getLogger('linopy').setLevel(logging.CRITICAL)


class Rtc(ControllerBase):

    def __init__(self, method='poi', **kwargs):

        # Call the super class
        super().__init__()

        # Store the method and kwargs
        self.method = method
        self.kwargs = kwargs

        # Mapping from input string to class name
        self.class_mapping = {
            'linopy': Linopy,
            'poi': POI,
            'rule-based': RuleBased
        }

    def run(self, **kwargs):
        # Return if no method is specified
        if self.method is None:
            return

        # Use the mapping to get the class
        controller_class = self.class_mapping.get(self.method.lower())

        if controller_class is None:
            raise ValueError(f"Unsupported method: {self.method}.\n"
                             f"The available methods are: {self.class_mapping.keys()}")

        return controller_class(**kwargs, mapping=c.COMP_MAP).run()


class RuleBased(RtcBase):  # Note the change in class name

    def __init__(self, **kwargs):
        pass

    def run(self):
        print('Running Rule-Based')
