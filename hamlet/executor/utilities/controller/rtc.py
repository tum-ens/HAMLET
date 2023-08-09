__author__ = "MarkusDoepfert"
__credits__ = ""
__license__ = ""
__maintainer__ = "MarkusDoepfert"
__email__ = "markus.doepfert@tum.de"

import linopy as lp
from pprint import pprint


class RtcBase:
    def run(self):
        raise NotImplementedError()


class Rtc:

    def __init__(self, **kwargs):
        self.method = kwargs['method'] if 'method' in kwargs.keys() else 'linopy'
        self.kwargs = kwargs

        # Mapping from input string to class name
        self.class_mapping = {
            'linopy': self.Linopy,
            'rule-based': self.RuleBased
        }

    def run(self, **kwargs):
        print(self.method)
        pprint(kwargs)
        # Return if no method is specified
        if self.method is None:
            return

        # Use the mapping to get the class
        controller_class = self.class_mapping.get(self.method.lower())

        if controller_class is None:
            raise ValueError(f"Unsupported language: {self.method}")

        return controller_class().run()

    class Linopy(RtcBase):
        def run(self):
            print('Running Linopy')

    class RuleBased(RtcBase):  # Note the change in class name
        def run(self):
            print('Running Rule-Based')

