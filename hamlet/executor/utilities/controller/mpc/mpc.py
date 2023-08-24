__author__ = "MarkusDoepfert"
__credits__ = ""
__license__ = ""
__maintainer__ = "MarkusDoepfert"
__email__ = "markus.doepfert@tum.de"

import linopy as lp
import pyomo.environ as pyo


class MpcBase:
    def run(self):
        raise NotImplementedError()


class Mpc:

    def __init__(self, **kwargs):
        self.method = kwargs['method'] if 'method' in kwargs.keys() else 'linopy'
        self.kwargs = kwargs

        # Mapping from input string to class name
        self.class_mapping = {
            'linopy': self.Linopy,
        }

    def run(self, **kwargs):
        # Return if no method is specified
        if self.method is None:
            return

        # Use the mapping to get the class
        controller_class = self.class_mapping.get(self.method.lower())

        if controller_class is None:
            raise ValueError(f"Unsupported language: {self.method}")

        return controller_class().run()

    class Linopy(MpcBase):
        def run(self):
            print('Running Linopy')
