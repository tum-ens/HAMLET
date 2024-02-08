__author__ = "MarkusDoepfert"
__credits__ = ""
__license__ = ""
__maintainer__ = "MarkusDoepfert"
__email__ = "markus.doepfert@tum.de"

# Used to ensure a consistent design of all markets


class ControllerBase:

    def __init__(self):
        pass

    def run(self):
        raise NotImplementedError()
