__author__ = "MarkusDoepfert"
__credits__ = ""
__license__ = ""
__maintainer__ = "MarkusDoepfert"
__email__ = "markus.doepfert@tum.de"

# Note: Used to contain the component mapping which was moved to constants.py to give universal access.
# For now no specific use case for this class but kept for consistency and possible future uses.


class ControllerBase:

    def __init__(self):
        pass

    def run(self):
        raise NotImplementedError()
