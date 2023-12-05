__author__ = "MarkusDoepfert"
__credits__ = ""
__license__ = ""
__maintainer__ = "MarkusDoepfert"
__email__ = "markus.doepfert@tum.de"

# Used to ensure a consistent design of all markets


class MarketBase:

    def __init__(self):
        pass

    def execute(self):
        raise NotImplementedError('This market type is not implemented yet but the structure is already in place.')
