__author__ = "TUM-Doepfert"
__credits__ = ""
__license__ = ""
__maintainer__ = "TUM-Doepfert"
__email__ = "markus.doepfert@tum.de"

from hamlet.creator.markets.markets import Markets


class HydrogenMarket(Markets):

    def __init__(self):
        super().__init__()
