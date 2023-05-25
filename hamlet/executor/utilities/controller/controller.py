# Class is supposed to be similar to the agent one where it allows to include new methods
# Currently really rough draft and needs some thinking. Should probably work like pyomo or so

from rtc import Rtc
from mpc import Mpc


class Controller:

    def __init__(self, method: str = None, **kwargs):
        self.kwargs = kwargs

        self.options = {
            'rtc': Rtc,
            'mpc': Mpc,
        }

        self.method = method if method in self.options else 'mpc'

    def rtc(self):
        return Rtc(**self.kwargs)
