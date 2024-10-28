__author__ = "Replace this with your name(s)"
__credits__ = ""
__license__ = ""
__maintainer__ = "MarkusDoepfert"
__email__ = "markus.doepfert@tum.de"

from hamlet.executor.utilities.controller.fbc.fbc_base import FbcBase


class ReinforcementLearning(FbcBase):  # Note the change in class name

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self):
        print('Running Reinforcement Learning')
        raise NotImplementedError()
