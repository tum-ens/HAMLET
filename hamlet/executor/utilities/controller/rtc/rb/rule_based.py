__author__ = "Replace this with your name(s)"
__credits__ = ""
__license__ = ""
__maintainer__ = "MarkusDoepfert"
__email__ = "markus.doepfert@tum.de"

from hamlet.executor.utilities.controller.rtc.rtc_base import RtcBase


class RuleBased(RtcBase):  # Note the change in class name

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self):
        print('Running Rule-Based')
        raise NotImplementedError()
