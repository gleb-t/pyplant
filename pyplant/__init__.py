from pyplant.pyplant import *
import pyplant.test as test
# from pyplant.test.PipeworkMock import *
# from pyplant.test.test_utils import *

# A hack which hides the magic methods from PyCharm,
# so that the missing field warnings still work.
ConfigBase.__getattribute__ = ConfigBase._getattribute_method
ConfigBase.__setattr__ = ConfigBase._setattr_method
