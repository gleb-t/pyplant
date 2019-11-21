from pyplant.pyplant import *
import pyplant.test as test

# A hack which hides the magic methods from PyCharm,
# so that the missing field warnings still work.
ConfigBase.__getattribute__ = ConfigBase._getattribute_method
ConfigBase.__setattr__ = ConfigBase._setattr_method
