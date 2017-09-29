import unittest
import shutil
import tempfile
import numpy as np

from pyplant import *

class PyPlantTest(unittest.TestCase):

    def __init__(self):
        self.executedReactors = []

    def setUp(self):
        super().setUp()

        self.plantDir = tempfile.mkdtemp()


    def tearDown(self):
        super().tearDown()

        shutil.rmtree(self.plantDir)

    def construct_plant(self, config):

        @ReactorFunc
        def reactor_a(pipe: Pipework):
            sliceParam = pipe.read_config('slice-param')
            yield

        plant = Plant(self.plantDir)
        plant.set_config(config)
        plant.add_reactors(reactor_a)

        return plant

    def shutdown_plant(self, plant: Plant):
        plant.shutdown()

    def test_slice_as_config_param(self):
        plant = self.construct_plant({
            'slice-param': slice(0, 10, None)
        })
