import unittest
import shutil
import tempfile
import inspect
from typing import Callable
import numpy as np

from pyplant import *


class PyPlantTest(unittest.TestCase):

    # Used for invalidating reactor hashes in a deterministic way.
    uniqueReactorHashPostfix = 0

    def __init__(self, methodName='runTest'):
        super().__init__(methodName)

        self.plant = None  # type: Plant

    def setUp(self):
        super().setUp()

        self.plantDir = tempfile.mkdtemp()
        self.startedReactors = []
        self.modifiedReactors = []

    def tearDown(self):
        super().tearDown()

        self._shutdown_plant()
        shutil.rmtree(self.plantDir)

    def _compute_function_hash(self, func: Callable) -> str:
        name = func.__name__
        if name in self.modifiedReactors:
            PyPlantTest.uniqueReactorHashPostfix += 1
            return name + str(PyPlantTest.uniqueReactorHashPostfix)

        return name

    def _construct_plant(self, config):

        @ReactorFunc
        def reactor_a(pipe: Pipework):
            sliceParam = pipe.read_config('slice-param')

            pipe.send('a-to-b', 1, Ingredient.Type.simple)
            yield

        @ReactorFunc
        def reactor_b(pipe: Pipework):
            aToB = yield pipe.receive('a-to-b')

            yield

        def _on_reactor_started(eventType, reactorName, reactorFunc):
            self.startedReactors.append(reactorName)

        self.plant = Plant(self.plantDir, functionHash=self._compute_function_hash)
        self.plant.__enter__()
        self.plant.add_event_listener(Plant.EventType.reactor_started, _on_reactor_started)
        self.plant.set_config(config)
        self.plant.add_reactors(reactor_a, reactor_b)

        return self.plant

    def _reconstruct_plant(self, config):
        self._shutdown_plant()
        self.startedReactors = []

        self._construct_plant(config)

    def _shutdown_plant(self):
        self.plant.shutdown()

    def test_slice_as_config_param(self):
        config = {'slice-param': slice(0, 10, None)}

        # First pass.
        self._construct_plant(config)
        self.plant.run_reactor('reactor_b')

        self.assertEqual(self.startedReactors, ['reactor_b', 'reactor_a'])

        # Second pass, config changed.
        config['slice-param'] = slice(0, 20, None)
        self._reconstruct_plant(config)
        self.plant.run_reactor('reactor_b')

        self.assertEqual(self.startedReactors, ['reactor_b', 'reactor_a'])

        # Third pass, config unchanged.
        self._reconstruct_plant(config)
        self.plant.run_reactor('reactor_b')
        self.assertEqual(self.startedReactors, ['reactor_b'])


