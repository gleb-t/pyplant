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
        self.startedSubreactors = []
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

    def _construct_plant(self, config, reactors=None):

        def _on_reactor_started(eventType, reactorName, reactorFunc=None):
            if eventType == Plant.EventType.reactor_started:
                self.startedReactors.append(reactorName)
            elif eventType == Plant.EventType.subreactor_started:
                self.startedSubreactors.append(reactorName)

        self.plant = Plant(self.plantDir, functionHash=self._compute_function_hash)
        self.plant.__enter__()
        self.plant.add_event_listener(Plant.EventType.reactor_started, _on_reactor_started)
        self.plant.add_event_listener(Plant.EventType.subreactor_started, _on_reactor_started)
        self.plant.set_config(config)
        if reactors is not None:
            self.plant.add_reactors(*tuple(reactors))

        return self.plant

    def _reconstruct_plant(self, config):
        reactors = self.plant.get_reactors()
        self._shutdown_plant()
        self.startedReactors = []
        self.startedSubreactors = []

        self._construct_plant(config)
        self.plant.add_reactors(*tuple(reactors))

    def _shutdown_plant(self):
        self.plant.shutdown()

    def test_events(self):
        # We rely on events working to perform other tests.
        # Here we check whether the events themselves work correctly
        # by manually tracking which reactors start execution.
        actuallyStartedReactors, actuallyStartedSubreactors = [], []

        @SubreactorFunc
        def subreactor_a(pipe: Pipework):
            actuallyStartedSubreactors.append('subreactor_a')
            a = 3 + 10  # Do stuff.
            yield

        @SubreactorFunc
        def subreactor_b(pipe: Pipework):
            actuallyStartedSubreactors.append('subreactor_b')
            yield from subreactor_a(pipe)

        @ReactorFunc
        def reactor_a(pipe: Pipework):
            actuallyStartedReactors.append('reactor_a')
            yield from subreactor_b(pipe)
            pipe.send('a-to-b', 1, Ingredient.Type.simple)
            yield

        @ReactorFunc
        def reactor_b(pipe: Pipework):
            actuallyStartedReactors.append('reactor_b')
            aToB = yield pipe.receive('a-to-b')
            yield from subreactor_a(pipe)
            yield

        @ReactorFunc
        def reactor_c(pipe: Pipework):
            actuallyStartedReactors.append('reactor_c')
            yield from subreactor_a(pipe)

        self._construct_plant({}, [reactor_a, reactor_b, reactor_c])
        self.plant.run_reactor('reactor_b')

        # Note, that we don't care about the concrete execution order here,
        # we just care that it's the same as the event system reports it to be.
        self.assertEqual(self.startedReactors, actuallyStartedReactors)
        self.assertEqual(self.startedSubreactors, actuallyStartedSubreactors)

        # Now test with reactor C.
        self._reconstruct_plant({})
        actuallyStartedReactors, actuallyStartedSubreactors = [], []
        self.plant.run_reactor('reactor_c')

        self.assertEqual(self.startedReactors, actuallyStartedReactors)
        self.assertEqual(self.startedSubreactors, actuallyStartedSubreactors)

    def test_slice_as_config_param(self):
        config = {'slice-param': slice(0, 10, None)}

        @ReactorFunc
        def reactor_a(pipe: Pipework):
            sliceParam = pipe.read_config('slice-param')

            pipe.send('a-to-b', 1, Ingredient.Type.simple)
            yield

        @ReactorFunc
        def reactor_b(pipe: Pipework):
            aToB = yield pipe.receive('a-to-b')

            yield

        # First pass.
        self._construct_plant(config, [reactor_a, reactor_b])
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

    def test_simple_ingredients(self):

        ingredients = {
            'int': 42,
            'float': 19.84,
            'bool': False,
            'string': 'Ex machina',
            'list': [4, 8, 15, 16, 23, 42],
            'tuple': (1, 2, 4, 1),
            'dict': {'caesar': 'caesar\'s', 'lord': 'lord\'s'}
        }

        received = None

        @ReactorFunc
        def producer(pipe: Pipework):
            for name, value in ingredients.items():
                pipe.send(name, value, Ingredient.Type.simple)

            yield

        @ReactorFunc
        def consumer(pipe: Pipework):
            nonlocal received
            received = {}

            for name in ingredients:
                value = yield pipe.receive(name)
                received[name] = value

            yield

        self._construct_plant({}, [producer, consumer])
        self.plant.run_reactor(consumer)

        self.assertDictEqual(ingredients, received)
        for name, trueVal in ingredients.items():
            self.assertEqual(trueVal, received[name])
            self.assertEqual(type(trueVal), type(received[name]))

        received = None
        self._reconstruct_plant({})
        self.plant.run_reactor(consumer)

        self.assertEqual(self.startedReactors, ['consumer'], msg='The producer should be cached.')
        self.assertDictEqual(ingredients, received)
        for name, trueVal in ingredients.items():
            self.assertEqual(trueVal, received[name])
            self.assertEqual(type(trueVal), type(received[name]))

    def test_subreactors_basic(self):
        config = {'param-a': 1, 'param-b': 2}

        @SubreactorFunc
        def subreactor_a(pipe: Pipework):
            a = 3 + 10  # Do stuff.
            yield

        @SubreactorFunc
        def subreactor_b(pipe: Pipework):
            b = 3 + 10  # Do stuff.
            yield

        @ReactorFunc
        def reactor_a(pipe: Pipework):
            paramA = pipe.read_config('param-a')
            yield from subreactor_a(pipe)
            yield from subreactor_b(pipe)
            pipe.send('a-to-b', 1, Ingredient.Type.simple)
            yield

        @ReactorFunc
        def reactor_b(pipe: Pipework):
            aToB = yield pipe.receive('a-to-b')
            yield from subreactor_b(pipe)
            yield

        # First pass, everything is executed.
        self._construct_plant(config, [reactor_a, reactor_b])
        self.plant.run_reactor('reactor_b')

        self.assertEqual(self.startedSubreactors, ['subreactor_a', 'subreactor_b', 'subreactor_b'])

        # Reactor A should be cached, subreactor B should still be called.
        self._reconstruct_plant(config)
        self.plant.run_reactor(reactor_b)

        self.assertEqual(self.startedSubreactors, ['subreactor_b'])

        # Changing a subreactor should cause reactor re-execution.
        self.modifiedReactors.append('subreactor_a')
        self._reconstruct_plant(config)
        self.plant.run_reactor(reactor_b)

        self.assertEqual(self.startedReactors, ['reactor_b', 'reactor_a'])
        self.assertEqual(self.startedSubreactors, ['subreactor_a', 'subreactor_b', 'subreactor_b'])

    def test_subreactors_nested(self):

        config = {'param-a': 1, 'param-b': 2}

        @SubreactorFunc
        def subreactor_a(pipe: Pipework):
            a = 3 + 10  # Do stuff.
            yield

        @SubreactorFunc
        def subreactor_b(pipe: Pipework):
            yield from subreactor_a(pipe)

        @ReactorFunc
        def reactor_a(pipe: Pipework):
            yield from subreactor_b(pipe)
            pipe.send('a-to-b', 1, Ingredient.Type.simple)
            yield

        @ReactorFunc
        def reactor_b(pipe: Pipework):
            aToB = yield pipe.receive('a-to-b')
            yield from subreactor_a(pipe)
            yield

        # First pass, everything is executed.
        self._construct_plant(config, [reactor_a, reactor_b])
        self.plant.run_reactor('reactor_b')

        self.assertEqual(self.startedSubreactors, ['subreactor_b', 'subreactor_a', 'subreactor_a'])

        # Invalidating the nested subreactor should cause re-execution.
        self.modifiedReactors.append('subreactor_a')
        self._reconstruct_plant(config)
        self.plant.run_reactor(reactor_b)

        self.assertEqual(self.startedSubreactors, ['subreactor_b', 'subreactor_a', 'subreactor_a'])

    def test_subreactors_input_dependencies(self):
        config = {'param-a': 1, 'param-b': 2}

        @SubreactorFunc
        def subreactor_a(pipe: Pipework):
            a = 3 + 10  # Do stuff.
            pipe.read_config('param-a')
            yield

        @SubreactorFunc
        def subreactor_b(pipe: Pipework):
            midToB = yield pipe.receive('mid-to-b')
            yield pipe.receive('a-to-b')

        @ReactorFunc
        def reactor_a(pipe: Pipework):
            yield from subreactor_a(pipe)
            pipe.send('a-to-b', 1, Ingredient.Type.simple)
            pipe.send('a-to-mid', 1, Ingredient.Type.simple)
            yield

        @ReactorFunc
        def reactor_mid(pipe: Pipework):
            aToMid = yield pipe.receive('a-to-mid')
            pipe.send('mid-to-b', 5, Ingredient.Type.simple)
            yield

        @ReactorFunc
        def reactor_b(pipe: Pipework):
            yield from subreactor_b(pipe)

        # First pass, everything is executed, reactor B depends on A through a subreactor.
        self._construct_plant(config, [reactor_a, reactor_mid, reactor_b])
        self.plant.run_reactor('reactor_b')

        self.assertEqual(self.startedSubreactors, ['subreactor_b', 'subreactor_a'])

        # Results are cached.
        self._reconstruct_plant(config)
        self.plant.run_reactor(reactor_b)

        self.assertEqual(self.startedReactors, ['reactor_b'])
        self.assertEqual(self.startedSubreactors, ['subreactor_b'])

        # Modifying a parameter used by a subreactor causes re-execution.
        config['param-a'] = 99
        self._reconstruct_plant(config)
        self.plant.run_reactor(reactor_b)

        self.assertEqual(self.startedReactors, ['reactor_b', 'reactor_mid', 'reactor_a'])
        self.assertEqual(self.startedSubreactors, ['subreactor_b', 'subreactor_a'])

        # Modifying a dependency of a subreactor, causes its re-execution.
        self.modifiedReactors.append('reactor_mid')
        self._reconstruct_plant(config)
        self.plant.run_reactor(reactor_b)

        self.assertEqual(self.startedReactors, ['reactor_b', 'reactor_mid'])
        self.assertEqual(self.startedSubreactors, ['subreactor_b'])

    def test_state_saved_on_crash(self):
        """
        Test that when a crash occurs, we are saving the results of all the previously executed reactors.
        This feature helps to avoid re-running data loading stuff when new fancy functionality
        crashes late in the pipeline.

        :return:
        """
        config = {
            'param-a': 5,
            'param-b': 1
        }

        @ReactorFunc
        def reactor_a(pipe: Pipework):
            a = 3 + 10
            pipe.read_config('param-a')

            pipe.send('simple', 5, Ingredient.Type.simple)
            yield

        @ReactorFunc
        def reactor_b(pipe: Pipework):
            a = 3 + 10
            simple = yield pipe.receive('simple')
            b = pipe.read_config('param-b')
            if b == 1:
                raise RuntimeError('Intentional crash.')

            yield

        self._construct_plant(config, [reactor_a, reactor_b])
        with self.assertRaises(RuntimeError):
            self.plant.run_reactor(reactor_b)

        self.assertEqual(self.startedReactors, ['reactor_b', 'reactor_a'])

        # Terminate the current plant without saving, then recreate it.
        # self._reconstruct_plant(config)  <-- This would be a graceful termination.
        self.startedReactors = []
        self.plant.warehouse.close()
        self._construct_plant(config)
        self.plant.add_reactors(reactor_a, reactor_b)

        with self.assertRaises(RuntimeError):
            self.plant.run_reactor(reactor_b)

        self.assertEqual(self.startedReactors, ['reactor_b'])  # No need to rerun reactor_a.