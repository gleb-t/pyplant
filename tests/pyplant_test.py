import logging
import os
import shutil
import tempfile
import unittest
from typing import Callable, Optional, Any, Dict

import numpy as np

from pyplant import *
from pyplant.pyplant import IngredientTypeSpec, Warehouse


class DummyClass:
    pass

class CustomSpec(IngredientTypeSpec):

    # Some custom type for this ingredient's values.
    class ValueWrapper:
        def __init__(self, val):
            self.val = val

    def __init__(self):
        super().__init__()
        self.valueStore = {}

    @classmethod
    def get_name(cls) -> str:
        return 'test_custom_spec'

    def store(self, warehouse: Warehouse, name: str, value: Any) -> Optional[Dict[str, Any]]:
        self.valueStore[name] = value
        return None

    def fetch(self, warehouse: Warehouse, name: str, meta: Optional[Dict[str, Any]]) -> Any:
        return self.valueStore.get(name, default=None)

    def is_instance(self, value) -> bool:
        return isinstance(value, CustomSpec.ValueWrapper)


# noinspection PyCompatibility
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

        # Ingredient specs that are enabled by default.
        self.hdfSpec = specs.HdfArraySpec()

    def tearDown(self):
        super().tearDown()

        self._shutdown_plant()
        shutil.rmtree(self.plantDir)
        # Important for testing some convenience functionality that relies on static fields.
        Plant._clear_global_state()

    def _compute_test_function_hash(self, func: Callable) -> str:
        """
        A simple "hash" function use for source code hashing during unit testing.
        Allows to mark arbitrary reactors as modified.
        """
        name = func.__name__
        if name in self.modifiedReactors:
            PyPlantTest.uniqueReactorHashPostfix += 1
            return name + str(PyPlantTest.uniqueReactorHashPostfix)

        return name

    def _construct_plant(self, config, reactors=None, useTestFuncHash: bool = True):

        def _on_reactor_started(eventType, reactorName, reactorFunc=None):
            if eventType == Plant.EventType.reactor_started:
                self.startedReactors.append(reactorName)
            elif eventType == Plant.EventType.subreactor_started:
                self.startedSubreactors.append(reactorName)

        if useTestFuncHash:
            # Be default use a special hash designed for easier testing.
            self.plant = Plant(self.plantDir, functionHash=self._compute_test_function_hash)
        else:
            self.plant = Plant(self.plantDir)

        self.plant.__enter__()
        self.plant.add_event_listener(Plant.EventType.reactor_started, _on_reactor_started)
        self.plant.add_event_listener(Plant.EventType.subreactor_started, _on_reactor_started)
        self.plant.set_config(config)
        self.plant.warehouse.register_ingredient_specs([self.hdfSpec])
        if reactors is not None:
            self.plant.add_reactors(*tuple(reactors))

        return self.plant

    def _reconstruct_plant(self, config, addReactors: bool = True, useTestFuncHash: bool = True):
        reactors = self.plant.get_reactors()
        self._shutdown_plant()
        self.startedReactors = []
        self.startedSubreactors = []

        self._construct_plant(config, useTestFuncHash=useTestFuncHash)
        if addReactors:
            self.plant.add_reactors(*tuple(reactors))

    def _shutdown_plant(self):
        self.plant.shutdown()

    def test_reactors_basic(self):
        @ReactorFunc
        def reactor_a(pipe: Pipework):
            pipe.send('a', 1, Ingredient.Type.simple)
            yield

        @ReactorFunc
        def reactor_b(pipe: Pipework):
            a = yield pipe.receive('a')
            pipe.send('b', a * 2, Ingredient.Type.simple)

        @ReactorFunc
        def reactor_c(pipe: Pipework):
            b = yield pipe.receive('b')
            self.assertEqual(b, 2)

        self._construct_plant({}, [reactor_a, reactor_b, reactor_c])
        self.plant.run_reactor(reactor_c)
        # All reactors should have been started. Order is not important.
        self.assertSetEqual(set(self.startedReactors), {'reactor_c', 'reactor_b', 'reactor_a'})

        self._reconstruct_plant({})

        self.plant.run_reactor(reactor_c)
        # The first two reactors shouldn't re-run.
        self.assertEqual(self.startedReactors, ['reactor_c'])

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

    def test_file_ingredients(self):

        @ReactorFunc
        def producer(pipe: Pipework):
            filepathA = pipe.allocate('file-a', Ingredient.Type.file)
            filepathB = pipe.allocate('file-b', Ingredient.Type.file)

            with open(filepathA, 'w') as file:
                file.write('test-string-a')
            pipe.send('file-a', filepathA, Ingredient.Type.file)

            with open(filepathB, 'w') as file:
                file.write('test-string-b')
            pipe.send('file-b', filepathB, Ingredient.Type.file)

            yield

        @ReactorFunc
        def consumer(pipe: Pipework):
            filepathA = yield pipe.receive('file-a')
            filepathB = yield pipe.receive('file-b')

            with open(filepathA, 'r') as file:
                self.assertEqual('test-string-a', file.readline())
                self.assertEqual('', file.readline())
            with open(filepathB, 'r') as file:
                self.assertEqual('test-string-b', file.readline())
                self.assertEqual('', file.readline())

        self._construct_plant({}, [producer, consumer])
        self.plant.run_reactor(consumer)

        self._reconstruct_plant({})
        self.plant.run_reactor(consumer)

    def test_config_object(self):

        class Config(ConfigBase):

            def __init__(self):
                # noinspection PyCompatibility
                super().__init__({})

                self.paramA = 1
                self.paramB = 'asd'
                self.paramC = (1, 2)

        @ReactorFunc
        def producer(pipe: Pipework, config: Config):

            paramA = config.paramA
            paramB = config.paramB

            pipe.send('ingredient', 1, Ingredient.Type.simple)

            yield

        @ReactorFunc
        def consumer(pipe: Pipework, config: Config):

            paramC = config.paramC

            ingredient = yield pipe.receive('ingredient')  # type: int
            self.assertEqual(ingredient, 1)

            yield

        config = Config()
        self._construct_plant(config, [producer, consumer])
        self.plant.run_reactor(consumer)

        self._reconstruct_plant(config)
        self.plant.run_reactor(consumer)
        self.assertEqual(self.startedReactors, ['consumer'], msg='The producer should be cached.')

        config.paramC = (2, 3)
        self._reconstruct_plant(config)
        self.plant.run_reactor(consumer)
        self.assertEqual(self.startedReactors, ['consumer'], msg='The producer should be cached.')

        config.paramA = 2
        self._reconstruct_plant(config)
        self.plant.run_reactor(consumer)
        self.assertEqual(self.startedReactors, ['consumer', 'producer'], msg='The producer should be rerun.')

    # noinspection DuplicatedCode
    def test_config_value_is_func(self):
        """
        When using functions as config values (e.g. for code-defined models),
        the plant should hash them using the source code and not their address,
        allowing the dependent reactors to be properly cached.
        """

        import tempfile
        import random
        import string
        import types
        # noinspection PyCompatibility
        import importlib.machinery

        def _load_func_from_file(filepath: str, funcName: str):

            randomString = ''.join(random.choice(string.ascii_uppercase) for _ in range(10))
            loader = importlib.machinery.SourceFileLoader("_custom_module." + randomString, filepath)
            module = types.ModuleType(loader.name)
            loader.exec_module(module)

            return getattr(module, funcName)

        def _load_func_from_code(sourceCode, funcName):
            fd, filepath = None, None
            try:
                fd, filepath = tempfile.mkstemp(suffix='.py')
                os.write(fd, sourceCode.encode('utf-8'))
                os.fsync(fd)
                os.close(fd)

                return _load_func_from_file(filepath, funcName)
            except Exception as e:
                raise
            finally:
                # If we remove the file, pyplant won't be able to laod the source.
                # So we leave it up to OS to clear the temp dir.
                if fd is not None and os.path.exists(filepath):
                    pass
                    # os.remove(filepath)

        class Config(ConfigBase):

            def __init__(self):
                # noinspection PyCompatibility
                super().__init__({})

                self.funcParam = lambda: 1  # We'll fill this in later.

        @ReactorFunc
        def producer(pipe: Pipework, config: Config):

            funcParam = config.funcParam
            result = funcParam()
            print(result)
            print(funcParam)

            pipe.send('ingredient', result, Ingredient.Type.simple)

            yield

        @ReactorFunc
        def consumer(pipe: Pipework, config: Config):

            ingredient = yield pipe.receive('ingredient')  # type: int

            yield

        func1 = _load_func_from_code('def func():\n\treturn "one"\n', 'func')
        func2 = _load_func_from_code('def func():\n\treturn "two"\n', 'func')

        config = Config()
        config.funcParam = func1
        # Disable the custom function hash, use the standard source-based hash function that we want to test.
        self._construct_plant(config, [producer, consumer], useTestFuncHash=False)
        self.plant.run_reactor(consumer)
        self.assertEqual(self.startedReactors, ['consumer', 'producer'], msg='Both should run initially.')

        self._reconstruct_plant(config, useTestFuncHash=False)
        self.plant.run_reactor(consumer)
        self.assertEqual(self.plant.fetch_ingredient('ingredient'), 'one')
        self.assertEqual(self.startedReactors, ['consumer'], msg='The producer should be cached.')

        # Reload the function to change their address, but keep the source unchanged.
        # This shouldn't trigger reactor re-execution.
        func1 = _load_func_from_code('def func():\n\treturn "one"\n', 'func')
        config.funcParam = func1
        self._reconstruct_plant(config, useTestFuncHash=False)
        self.plant.run_reactor(consumer)
        self.assertEqual(self.plant.fetch_ingredient('ingredient'), 'one')
        self.assertEqual(self.startedReactors, ['consumer'], msg='The producer should be cached.')

        # Now give it a different function.
        config.funcParam = func2
        self._reconstruct_plant(config, useTestFuncHash=False)
        self.plant.run_reactor(consumer)
        self.assertEqual(self.plant.fetch_ingredient('ingredient'), 'two')
        self.assertEqual(self.startedReactors, ['consumer', 'producer'], msg='The producer should be rerun.')

    def test_scipy_sparse(self):
        import scipy.sparse as sp
        from pyplant.specs import ScipySparseSpec

        sparseSpec = ScipySparseSpec()
        @ReactorFunc
        def producer(pipe: Pipework):
            eye = sp.eye(12)
            pipe.send('eye', eye, ScipySparseSpec)

        @ReactorFunc
        def consumer(pipe: Pipework):
            eye = yield pipe.receive('eye')

            np.testing.assert_array_equal(eye.todense(), sp.eye(12).todense())

        self._construct_plant({}, [producer, consumer])
        self.plant.warehouse.register_ingredient_specs([sparseSpec])
        self.plant.run_reactor(consumer)

    def test_hdf_arrays(self):

        @ReactorFunc
        def producer(pipe: Pipework):
            array1 = pipe.allocate('array1', specs.HdfArraySpec, shape=(4, 5, 6, 7), dtype=np.uint8)
            array2 = pipe.allocate('array2', specs.HdfArraySpec, shape=(3, 500), dtype=np.bool)

            array1[:, 3, 4, :] = 42
            array2[2, :250] = True

            pipe.send('array1', array1, specs.HdfArraySpec)
            pipe.send('array2', array2, specs.HdfArraySpec)

            yield

        @ReactorFunc
        def consumer(pipe: Pipework):

            array1 = yield pipe.receive('array1')
            array2 = yield pipe.receive('array2')

            self.assertTrue(np.all(np.equal(array1[:, 3, 4, :], 42)))
            self.assertTrue(np.all(np.not_equal(array1[:, :3, :, :], 42)))
            self.assertTrue(np.all(np.not_equal(array1[:, 4:, :, :], 42)))
            self.assertTrue(np.all(np.not_equal(array1[:, :, :4, :], 42)))
            self.assertTrue(np.all(np.not_equal(array1[:, :, 5:, :], 42)))

            self.assertTrue(np.all(np.equal(array2[2, :250], True)))
            self.assertTrue(np.all(np.not_equal(array2[2, 250:], True)))
            self.assertTrue(np.all(np.not_equal(array2[:2, :], True)))

            yield

        self._construct_plant({}, [producer, consumer])
        self.plant.run_reactor(consumer)

        self.assertEqual(self.startedReactors, ['consumer', 'producer'],
                         msg='Both reactors should run initially.')

        self._reconstruct_plant({})
        self.plant.run_reactor(consumer)

        self.assertEqual(self.startedReactors, ['consumer'],
                         msg='The producer should be cached.')

    def test_keras_models(self):
        import keras

        weights = None
        kerasSpec = specs.KerasModelSpec()

        @ReactorFunc
        def producer(pipe: Pipework):
            m = keras.models.Sequential()
            m.add(keras.layers.Dense(100, activation='relu', input_shape=(10,)))

            m.compile(keras.optimizers.SGD(), loss='mse')

            nonlocal weights
            weights = m.get_weights()[0]

            pipe.send('model', m, specs.KerasModelSpec)

        @ReactorFunc
        def consumer(pipe: Pipework):
            m = yield pipe.receive('model')  # type: keras.models.Sequential

            loadedWeights = m.get_weights()[0]
            np.testing.assert_array_equal(loadedWeights, weights)

        self._construct_plant({}, [producer, consumer])
        self.plant.warehouse.register_ingredient_specs([kerasSpec])
        self.plant.run_reactor(consumer)

        self._reconstruct_plant({})
        self.plant.warehouse.register_ingredient_specs([kerasSpec])
        self.plant.run_reactor(consumer)

    def test_custom_ingredient_spec(self):
        spec = CustomSpec()

        @ReactorFunc
        def producer(pipe: Pipework):
            val = CustomSpec.ValueWrapper(15)
            pipe.send('val', val, CustomSpec)

        @ReactorFunc
        def consumer(pipe: Pipework):
            val = yield pipe.receive('val')  # type: CustomSpec.ValueWrapper

            self.assertEqual(val.val, 15)

        self._construct_plant({}, [producer, consumer])
        self.plant.warehouse.register_ingredient_specs([spec])
        self.plant.run_reactor(consumer)

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

    def test_fetching_own_products(self):
        config = {'param-a': 1, 'param-b': 2}

        reactorBResult = None

        @SubreactorFunc
        def subreactor_a(pipe: Pipework):
            paramA = pipe.read_config('param-a')
            a = paramA + 10  # Do stuff.

            pipe.send('subreactor-a-product', a, Ingredient.Type.simple)
            yield

        @SubreactorFunc
        def subreactor_b(pipe: Pipework):
            aProduct = yield pipe.receive('subreactor-a-product')
            b = aProduct + 15
            pipe.send('subreactor-b-product', b, Ingredient.Type.simple)

        @ReactorFunc
        def reactor_a(pipe: Pipework):
            yield from subreactor_a(pipe)
            yield from subreactor_b(pipe)

        @ReactorFunc
        def reactor_b(pipe: Pipework):
            nonlocal reactorBResult
            reactorBResult = yield pipe.receive('subreactor-b-product')

        # First pass, everything is executed, reactor B depends on A through a subreactor.
        self._construct_plant(config, [reactor_a, reactor_b])
        self.plant.run_reactor('reactor_b')

        self.assertEqual(self.startedSubreactors, ['subreactor_a', 'subreactor_b'])
        self.assertEqual(reactorBResult, 1 + 10 + 15)

        # Test that the results are cached.
        self._reconstruct_plant(config)
        self.plant.run_reactor(reactor_b)

        self.assertEqual(self.startedReactors, ['reactor_b'])
        self.assertEqual(self.startedSubreactors, [])
        self.assertEqual(reactorBResult, 1 + 10 + 15)

        # Modifying a parameter used by a subreactor causes re-execution.
        config['param-a'] = 5
        self._reconstruct_plant(config)
        self.plant.run_reactor(reactor_b)

        self.assertEqual(self.startedReactors, ['reactor_b', 'reactor_a'])
        self.assertEqual(self.startedSubreactors, ['subreactor_a', 'subreactor_b'])
        self.assertEqual(reactorBResult, 5 + 10 + 15)

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

    def test_reactor_rerun_on_crash(self):

        isFirstRun = True

        @ReactorFunc
        def reactor_a(pipe: Pipework):
            pipe.send('a', 1, Ingredient.Type.simple)
            yield

        @ReactorFunc
        def reactor_b(pipe: Pipework):
            a = yield pipe.receive('a')

            nonlocal isFirstRun
            if isFirstRun:
                isFirstRun = False
                raise RuntimeError("Testing crash behavior.")

            pipe.send('b', 2, Ingredient.Type.simple)

        @ReactorFunc
        def reactor_c(pipe: Pipework):
            b = yield pipe.receive('b')
            yield

        self._construct_plant({}, [reactor_a, reactor_b, reactor_c])
        with self.assertRaises(RuntimeError):
            self.plant.run_reactor(reactor_c)

        # All reactors should have been started. Order is not important.
        self.assertSetEqual(set(self.startedReactors), {'reactor_c', 'reactor_b', 'reactor_a'})

        self._reconstruct_plant({})

        self.plant.run_reactor(reactor_c)
        # The plant should finish successfully now, 'reactor_a' should've not re-ran.
        self.assertEqual(self.startedReactors, ['reactor_c', 'reactor_b'])

    def test_continue_after_wrong_producer_guess(self):
        """
        This test covers a specific bug, where a producer become a consumer of that ingredient
        would cause the plant to loop indefinitely, because it would not update metadata correctly.
        """

        @ReactorFunc
        def reactor_a(pipe: Pipework):
            pipe.send('a', 1, Ingredient.Type.simple)
            yield

        @ReactorFunc
        def reactor_b(pipe: Pipework):
            a = yield pipe.receive('a')

        # Use the standard hash, we're going to modify reactors manually by re-defining them.
        self._construct_plant({}, [reactor_a, reactor_b], useTestFuncHash=False)
        self.plant.run_reactor(reactor_b)
        self.assertEqual(self.startedReactors, ['reactor_b', 'reactor_a'])

        # Now, we swap the producer and consumer. The plant should update its metadata correctly.
        @ReactorFunc
        def reactor_a(pipe: Pipework):
            a = yield pipe.receive('a')

        @ReactorFunc
        def reactor_b(pipe: Pipework):
            pipe.send('a', 1, Ingredient.Type.simple)

        self._shutdown_plant()
        self._construct_plant({}, [reactor_a, reactor_b])
        self.startedReactors = []
        self.plant.run_reactor(reactor_a)

        # Both reactors should run without crashes.
        self.assertEqual(self.startedReactors, ['reactor_a', 'reactor_b'])

    def test_non_generator_reactor(self):

        nonGenRun = False
        genRun = False

        @ReactorFunc
        def non_generator(pipe: Pipework, config):
            nonlocal nonGenRun
            nonGenRun = True

            pipe.send('x', 5, Ingredient.Type.simple)

        @ReactorFunc
        def generator(pipe: Pipework, config):
            x = yield pipe.receive('x')
            self.assertEqual(x, 5)

            nonlocal genRun
            genRun = True

        self._construct_plant(ConfigBase(), [non_generator, generator])
        self.plant.run_reactor(generator)

        self.assertTrue(nonGenRun)
        self.assertTrue(genRun)

    def test_non_generator_subreactor(self):

        nonGenRun = False

        @SubreactorFunc
        def non_generator(arg1):
            nonlocal nonGenRun
            nonGenRun = True

            return arg1 * 2

        @ReactorFunc
        def reactor(pipe: Pipework, config):
            x = yield from non_generator(10)
            self.assertEqual(x, 20)

        self._construct_plant(ConfigBase(), [reactor])
        self.plant.run_reactor(reactor)

        self.assertTrue(nonGenRun)

    def test_reactors_added_automatically(self):

        @ReactorFunc
        def reactor_a(pipe: Pipework):
            pipe.send('a', 1, Ingredient.Type.simple)
            yield

        @ReactorFunc
        def reactor_b(pipe: Pipework):
            a = yield pipe.receive('a')
            pipe.send('b', a * 2, Ingredient.Type.simple)

        @ReactorFunc
        def reactor_c(pipe: Pipework):
            b = yield pipe.receive('b')
            self.assertEqual(b, 2)

        self._construct_plant({})
        self.plant.run_reactor(reactor_c)
        # All reactors should have been started. Order is not important.
        self.assertSetEqual(set(self.startedReactors), {'reactor_c', 'reactor_b', 'reactor_a'})

        self._reconstruct_plant({}, addReactors=False)

        self.plant.run_reactor(reactor_c)
        # The first two reactors shouldn't re-run.
        self.assertEqual(self.startedReactors, ['reactor_c'])

    def test_ingredient_type_inference(self):
        import keras
        import scipy.sparse as sp

        modelWeights = None  # type: Optional[np.ndarray]

        @ReactorFunc
        def producer(pipe: Pipework, config):
            # Send various ingredients without specifying their type.

            pipe.send('simple', 5)
            pipe.send('list', ['a', 'b', 'c'])
            pipe.send('array', np.arange(0, 100))

            hdfArray = pipe.allocate('hdf-array', specs.HdfArraySpec, shape=(100,), dtype=np.float32)
            hdfArray[...] = np.arange(0, 100)
            pipe.send('hdf-array', hdfArray)

            obj = DummyClass()
            obj.__dict__ = {'a': 1, 'b': 2}
            pipe.send('object', obj)

            model = keras.models.Sequential()
            model.add(keras.layers.Dense(100, activation='relu', input_shape=(10,)))
            model.compile(keras.optimizers.SGD(), loss='mse')
            nonlocal modelWeights
            modelWeights = model.get_weights()[0]
            pipe.send('keras-model', model)

            sparse = sp.eye(12)
            pipe.send('sparse', sparse)

        @ReactorFunc
        def consumer(pipe: Pipework, config):
            simple = yield pipe.receive('simple')
            self.assertEqual(simple, 5)

            lst = yield pipe.receive('list')
            self.assertEqual(lst, ['a', 'b', 'c'])

            array = yield pipe.receive('array')
            np.testing.assert_array_equal(array, np.arange(0, 100))

            hdfArray = yield pipe.receive('hdf-array')
            np.testing.assert_array_equal(hdfArray[...], np.arange(0, 100))

            obj = yield pipe.receive('object')
            self.assertTrue(isinstance(obj, DummyClass))
            self.assertEqual(obj.__dict__, {'a': 1, 'b': 2})

            model = yield pipe.receive('keras-model')  # type: keras.models.Model
            np.testing.assert_array_equal(model.get_weights()[0], modelWeights)

            sparse = yield pipe.receive('sparse')  # type: sp.spmatrix
            np.testing.assert_array_equal(sparse.todense(), sp.eye(12).todense())

        self._construct_plant(ConfigBase(), [producer, consumer])
        self.plant.run_reactor(consumer)

        pass


class PyPlantWarehouseTest(unittest.TestCase):

    def setUp(self):
        super().setUp()

        self.dir = tempfile.mkdtemp()

        self.ingredientsToTest = {
            'simple': (Ingredient.Type.simple, 'Value1'),
            'object': (Ingredient.Type.object, slice(1, 10, 2)),
            'array': (Ingredient.Type.array, np.ones(10, dtype=np.int32)),
            'huge_array': (specs.HdfArraySpec.get_name(), np.ones(10, dtype=np.int32)),
        }

        self.warehouse = pyplant.Warehouse(self.dir, logger=logging.getLogger('temp'))
        self.hdfSpec = specs.HdfArraySpec()
        self.warehouse.register_ingredient_specs([self.hdfSpec])

        for name, (type, value) in self.ingredientsToTest.items():
            ingredient = Ingredient(name)
            ingredient.type = type
            ingredient.set_current_signature('initialSignature')

            if ingredient.type == specs.HdfArraySpec.get_name():
                value = self.warehouse.allocate(ingredient, shape=(10,), dtype=np.int32, data=value)

            self.warehouse.store(ingredient, value)

    def tearDown(self):
        super().tearDown()

        self.warehouse.close()
        shutil.rmtree(self.dir)

    def _assert_equal(self, type: Ingredient.Type, valA, valB, msg=None):
        if type == Ingredient.Type.simple or type == Ingredient.Type.object:
            self.assertEqual(valA, valB, msg=msg)
        elif type == Ingredient.Type.array or type == specs.HdfArraySpec.get_name():
            self.assertTrue(np.all(np.equal(valA[...], valB[...])))
        else:
            raise RuntimeError("Unsupported ingredient type: '{}'".format(type))

    def test_warehouse_fetches(self):

        for name, (type, value) in self.ingredientsToTest.items():
            fetchedValue = self.warehouse.fetch(name, 'initialSignature')
            self._assert_equal(type, value, fetchedValue)
    
    # def test_warehouse_prunes_on_fetch(self):
    #
    #     for name, (type, value) in self.ingredientsToTest.items():
    #         fetchedValue = self.warehouse.fetch(name, 'changedSignature')
    #         self.assertIsNone(fetchedValue, msg="Signature changed, shouldn't return anything")
    #
    #     for name, (type, value) in self.ingredientsToTest.items():
    #         fetchedValue = self.warehouse.fetch(name, 'initialSignature')
    #         self.assertIsNone(fetchedValue, msg="Should prune the old value.")

    def test_warehouse_prunes_on_store(self):

        # Restart the warehouse, so we aren't overwriting ingredients within the same session (this is not allowed).
        self.warehouse.close()
        self.warehouse = pyplant.Warehouse(self.dir, logging.getLogger('temp'))
        self.warehouse.register_ingredient_specs([specs.HdfArraySpec()])

        for name, (type, value) in self.ingredientsToTest.items():
            ingredient = Ingredient(name)
            ingredient.type = type
            ingredient.set_current_signature('changedSignature')

            if ingredient.type == specs.HdfArraySpec.get_name():
                value = self.warehouse.allocate(ingredient, shape=(10,), dtype=np.int32, data=value)

            self.warehouse.store(ingredient, value)

        for name, (type, value) in self.ingredientsToTest.items():
            fetchedValue = self.warehouse.fetch(name, 'initialSignature')
            self.assertIsNone(fetchedValue, msg="New value stored, should prune the old one.")

    def test_overwriting_raises_exception(self):

        for name, (type, value) in self.ingredientsToTest.items():
            ingredient = Ingredient(name)
            ingredient.type = type
            ingredient.set_current_signature('changedSignature')

            if ingredient.type == specs.HdfArraySpec.get_name():
                value = self.warehouse.allocate(ingredient, shape=(10, 10), dtype=np.int32)

            with self.assertRaises(RuntimeError):
                self.warehouse.store(ingredient, value)

    def test_graceful_overwrite_on_corrupted_hdf_files(self):
        # Store a new huge array ingredient.
        ingredient = Ingredient('test-dataset')
        ingredient.type = specs.HdfArraySpec.get_name()

        dataset = self.warehouse.allocate(ingredient, shape=(300, 20, 20), dtype=np.float32)
        dataset[100:200] = 66
        self.warehouse.store(ingredient, dataset)

        self.warehouse.close()
        hdfFilePath = self.hdfSpec._get_hdf_array_filepath(self.warehouse, 'test-dataset')
        with open(hdfFilePath, mode='wb') as file:
            file.write(bytearray([255] * 300))

        self.warehouse = pyplant.Warehouse(self.dir, logger=logging.getLogger('temp'))
        self.warehouse.register_ingredient_specs([specs.HdfArraySpec()])
        dataset = self.warehouse.fetch('test-dataset', None)

        self.assertIsNone(dataset)
        self.assertFalse(os.path.exists(hdfFilePath))

    def test_corrupted_pickle_store_doesnt_crash(self):
        # Store a basic ingredient.
        ingredient = Ingredient('test')
        ingredient.type = Ingredient.Type.simple

        self.warehouse.store(ingredient, 42)
        self.warehouse.close()

        # Mess up the file store.
        with open(self.warehouse.simpleStorePath, 'w+') as file:
            file.write('nonsense')

        # This shouldn't throw an exception and the ingredient should be gone.
        self.warehouse = pyplant.Warehouse(self.dir, logger=logging.getLogger('temp'))
        self.assertIsNone(self.warehouse.fetch('test'))
