import os
import pickle
import inspect
import hashlib
from typing import Callable, Optional, Any, Dict, Generator, Union
from enum import Enum
from types import SimpleNamespace

import numpy as np
import h5py

__all__ = ['Plant', 'ReactorFunc', 'Pipework', 'Ingredient']


def ReactorFunc(func):
    assert('pyplant' not in func.__dict__)

    func.pyplant = SimpleNamespace()
    func.pyplant.isReactor = True

    return func


class Plant:

    class RunningReactor:

        def __init__(self, name: str, reactorObject: 'Reactor', generator: object):

            self.name = name
            self.reactorObject = reactorObject
            self.generator = generator  # type: Generator[Any, Any]
            self.awaitedIngredient = None  # type: str

    class IngredientAwaitedCommand:

        def __init__(self, ingredientName: str):
            self.ingredientName = ingredientName

    def __init__(self, plantDir: str):

        if not os.path.exists(plantDir):
            os.makedirs(plantDir)

        self.plantDir = plantDir
        self.reactors = {}
        self.runningReactors = {}  # type: Dict[str, Plant.RunningReactor]
        self.ingredients = {}  # type: Dict[str, Ingredient]
        self.warehouse = Warehouse(plantDir)
        self.pipeworks = {}
        self.config = {}

        plantPath = os.path.join(self.plantDir, 'plant.pcl')
        if os.path.exists(plantPath):
            with open(plantPath, 'rb') as file:
                plantCache = pickle.load(file)
                self.reactors = plantCache['reactors']
                self.ingredients = plantCache['ingredients']

            for ingredient in self.ingredients.values():
                ingredient.on_cache_load()

            for reactor in self.reactors.values():
                reactor.on_cache_load()

    def shutdown(self):
        print("Shutting down the plant.")

        with open(os.path.join(self.plantDir, 'plant.pcl'), 'wb') as file:
            pickle.dump({
                'reactors': self.reactors,
                'ingredients': self.ingredients
            }, file)

        self.warehouse.close()

    def add_reactors(self, *args):
        for func in args:
            assert(callable(func))

            if 'pyplant' not in func.__dict__:
                raise RuntimeError("{} is not a @ReactorFunc!".format(func))

            name = func.__name__
            reactor = Reactor(name, func)

            print("Adding reactor '{}'.".format(name))

            if name in self.reactors and reactor.get_signature() == self.reactors[name].get_signature():
                print("Reactor is already known. Storing the function.")
                self.reactors[name].func = func
            else:
                self.reactors[name] = reactor

    def is_ingredient_known(self, name: str):
        return name in self.ingredients

    def run_reactor(self, reactorFunc: Callable):
        assert(callable(reactorFunc))
        reactor = self.reactors[reactorFunc.__name__]

        assert(len(self.runningReactors) == 0)
        # Schedule the reactor for running.
        self._start_reactor(reactor)
        # Execute.
        self._finish_all_running_reactors()

    def _compute_ingredient_signature(self, ingredientName):
        print("Computing signature for ingredient '{}'.".format(ingredientName))
        ingredient = self._get_ingredient(ingredientName)
        if ingredient is not None and ingredient.isSignatureFresh:
            print("Found the signature in cache: '{}'".format(ingredient.signature))
            return ingredient.signature

        reactor = self._get_producing_reactor(ingredientName)

        # If we don't know the producing reactor, or it has changed and wasn't run yet,
        # we cannot compute the signature.
        if reactor is None or (not reactor.wasRun and not self._is_reactor_running(reactor.name)):
            print("Producing reactor is unknown, signature is unknown.")
            return None

        print("Collecting sub-ingredient signatures...")
        subingredientSignatures = []
        for subingredientName in sorted(reactor.get_inputs()):
            subsignature = self._compute_ingredient_signature(subingredientName)
            if subsignature is None:
                print("Signature for '{}' is unknown.".format(ingredientName))
                return None
            subingredientSignatures.append(subsignature)

        print("Sub-ingredients' signatures found.")

        # Compute the signature as a hash of subingredients' signatures, config parameters
        # and reactor's signature (code).
        subingredientSignature = hashlib.sha1(''.join(subingredientSignatures).encode('utf-8')).hexdigest()

        parameterStrings = ['{}_{}'.format(name, self.config[name]) for name in sorted(reactor.get_params())]
        parameterSignature = hashlib.sha1(''.join(parameterStrings).encode('utf-8')).hexdigest()

        signatureParts = [reactor.get_signature(), subingredientSignature, parameterSignature]
        fullSignature = hashlib.sha1(''.join(signatureParts).encode('utf-8')).hexdigest()

        ingredient = self._get_or_create_ingredient(ingredientName)
        ingredient.set_current_signature(fullSignature)

        print("Computed signature for '{}': '{}'".format(ingredient.name, ingredient.signature))
        return fullSignature

    def _try_produce_ingredient(self, ingredientName):
        """
        Tries to produce an ingredient by checking which reactor used to
        produce an ingredient with this name.
        (An educated guess, better than running all new reactors.)

        :param ingredientName:
        :return:
        """
        print("Trying to produce ingredient '{}'.".format(ingredientName))

        self._get_or_create_ingredient(ingredientName)
        reactor = self._get_producing_reactor(ingredientName)
        if reactor is not None:
            print("Found the producing reactor '{}', scheduling.".format(reactor.name))
            self._start_reactor(reactor)
        else:
            print("Do not know how to produce '{}'.".format(ingredientName))

    def _start_reactor(self, reactorObject: 'Reactor') -> 'RunningReactor':
        print("Starting reactor '{}'.".format(reactorObject.name))
        if not reactorObject.wasRun:
            reactorObject.reset_metadata()

        # Each reactor needs a pipework to send and receive data.
        pipework = self._get_or_create_pipework(reactorObject)

        # Reactors are generator functions. Obtain a generator object.
        generator = reactorObject.func(pipework)
        if not inspect.isgenerator(generator):
            raise RuntimeError("Reactor '{}' is not a generator function!".format(reactorObject.name))

        # Schedule execution.
        runningReactor = Plant.RunningReactor(reactorObject.name, reactorObject, generator)
        self.runningReactors[reactorObject.name] = runningReactor

        return runningReactor

    def _finish_all_running_reactors(self):
        """
        Main function governing reactor execution.
        Maintains a queue of running reactors and ingredients that they're waiting for.
        Runs additional reactors as necessary to produce missing ingredients.
        :return:
        """
        print("Trying to finish all running reactors.")

        # Main loop over the running reactor queue.
        while len(self.runningReactors) != 0:
            nextReactor = None  # type: Plant.RunningReactor
            # Check if we have a paused reactor waiting for an ingredient that was just produced,
            # or a reactor without any dependencies (just scheduled).
            for runningReactor in self.runningReactors.values():
                awaitedIngredient = runningReactor.awaitedIngredient
                if awaitedIngredient is None or self._is_ingredient_fresh(awaitedIngredient):
                    nextReactor = runningReactor
                    break

            # If no reactors are ready to run, look for missing ingredients by scheduling
            # new/modified reactors to run.
            if nextReactor is None:
                print("Some ingredients are missing. Running new reactors...")
                for reactor in self.reactors.values():
                    if not reactor.wasRun and not self._is_reactor_running(reactor.name):
                        nextReactor = self._start_reactor(reactor)
                        break

            if nextReactor is None:
                debugDump = "Running reactors: {} \n".format(self.runningReactors.keys())
                missingIngredients = [r.awaitedIngredient for r in self.runningReactors.values()
                                      if r.awaitedIngredient is not None]
                debugDump += "Missing ingredients: {} \n".format(missingIngredients)
                print(">>> Plant state dump: \n" + debugDump)

                raise RuntimeError("Could not find a reactor that should run next. Deadlock?")

            # We have now figured out which reactor to update next.
            print("Next reactor to run: '{}'".format(nextReactor.name))

            valueToSend = None
            awaitedIngredient = nextReactor.awaitedIngredient
            # If the reactor is waiting for an ingredient, fetch and send it.
            if awaitedIngredient is not None:
                print("Providing reactor '{}' with ingredient '{}'".format(nextReactor.name, awaitedIngredient))
                valueToSend = self.warehouse.fetch(awaitedIngredient)

            # A reactor is a generator function that yields either fetched ingredients,
            # or 'IngredientAwaited' commands, and should be sent the ingredient.
            # Keep iterating the generator until it either terminates, or requests a missing ingredient.
            missingIngredient = None
            while missingIngredient is None:
                try:
                    returnedObject = nextReactor.generator.send(valueToSend)
                    if type(returnedObject) is Plant.IngredientAwaitedCommand:
                        # A missing ingredient is was requested, will pause the reactor.
                        nextReactor.awaitedIngredient = returnedObject.ingredientName
                        missingIngredient = returnedObject.ingredientName
                    else:
                        # A reactor successfully fetched an ingredient, no need to pause, just use it.
                        valueToSend = returnedObject

                except StopIteration:
                    print("Reactor '{}' has finished running.".format(nextReactor.name))
                    del self.runningReactors[nextReactor.name]
                    nextReactor.reactorObject.wasRun = True
                    break

            if missingIngredient is not None:
                # A reactor paused due to a missing ingredient, try to produce it using previous knowledge.
                print("Will try to produce '{}' for reactor '{}'.".format(missingIngredient, nextReactor.name))
                self._try_produce_ingredient(missingIngredient)

        print("Finished running all reactors.")

    def _get_producing_reactor(self, ingredientName: str) -> Union['Reactor', None]:
        if ingredientName not in self.ingredients:
            return None
        producerName = self.ingredients[ingredientName].producerName
        return self.reactors[producerName] if producerName is not None else None

    def _get_ingredient(self, name: str) -> Union['Ingredient', None]:
        matchingIngredients = (i for n, i in self.ingredients.items() if n == name)
        return next(matchingIngredients, None)

    def _get_or_create_ingredient(self, name: str) -> 'Ingredient':
        ingredient = self._get_ingredient(name)
        if ingredient is None:
            ingredient = Ingredient(name)
            self.ingredients[name] = ingredient

        return ingredient

    def _is_ingredient_fresh(self, name: str):
        return name in self.ingredients and self.ingredients[name].isFresh

    def _is_reactor_running(self, name: str):
        return name in self.runningReactors

    def _get_or_create_pipework(self, reactor: 'Reactor') -> 'Pipework':
        if reactor.name not in self.pipeworks:
            print("Creating pipework for reactor '{}'.".format(reactor.name))
            self.pipeworks[reactor.name] = Pipework(self, self.warehouse, reactor)

        return self.pipeworks[reactor.name]

    def set_config(self, configMap):
        self.config = configMap

    def get_config_param(self, name: str) -> Any:
        if name in self.config:
            return self.config[name]

        raise RuntimeError("Unknown config parameter: '{}'".format(name))


class Pipework:
    """
    Used by reactors to send and receive ingredients.
    A unique instance is attached to each reactor,
    """

    def __init__(self, plant: Plant, warehouse: 'Warehouse', connectedReactor: 'Reactor'):
        self.plant = plant
        self.warehouse = warehouse
        self.connectedReactor = connectedReactor  # Which reactor this pipework is connected to.

    def receive(self, name) -> Any:
        print("Reactor '{}' is requesting ingredient '{}'".format(self.connectedReactor.name, name))
        self.connectedReactor.register_input(name)

        signature = self.plant._compute_ingredient_signature(name)
        if signature is None:
            return Plant.IngredientAwaitedCommand(name)

        ingredientValue = self.warehouse.fetch(name, signature)
        if ingredientValue is None:
            return Plant.IngredientAwaitedCommand(name)

        return ingredientValue

    def send(self, name: str, value: Any, type: 'Ingredient.Type'):
        print("Reactor '{}' is sending ingredient '{}'".format(self.connectedReactor.name, name))

        ingredient = self._register_output(name, type)
        self.warehouse.store(ingredient, value)

    def allocate(self, name: str, type: 'Ingredient.Type', **kwargs):
        print("Reactor '{}' is allocating ingredient '{}'".format(self.connectedReactor.name, name))

        ingredient = self._register_output(name, type)
        return self.warehouse.allocate(ingredient, **kwargs)

    def read_config(self, paramName: str) -> Any:
        self.connectedReactor.register_parameter(paramName)

        return self.plant.get_config_param(paramName)

    def _register_output(self, name, type):
        ingredient = self.plant._get_or_create_ingredient(name)
        ingredient.type = type
        ingredient.producerName = self.connectedReactor.name
        ingredient.isFresh = True

        self.connectedReactor.register_output(name)

        # Now that we have registered the ingredient, we can compute it's current signature.
        self.plant._compute_ingredient_signature(name)

        assert(ingredient.signature is not None)

        return ingredient


class Reactor:

    def __init__(self, name: str, func: Callable[[Pipework], None]):
        self.func = func
        self.generator = None
        self.name = name
        self.wasRun = False
        self.inputs = set({})
        self.outputs = set({})
        self.params = set({})

        sourceLines = inspect.getsourcelines(func)
        functionSource = ''.join(sourceLines[0])

        self.signature = hashlib.sha1(functionSource.encode('utf-8')).hexdigest()

    def get_signature(self):
        return self.signature

    def get_inputs(self):
        return self.inputs

    def get_params(self):
        return self.params

    def register_parameter(self, name):
        if 'params' not in self.__dict__:
            self.params = {name}
        else:
            self.params.add(name)

    def register_input(self, name):
        if 'inputs' not in self.__dict__:
            self.inputs = {name}
        else:
            self.inputs.add(name)

    def register_output(self, name):
        if 'outputs' not in self.__dict__:
            self.outputs = {name}
        else:
            self.outputs.add(name)

    def reset_metadata(self):
        self.inputs = set()
        self.params = set()

    def on_cache_load(self):
        self.func = None
        self.generator = None


class Ingredient:

    class Type(Enum):
        unknown = 0,
        simple = 1,
        list = 2,
        array = 3,
        huge_array = 4

    def __init__(self, name: str):
        self.name = name
        self.signature = None
        self.isSignatureFresh = False
        self.isFresh = False  # Whether has been produced during the current plant run (not loaded from disk).
        self.type = Ingredient.Type.unknown
        self.producerName = None

    def set_current_signature(self, signature):
        self.signature = signature
        self.isSignatureFresh = True

    def on_cache_load(self):
        self.isSignatureFresh = False
        self.isFresh = False


class Warehouse:

    def __init__(self, baseDir):
        self.baseDir = baseDir
        self.cache = {}
        self.h5File = h5py.File(os.path.join(self.baseDir, 'warehouse.h5py'), mode='a')

        manifestPath = os.path.join(os.path.join(self.baseDir, 'manifest.pcl'))
        if os.path.exists(manifestPath):
            with open(manifestPath, 'rb') as file:
                self.manifest = pickle.load(file)
        else:
            self.manifest = {}

    def fetch(self, name: str, signature: str = None) -> Any:
        """
        Fetch a stored ingredient with a given signature.
        If no signature is specified, the latest version of ingredient is returned.
        :param name:
        :param signature:
        :return:
        """
        print("Fetching ingredient '{}' from the warehouse.".format(name))

        if name not in self.manifest:
            print("Ingredient is not in the warehouse.")
            return None
        elif signature is not None and signature != self.manifest[name]['signature']:
            # The stored ingredient is outdated (Right now we only store a single version of an ingredient).
            print("Ingredient is outdated. Pruning from the warehouse.")
            self._prune(name)
            return None

        if name in self.cache:
            print("Found the ingredient in the cache.")
            return self.cache[name]

        print("Fetching the ingredient from disk.")
        type = self.manifest[name]['type']
        if type == Ingredient.Type.simple:
            return self._fetch_simple(name)
        elif type == Ingredient.Type.list:
            return self._fetch_list(name)
        elif type == Ingredient.Type.array:
            return self._fetch_array(name)
        elif type == Ingredient.Type.huge_array:
            return self._fetch_huge_array(name)

        raise RuntimeError("This should never happen! Unsupported ingredient type: {}".format(type))

    def store(self, ingredient: Ingredient, value: Any):
        print("Storing ingredient '{}' in the warehouse.".format(ingredient.name))
        if ingredient.name in self.manifest:
            print("Ingredient is already in the warehouse, pruning.")
            self._prune(ingredient.name)

        if ingredient.type == Ingredient.Type.simple:
            self._store_simple(ingredient.name, value)
        elif ingredient.type == Ingredient.Type.list:
            self._store_list(ingredient.name, value)
        elif ingredient.type == Ingredient.Type.array:
            self._store_array(ingredient.name, value)
        elif ingredient.type == Ingredient.Type.huge_array:
            self._store_huge_array(ingredient.name, value)
        else:
            raise RuntimeError("Unsupported ingredient type: {}".format(ingredient.type))

        self.manifest[ingredient.name] = {
            'signature': ingredient.signature,
            'type': ingredient.type
        }

        self.cache[ingredient.name] = value

    def allocate(self, ingredient: Ingredient, **kwargs):
        print("Allocating storage for ingredient '{}' in the warehouse.".format(ingredient.name))
        if ingredient.type != Ingredient.Type.huge_array:
            raise RuntimeError("Allocation is not supported for an ingredient of type {}".format(ingredient.type))

        return self._allocate_huge_array(ingredient.name, **kwargs)

    def _prune(self, name):
        pass

    def _store_simple(self, name, value):
        self.h5File.attrs[name] = value

    def _fetch_simple(self, name):
        if name in self.h5File.attrs:
            return self.h5File.attrs[name]
        return None

    def _store_list(self, name, value):
        pickle.dump(os.path.join(self.baseDir, '{}.pck'.format(name)), value)

    def _fetch_list(self, name):
        path = os.path.join(self.baseDir, '{}.pck'.format(name))
        if os.path.exists(path):
            with open(path, 'rb') as file:
                return pickle.load(file)

        return None

    def _store_array(self, name, value: np.ndarray):
        np.save(os.path.join(self.baseDir, '{}.npy'.format(name)), value, allow_pickle=False)

    def _fetch_array(self, name):
        return np.load(os.path.join(self.baseDir, '{}.npy'.format(name)), allow_pickle=False)

    def _allocate_huge_array(self, name, shape, dtype=np.float):
        dataset = self._fetch_huge_array(name)
        if dataset is None:
            dataset = self.h5File.create_dataset(name, shape=shape, dtype=dtype)

        return dataset

    def _store_huge_array(self, name, value: h5py.Dataset):
        pass  # H5py takes care of storing to disk on-the-fly.

    def _fetch_huge_array(self, name):
        if name in self.h5File:
            return self.h5File[name]

        return None

    def close(self):
        self.h5File.close()

        manifestPath = os.path.join(self.baseDir, 'manifest.pcl')
        with open(manifestPath, 'wb') as file:
            pickle.dump(self.manifest, file)
