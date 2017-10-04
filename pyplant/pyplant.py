import os
import sys
import pickle
import inspect
import hashlib
import logging
from typing import Callable, Any, Dict, Generator, Union, List
from enum import Enum
from types import SimpleNamespace

import numpy as np
import h5py
import keras.models
import time

__all__ = ['Plant', 'ReactorFunc', 'SubreactorFunc', 'Pipework', 'Ingredient']


# Random documentation:
#
# Reactors are units of execution, whose results can be cached.
# Subreactors are effectively 'plugged-in' into reactors, and execute code on their behalf.
# The purpose of subreactors is to allow reactors to depend on subroutines, such that a change
# to the code of a subroutine (subreactor) could be detected.
# If the subroutine is not likely to change (e.g. a library), their is no benefit in using a subreactor.
#

def ReactorFunc(func):
    assert('pyplant' not in func.__dict__)

    func.pyplant = SimpleNamespace()
    func.pyplant.isReactor = True

    if not inspect.isgeneratorfunction(func):
        raise RuntimeError("ReactorFunc '{}' is not a generator function!".format(func.__name__))

    return func


def SubreactorFunc(func):
    """
    Marks a 'subreactor' function which will be called as a sub-procedure from reactors.
    Unlike with normal function calls, changed to a subreactor will trigger re-execution
    of all dependent reactors.

    Subreactors must be called as: returnValue = yield from sub_reactor(pipe)
    Just like reactors, to request ingredients a subreactor should use 'yield' keyword.
    To return a value to the caller a subreactor function should use 'return'.

    All received/sent ingredients are attributed to the calling reactor. This means
    that a change to an ingredient/code will trigger re-execution of the whole reactor,
    not just of a subreactor.

    :param func:
    :return:
    """
    assert ('pyplant' not in func.__dict__)

    func.pyplant = SimpleNamespace()
    func.pyplant.isReactor = False
    func.pyplant.isSubreactor = True

    name = func.__name__

    if not inspect.isgeneratorfunction(func):
        raise RuntimeError("SubreactorFunc '{}' is not a generator function!".format(name))

    # Store the current source hash globally. Will be needed for reactor signature computation.
    hashString = _compute_function_hash(func)
    Plant.SubreactorSourceHashes[name] = hashString

    def _pyplant_subreactor_wrapper(*args, **kwargs):
        # Notify the plant, that a subreactor is being called.
        yield Plant.SubreactorStartedCommand(name)
        # Delegate to the  subreactor, as if the reactor was running/yielding.
        result = yield from func(*args, **kwargs)
        # Propagate the returned value to the caller
        return result

    return _pyplant_subreactor_wrapper


class Plant:

    # Global store of current sub-reactor source hashes.
    # Filled out during code importing through function augmentors.
    SubreactorSourceHashes = {}  # type: Dict[str, str]

    class RunningReactor:

        def __init__(self, name: str, reactorObject: 'Reactor', generator: object=None):

            self.name = name
            self.reactorObject = reactorObject
            self.generator = generator  # type: Generator[Any, Any]
            self.awaitedIngredient = None  # type: str
            self.subreactorCallStack = []  # type: List[str]

            self.totalRunTime = 0

    class IngredientAwaitedCommand:

        def __init__(self, ingredientName: str):
            self.ingredientName = ingredientName

    class SubreactorStartedCommand:

        def __init__(self, subreactorName: Callable):
            self.subreactorName = subreactorName

    def __init__(self, plantDir: str, logger: logging.Logger=None, logLevel: int=None):

        if not os.path.exists(plantDir):
            os.makedirs(plantDir)

        self.plantDir = plantDir
        self.logger = logger or logging.getLogger('pyplant')
        self.reactors = {}  # type: Dict[str, Reactor]
        self.subreactors = {}  # type: Dict[str, Subreactor]
        self.runningReactors = {}  # type: Dict[str, Plant.RunningReactor]
        self.ingredients = {}  # type: Dict[str, Ingredient]
        self.warehouse = Warehouse(plantDir, self.logger)
        self.pipeworks = {}
        self.config = {}
        # Marks which config params are 'auxiliary' and shouldn't affect ingredient signatures.
        self.configAuxiliaryFlag = {}  # type: Dict[str, bool]

        stdoutHandler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('[%(asctime)s - %(name)s - %(levelname)s] %(message)s')
        stdoutHandler.setFormatter(formatter)
        self.logger.addHandler(stdoutHandler)
        self.logger.setLevel(logLevel or logging.INFO)

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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    def shutdown(self):
        self.logger.debug("Shutting down the plant.")

        # Remove attributes that can't be properly saved.
        for reactor in self.reactors.values():
            reactor.on_cache_save()

        with open(os.path.join(self.plantDir, 'plant.pcl'), 'wb') as file:
            pickle.dump({
                'reactors': self.reactors,
                'ingredients': self.ingredients
            }, file)

        self.warehouse.close()

    def add_reactors(self, *args):
        for func in args:
            assert(callable(func))

            if 'pyplant' not in func.__dict__ or not func.__dict__['pyplant'].isReactor:
                raise RuntimeError("'{}' is not a @ReactorFunc!".format(func))

            name = func.__name__

            self.logger.debug("Adding reactor '{}'.".format(name))

            if name in self.reactors:
                self.logger.debug("Reactor is already known. Storing the function, comparing signatures.")

                oldSignature = self.reactors[name].signature
                self.reactors[name].func = func

                # Compute the current reactor signature, using whatever metadata was kept
                # from the previous runs.
                # (If there were no previous runs, the reactor will be re-run anyway,
                #  obtaining a fresh signature. So, we do not check it here.)
                self._compute_reactor_signature(name)

                if self.reactors[name].signature != oldSignature:
                    self.logger.info("Reactor '{}' signature changed to '{}'. Metadata marked as outdated."
                                     .format(name, self.reactors[name].signature))
                    self.reactors[name].wasRun = False  # This new version was never executed.
                else:
                    self.logger.debug("Reactor did not change.")

            else:
                self.reactors[name] = Reactor(name, func)

    def is_ingredient_known(self, name: str):
        return name in self.ingredients

    def run_reactor(self, reactorToRun: Union[Callable, str]):

        if callable(reactorToRun):
            reactorName = reactorToRun.__name__
        else:
            reactorName = reactorToRun

        assert(reactorName in self.reactors)  # All reactors must be added to the plant.
        reactorObject = self.reactors[reactorName]

        assert(len(self.runningReactors) == 0)
        # Schedule the reactor for running.
        self._start_reactor(reactorObject)
        # Execute.
        self._finish_all_running_reactors()

    def _compute_ingredient_signature(self, ingredientName):
        self.logger.debug("Computing signature for ingredient '{}'.".format(ingredientName))
        ingredient = self._get_ingredient(ingredientName)
        if ingredient is not None and ingredient.isSignatureFresh:
            self.logger.debug("Found the signature in cache: '{}'".format(ingredient.signature))
            return ingredient.signature

        reactor = self._get_producing_reactor(ingredientName)

        # If we don't know the producing reactor, or it has changed and wasn't run yet,
        # we cannot compute the signature.
        if reactor is None or not reactor.wasRun:
            self.logger.debug("Producing reactor is unknown, signature is unknown.")
            return None

        self.logger.debug("Collecting sub-ingredient signatures...")
        subingredientSignatures = []
        for subingredientName in sorted(reactor.get_inputs()):
            subsignature = self._compute_ingredient_signature(subingredientName)
            if subsignature is None:
                self.logger.debug("Signature for '{}' is unknown.".format(ingredientName))
                return None
            subingredientSignatures.append(subsignature)

        self.logger.debug("Sub-ingredients' signatures found.")

        # Compute the signature as a hash of subingredients' signatures, config parameters
        # and reactor's signature (code).
        subingredientSignature = hashlib.sha1(''.join(subingredientSignatures).encode('utf-8')).hexdigest()

        # Collect all parameters that affect the ingredient. (Aux. params don't affect it, by definition.)
        parameterStrings = ['{}_{}'.format(name, self.config[name]) for name in sorted(reactor.get_params())
                            if name not in self.configAuxiliaryFlag]
        parameterSignature = hashlib.sha1(''.join(parameterStrings).encode('utf-8')).hexdigest()

        signatureParts = [reactor.get_signature(), subingredientSignature, parameterSignature, ingredientName]
        fullSignature = hashlib.sha1(''.join(signatureParts).encode('utf-8')).hexdigest()

        ingredient = self._get_or_create_ingredient(ingredientName)
        ingredient.set_current_signature(fullSignature)

        self.logger.debug("Computed signature for '{}': '{}'".format(ingredient.name, ingredient.signature))
        return fullSignature

    def _compute_reactor_signature(self, reactorName: str) -> str:
        """
        Computes and *updates* the signature of a reactor and
        all its referenced subreactors.

        :param reactorName:
        :return:
        """
        assert(reactorName in self.reactors)  # Must be added already.

        reactor = self.reactors[reactorName]
        reactorBodyHash = _compute_function_hash(reactor.func)

        nestedHashes = []
        for subreactorName in sorted(reactor.subreactors):
            # When a subreactor is imported, its hash must be stored in a static dict.
            if subreactorName in Plant.SubreactorSourceHashes:
                subreactorBodyHash = Plant.SubreactorSourceHashes[subreactorName]
                nestedHashes.append(subreactorBodyHash)
            else:
                # If it's not there - subreactor was deleted, provide gibberish to change the signature,
                # effectively marking the reactor outdated.
                nestedHashes.append('unknown-hash-value-gibberish')

        reactor.signature = _compute_string_hash(reactorBodyHash + ''.join(nestedHashes))

        return reactor.signature

    def _try_produce_ingredient(self, ingredientName):
        """
        Tries to produce an ingredient by checking which reactor used to
        produce an ingredient with this name.
        (An educated guess, better than running all new reactors.)

        :param ingredientName:
        :return:
        """
        self.logger.debug("Trying to produce ingredient '{}'.".format(ingredientName))

        self._get_or_create_ingredient(ingredientName)
        reactor = self._get_producing_reactor(ingredientName)
        if reactor is not None:
            self.logger.debug("Found the producing reactor '{}', scheduling.".format(reactor.name))
            self._start_reactor(reactor)
        else:
            self.logger.debug("Do not know how to produce '{}'.".format(ingredientName))

    def _start_reactor(self, reactorObject: 'Reactor') -> 'RunningReactor':
        self.logger.debug("Starting reactor '{}'.".format(reactorObject.name))
        if not reactorObject.wasRun:
            reactorObject.reset_metadata()

        if reactorObject.func is None:
            self.logger.info("Reactor '{}' was removed or renamed.".format(reactorObject.name))
            return None

        # Each reactor needs a pipework to send and receive data.
        pipework = self._get_or_create_pipework(reactorObject)

        # Schedule execution.
        runningReactor = Plant.RunningReactor(reactorObject.name, reactorObject)
        self.runningReactors[reactorObject.name] = runningReactor

        # Reactors are generator functions. Obtain a generator object (also executes until the first 'yield').
        generator = reactorObject.func(pipework)
        if not inspect.isgenerator(generator):
            raise RuntimeError("Reactor '{}' is not a generator function!".format(reactorObject.name))

        # Save the generator object for further execution.
        runningReactor.generator = generator

        return runningReactor

    def _finish_all_running_reactors(self):
        """
        Main function governing reactor execution.
        Maintains a queue of running reactors and ingredients that they're waiting for.
        Runs additional reactors as necessary to produce missing ingredients.
        :return:
        """
        self.logger.debug("Trying to finish all running reactors.")

        # Main loop over the running reactors.
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
                self.logger.info("Some ingredients are missing. Running new reactors...")
                for reactor in self.reactors.values():
                    if not reactor.wasRun and not self._is_reactor_running(reactor.name):
                        nextReactor = self._start_reactor(reactor)
                        break

            if nextReactor is None:
                debugDump = "Running reactors: {} \n".format(self.runningReactors.keys())
                missingIngredients = [r.awaitedIngredient for r in self.runningReactors.values()
                                      if r.awaitedIngredient is not None]
                debugDump += "Missing ingredients: {} \n".format(missingIngredients)
                self.logger.debug(">>> Plant state dump: \n" + debugDump)

                raise RuntimeError("Could not find a reactor that should run next. Reactors not added? Deadlock?")

            # We have now figured out which reactor to update next.
            self.logger.info("Next reactor to run: '{}'".format(nextReactor.name))

            valueToSend = None
            awaitedIngredient = nextReactor.awaitedIngredient
            # If the reactor is waiting for an ingredient, fetch and send it.
            if awaitedIngredient is not None:
                self.logger.debug("Providing reactor '{}' with ingredient '{}'".format(nextReactor.name, awaitedIngredient))
                valueToSend = self.warehouse.fetch(awaitedIngredient)

            # A reactor is a generator function that yields either fetched ingredients,
            # or 'IngredientAwaited' commands, and should be sent the ingredient.
            # Keep iterating the generator until it either terminates, or requests a missing ingredient.
            missingIngredient = None
            while missingIngredient is None:
                try:
                    timeBefore = time.time()
                    returnedObject = nextReactor.generator.send(valueToSend)  # Execute a step of the reactor
                    nextReactor.totalRunTime += time.time() - timeBefore  # Track time spend in execution.
                    if type(returnedObject) is Plant.IngredientAwaitedCommand:
                        # A missing ingredient is was requested, will pause the reactor.
                        nextReactor.awaitedIngredient = returnedObject.ingredientName
                        missingIngredient = returnedObject.ingredientName
                    elif type(returnedObject) is Plant.SubreactorStartedCommand:
                        # A sub-reactor is being launched. Remember the dependency and continue execution.
                        subreactorName = returnedObject.subreactorName
                        self.logger.info("Reactor '{}' is starting subreactor '{}'.".format(nextReactor.name, subreactorName))
                        nextReactor.reactorObject.register_subreactor(subreactorName)
                    else:
                        # A reactor successfully fetched an ingredient, no need to pause, just use it.
                        valueToSend = returnedObject

                except StopIteration:
                    totalRuntimeMin = self.runningReactors[nextReactor.name].totalRunTime / 60.0
                    self.logger.info("Reactor '{}' has finished running in {:.3} min."
                                     .format(nextReactor.name, totalRuntimeMin))
                    del self.runningReactors[nextReactor.name]

                    # Finished running, update the signature, since now we surely know
                    # all the referenced subreactors.
                    self._compute_reactor_signature(nextReactor.name)
                    nextReactor.reactorObject.wasRun = True

                    # Now that the reactor has finished running, we can compute the signatures for all
                    # the ingredients that it has produced. (Now we now all the inputs and sub-reactors.)
                    for outputName in nextReactor.reactorObject.outputs:
                        signature = self._compute_ingredient_signature(outputName)
                        assert(signature is not None)
                        self.warehouse.sign_fresh_ingredient(outputName, signature)

                    break
                except Exception as e:
                    self.logger.critical("Encountered an exception while executing reactor '{}'."
                                         .format(nextReactor.name))
                    raise

            if missingIngredient is not None:
                # A reactor paused due to a missing ingredient, try to produce it using previous knowledge.
                self.logger.info("Will try to produce '{}' for reactor '{}'.".format(missingIngredient, nextReactor.name))
                self._try_produce_ingredient(missingIngredient)

        self.logger.info("Finished running all reactors.")

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
            self.logger.debug("Creating pipework for reactor '{}'.".format(reactor.name))
            self.pipeworks[reactor.name] = Pipework(self, self.warehouse, reactor, self.logger)

        return self.pipeworks[reactor.name]

    def set_config(self, configMap):
        self.config = configMap

    def mark_params_as_auxiliary(self, params: List[str]):
        for name in params:
            if name not in self.config:
                raise RuntimeError("Parameter '{}' cannot be found.".format(name))
            self.configAuxiliaryFlag[name] = True

    def get_config_param(self, name: str) -> Any:
        if name in self.config:
            return self.config[name]

        raise RuntimeError("Unknown config parameter: '{}'".format(name))


class Pipework:
    """
    Used by reactors to send and receive ingredients.
    A unique instance is attached to each reactor,
    """

    def __init__(self, plant: Plant, warehouse: 'Warehouse', connectedReactor: 'Reactor', logger: logging.Logger):
        self.plant = plant
        self.warehouse = warehouse
        self.connectedReactor = connectedReactor  # Which reactor this pipework is connected to.
        self.logger = logger

    def receive(self, name) -> Any:
        self.logger.debug("Reactor '{}' is requesting ingredient '{}'".format(self.connectedReactor.name, name))
        self.connectedReactor.register_input(name)

        signature = self.plant._compute_ingredient_signature(name)
        if signature is None:
            return Plant.IngredientAwaitedCommand(name)

        ingredientValue = self.warehouse.fetch(name, signature)
        if ingredientValue is None:
            return Plant.IngredientAwaitedCommand(name)

        return ingredientValue

    def send(self, name: str, value: Any, type: 'Ingredient.Type'):
        self.logger.debug("Reactor '{}' is sending ingredient '{}'".format(self.connectedReactor.name, name))

        ingredient = self._register_output(name, type)
        self.warehouse.store(ingredient, value)

    def allocate(self, name: str, type: 'Ingredient.Type', **kwargs):
        self.logger.debug("Reactor '{}' is allocating ingredient '{}'".format(self.connectedReactor.name, name))

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

        return ingredient


class Reactor:

    def __init__(self, name: str, func: Callable[[Pipework], None]):

        # Whether the current reactor version was executed in the past.
        # If it was, we now that we can trust the metadata (inputs, outputs).
        # Subreactors don't have this flag, since they don't have the metadata,
        # they don't produce ingredients, it's all attributed to the parent reactor.
        self.wasRun = False

        self.name = name
        self.func = func
        self.generator = None
        self.inputs = set({})
        self.outputs = set({})
        self.params = set({})
        self.subreactors = set([])
        self.signature = None

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

    def register_subreactor(self, subreactorName):
        self.subreactors.add(subreactorName)

    def reset_metadata(self):
        self.inputs = set()
        self.outputs = set()
        self.params = set()
        self.subreactors = set()

    def on_cache_load(self):
        self.func = None
        self.generator = None

    def on_cache_save(self):
        self.func = None
        self.generator = None


class Ingredient:

    class Type(Enum):
        unknown = 0,
        simple = 1,
        list = 2,
        array = 3,
        huge_array = 4,
        object = 5,
        keras_model = 6

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

    def __init__(self, baseDir, logger: logging.Logger):
        self.baseDir = baseDir
        self.cache = {}
        self.h5File = h5py.File(os.path.join(self.baseDir, 'warehouse.h5py'), mode='a')
        self.logger = logger

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
        self.logger.debug("Fetching ingredient '{}' from the warehouse.".format(name))

        if name not in self.manifest:
            self.logger.debug("Ingredient is not in the warehouse.")
            return None
        elif signature is not None and signature != self.manifest[name]['signature']:
            # The stored ingredient is outdated (Right now we only store a single version of an ingredient).
            self.logger.debug("Ingredient is outdated. Pruning from the warehouse.")
            self._prune(name)
            return None

        if name in self.cache:
            self.logger.debug("Found the ingredient in the cache.")
            return self.cache[name]

        self.logger.debug("Fetching the ingredient from disk.")
        type = self.manifest[name]['type']
        if type == Ingredient.Type.simple:
            return self._fetch_simple(name)
        elif type == Ingredient.Type.list:
            return self._fetch_object(name)
        elif type == Ingredient.Type.array:
            return self._fetch_array(name)
        elif type == Ingredient.Type.huge_array:
            return self._fetch_huge_array(name)
        elif type == Ingredient.Type.object:
            return self._fetch_object(name)
        elif type == Ingredient.Type.keras_model:
            return self._fetch_keras_model(name)
        else:
            raise RuntimeError("This should never happen! Unsupported ingredient type: {}".format(type))

    def store(self, ingredient: Ingredient, value: Any):
        self.logger.debug("Storing ingredient '{}' in the warehouse.".format(ingredient.name))

        if ingredient.name in self.cache:
            raise RuntimeError("Ingredient is being overwritten, this is not yet supported.")

        if ingredient.name in self.manifest:
            self.logger.debug("Ingredient is already in the warehouse, pruning (not yet implemented).")  #todo
            self._prune(ingredient.name)

        if ingredient.type == Ingredient.Type.simple:
            self._store_simple(ingredient.name, value)
        elif ingredient.type == Ingredient.Type.list:
            self._store_object(ingredient.name, value)
        elif ingredient.type == Ingredient.Type.array:
            self._store_array(ingredient.name, value)
        elif ingredient.type == Ingredient.Type.huge_array:
            self._store_huge_array(ingredient.name, value)
        elif ingredient.type == Ingredient.Type.object:
            self._store_object(ingredient.name, value)
        elif ingredient.type == Ingredient.Type.keras_model:
            self._store_keras_model(ingredient.name, value)
        else:
            raise RuntimeError("Unsupported ingredient type: {}".format(ingredient.type))

        self.manifest[ingredient.name] = {
            'signature': ingredient.signature,
            'type': ingredient.type
        }

        self.cache[ingredient.name] = value

    def allocate(self, ingredient: Ingredient, **kwargs):
        self.logger.debug("Allocating storage for ingredient '{}' in the warehouse.".format(ingredient.name))
        if ingredient.type != Ingredient.Type.huge_array:
            raise RuntimeError("Allocation is not supported for an ingredient of type {}".format(ingredient.type))

        return self._allocate_huge_array(ingredient.name, **kwargs)

    def sign_fresh_ingredient(self, ingredientName: str, signature: str):
        self.logger.debug("Signing ingredient '{}' with signature '{}'.".format(ingredientName, signature))
        assert(signature is not None)
        assert(ingredientName in self.manifest)
        assert(ingredientName in self.cache)  # Must be fresh, i.e. must be in cache.

        # Store the new signature.
        self.manifest[ingredientName]['signature'] = signature

    def _prune(self, name):
        pass

    def _store_simple(self, name, value):
        self.h5File.attrs[name] = value

    def _fetch_simple(self, name):
        if name in self.h5File.attrs:
            return self.h5File.attrs[name]
        return None

    def _store_object(self, name, value):
        with open(os.path.join(self.baseDir, '{}.pck'.format(name)), 'wb') as file:
            pickle.dump(value, file)

    def _fetch_object(self, name):
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

        # If the dataset already exists, but has a wrong shape/type, recreate it.
        if dataset is not None and (dataset.shape != shape or dataset.dtype != dtype):
            del self.h5File[name]
            dataset = None

        if dataset is None:
            dataset = self.h5File.create_dataset(name, shape=shape, dtype=dtype)

        return dataset

    def _store_huge_array(self, name, value: h5py.Dataset):
        pass  # H5py takes care of storing to disk on-the-fly.

    def _fetch_huge_array(self, name):
        if name in self.h5File:
            return self.h5File[name]

        return None

    def _store_keras_model(self, name, value: keras.models.Model):
        value.save(os.path.join(self.baseDir, '{}.keras'.format(name)), overwrite=True)
        pass

    def _fetch_keras_model(self, name):
        modelPath = os.path.join(self.baseDir, '{}.keras'.format(name))
        if os.path.exists(modelPath):
            return keras.models.load_model(modelPath)

        return None

    def close(self):
        self.h5File.close()

        manifestPath = os.path.join(self.baseDir, 'manifest.pcl')
        with open(manifestPath, 'wb') as file:
            pickle.dump(self.manifest, file)


def _compute_function_hash(func: Callable) -> str:
    sourceLines = inspect.getsourcelines(func)
    functionSource = ''.join(sourceLines[0])

    return _compute_string_hash(functionSource)


def _compute_string_hash(string: str) -> str:
    return hashlib.sha1(string.encode('utf-8')).hexdigest()
