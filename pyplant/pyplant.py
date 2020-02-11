import copy
import hashlib
import inspect
import logging
import os
import pickle
import sys
import time
import warnings
from enum import Enum
from types import SimpleNamespace
from typing import *

import numpy as np
from dateutil.relativedelta import relativedelta

if TYPE_CHECKING:
    # These packages are needed to handle specific ingredient types, but they are only an optional dependency.
    import h5py
    import keras
    import scipy.sparse as sp

__all__ = ['Plant', 'ReactorFunc', 'SubreactorFunc', 'ConfigBase', 'ConfigValue', 'Pipework', 'Ingredient']


# Random documentation:
#
# Reactors are units of execution, whose results can be cached.
# Subreactors are effectively 'plugged-in' into reactors, and execute code on their behalf.
# The purpose of subreactors is to allow reactors to depend on subroutines, such that a change
# to the code of a subroutine (subreactor) could be detected.
# If the subroutine is not likely to change (e.g. a library), there is no benefit in using a subreactor.
#

def _compute_function_hash(func: Callable) -> str:
    try:
        sourceLines = inspect.getsourcelines(func)
    except OSError as e:
        warnings.warn("Failed to get function source code: {}".format(e))
        sourceLines = 'source-not-available_{}'.format(time.time())
    functionSource = ''.join(sourceLines[0])

    return _compute_string_hash(functionSource)


def _compute_string_hash(string: str) -> str:
    return hashlib.sha1(string.encode('utf-8')).hexdigest()


def ReactorFunc(func):
    # todo We should allow functions without this decorator to be added as reactors. It's not essential anymore.
    assert('pyplant' not in func.__dict__)

    func.pyplant = SimpleNamespace()
    func.pyplant.isReactor = True

    # Store the function globally, needed for implementing a convenience 'add all reactors' function,
    # that adds all visible reactors to a plant, without having to enumerate them manually.
    # noinspection PyProtectedMember
    Plant._ReactorFunctions.append(func)

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

    # Store the current function globally. Will be needed for reactor signature computation.
    # noinspection PyProtectedMember
    Plant._SubreactorFunctions[name] = func

    # noinspection PyCompatibility
    def _pyplant_subreactor_wrapper(*args, **kwargs):
        # Notify the plant, that a subreactor is being called.
        yield Plant.SubreactorStartedCommand(name)
        # Delegate to the subreactor, as if the reactor was running/yielding.
        if inspect.isgeneratorfunction(func):
            result = yield from func(*args, **kwargs)
        else:
            # If not a generator, simply call the function.
            result = func(*args, **kwargs)

        # Propagate the returned value to the caller
        return result

    return _pyplant_subreactor_wrapper


class Plant:

    # todo This a horrible hack which requires the users of PyPlant to provide a reference to the BNA constructor
    # todo We should pull BNAs into a separate package and have PyPlant rely on it. But I don't have time for it now.
    bnaConstructor = None  # type: Callable[[str, Warehouse.IBufferedArray.FileMode, Tuple, Type], Warehouse.IBufferedArray]

    class EventType(Enum):
        unknown = 0,
        reactor_started = 1,
        subreactor_started = 2,
        reactor_finished = 3,
        subreactor_finished = 4,  # Not implemented yet.
        reactor_step = 5

    # Global store of reactor functions, needed for the implementation of convenience function 'add all reactors'.
    _ReactorFunctions = []

    # Global store of current sub-reactor functions.
    # Filled out during code importing through function augmentors.
    _SubreactorFunctions = {}  # type: Dict[str, Callable]

    class RunningReactor:

        def __init__(self, name: str, reactorObject: 'Reactor', generator: object=None):

            self.name = name
            self.reactorObject = reactorObject
            self.generator = generator  # type: Generator[Any, Any]
            self.awaitedIngredient = None  # type: str
            self.subreactorCallStack = []  # type: List[str]

            self.totalRunTime = 0

    class ExecutionRecord:

        def __init__(self, reactorName: str, startTime: float = time.time(), totalRuntime: float = 0):
            self.reactorName = reactorName  # type: str
            self.startTime = startTime  # type: float
            self.totalRuntime = totalRuntime  # type: float
            self.finishTime = None  # type: str

        def __str__(self, *args, **kwargs):
            return str(self.__dict__)

        def __repr__(self, *args, **kwargs):
            return str(self.__dict__)

    class IngredientAwaitedCommand:

        def __init__(self, ingredientName: str):
            self.ingredientName = ingredientName

    class SubreactorStartedCommand:

        def __init__(self, subreactorName: str):
            self.subreactorName = subreactorName

    def __init__(self, plantDir: str, logger: logging.Logger=None, logLevel: int=None,
                 functionHash: Callable[[Callable], str] = _compute_function_hash,
                 stringHash: Callable[[str], str] = _compute_string_hash):
        """

        :param plantDir:
        :param logger:
        :param logLevel:
        :param functionHash: A Callable that will be used for computing function hashes.
        :param stringHash:  A Callable that will be used for computing string hashes.
        """

        if not os.path.exists(plantDir):
            os.makedirs(plantDir)

        self._setup_logger(logLevel, logger)
        self.plantDir = plantDir
        self.reactors = {}  # type: Dict[str, Reactor]
        self.runningReactors = {}  # type: Dict[str, Plant.RunningReactor]
        self.executionHistory = {}  # type: Dict[str, Plant.ExecutionRecord]
        self.ingredients = {}  # type: Dict[str, Ingredient]
        self.warehouse = Warehouse(plantDir, self.logger)
        self.pipeworks = {}
        self.config = ConfigBase()  # type: ConfigBase
        # Marks which config params are 'auxiliary' and shouldn't affect ingredient signatures.

        self.functionHash = functionHash
        self.stringHash = stringHash

        self.eventListeners = {}  # type: Dict[Plant.EventType, List[Callable[[Plant.EventType, Any], None]]]

        plantPath = os.path.join(self.plantDir, 'plant.pcl')
        if os.path.exists(plantPath):
            with open(plantPath, 'rb') as file:
                plantCache = pickle.load(file)
                self.reactors = plantCache['reactors']  # type: Dict[str, Reactor]
                self.ingredients = plantCache['ingredients']

    @staticmethod
    def _clear_global_state():
        """
        This method should only be called during unit testing, e.g. to test auto-adding reactors.
        """
        Plant._SubreactorFunctions = {}
        Plant._ReactorFunctions = []

    def _setup_logger(self, logLevel, logger):
        self.logger = logger
        if logger is None:
            self.logger = logging.getLogger('pyplant')
            self.logger.handlers = []  # In case the logger already exists.
            stdoutHandler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter('[%(asctime)s - %(name)s - %(levelname)s] %(message)s')
            stdoutHandler.setFormatter(formatter)
            self.logger.addHandler(stdoutHandler)
            self.logger.setLevel(logLevel or logging.INFO)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    def shutdown(self):
        self.logger.debug("Shutting down the plant.")

        self._save_cache()
        self.warehouse.close()

    def _save_cache(self):
        self.logger.info("Saving plant state to disk.")
        with open(os.path.join(self.plantDir, 'plant.pcl'), 'wb') as file:
            pickle.dump({
                'reactors': self.reactors,
                'ingredients': self.ingredients
            }, file)

    def add_all_visible_reactors(self):
        self.add_reactors(*Plant._ReactorFunctions)

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
                    self.logger.info("Reactor '{}' signature changed to '{}' from '{}'. Metadata marked as outdated."
                                     .format(name, self.reactors[name].signature, oldSignature))
                    self.reactors[name].wasRun = False  # This new version was never executed.
                else:
                    self.logger.debug("Reactor did not change.")

            else:
                self.reactors[name] = Reactor(name, func)

    def get_reactors(self):
        return [r.func for n, r in self.reactors.items()]

    def is_ingredient_known(self, name: str):
        return name in self.ingredients

    def run_reactor(self, reactorToRun: Union[Callable, str]):

        if callable(reactorToRun):
            reactorName = reactorToRun.__name__
        else:
            reactorName = reactorToRun

        # Be nice: if no reactors were added at all, try to add them automatically.
        # To check if the reactor was added, check if the function is known.
        if len([r for r in self.reactors.values() if r.func_exists()]) == 0:
            self.logger.debug("No reactors were added, will try to add them automatically.")
            self.add_all_visible_reactors()

        if reactorName not in self.reactors:
            raise ValueError("Couldn't find reactor '{}'. Was it added to the plant?".format(reactorName))

        assert(reactorName in self.reactors)  # All reactors must be added to the plant.
        reactorObject = self.reactors[reactorName]

        assert(len(self.runningReactors) == 0)

        # The reactor might have already been executed during this session,
        # for example when running unknown/modified reactors to produce
        # a missing ingredient.
        if reactorName in self.executionHistory:
            self.logger.info("Ignoring the command to start reactor '{}', since it was already run in this session."
                             .format(reactorName))
            return

        # Schedule the reactor for running.
        self._start_reactor(reactorObject)
        # Execute.
        self._finish_all_running_reactors()

    def fetch_ingredient(self, name: str):
        return self.warehouse.fetch(name)

    def add_event_listener(self, eventType: 'Plant.EventType', callback: Callable):
        if eventType not in self.eventListeners:
            self.eventListeners[eventType] = []

        self.eventListeners[eventType].append(callback)

    def remove_event_listener(self, callback: Callable[['Plant.EventType', Any], None]):
        for k, listenerList in self.eventListeners:
            if callback in listenerList:
                listenerList.remove(callback)

    def set_config(self, config: Union[Dict[str, Any], 'ConfigBase']):
        if isinstance(config, ConfigBase):
            self.config = config
        else:
            self.config = ConfigBase(dictionary=config.copy())

    def mark_params_as_auxiliary(self, params: List[str]):
        """
        Auxiliary config parameters do not affect the signature of ingredients,
        so that they can be changed without triggering re-production of the affected ingredients.
        """
        self.config.mark_auxiliary(params)

    def get_config_object(self) -> 'ConfigBase':
        return self.config

    def get_ingredient_object(self, name: str) -> 'Ingredient':
        if name not in self.ingredients:
            raise ValueError("Ingredient '{}' is not found.".format(name))

        return self.ingredients[name]

    def _peek_config_param(self, name: str) -> Any:
        return self.config.peek(name)

    def get_execution_history(self) -> Dict[str, ExecutionRecord]:
        return self.executionHistory

    def print_execution_history(self, printFn: Callable[[Any], Any]=print):
        recordsSorted = sorted(self.executionHistory.values(), key=lambda r: r.totalRuntime, reverse=True)
        printFn("====================================================================================================")
        printFn("{:50} day hrs min sec total".format("Reactor"))
        for record in recordsSorted:  # type: Plant.ExecutionRecord
            dur = relativedelta(seconds=record.totalRuntime)
            bits = ["{:3d}".format(int(getattr(dur, attr))) for attr in ['days', 'hours', 'minutes', 'seconds']]
            printFn("{:-<50} {} {:.2f}s".format(record.reactorName, " ".join(bits), record.totalRuntime))
        printFn("====================================================================================================")

    def _trigger_event(self, eventType: 'Plant.EventType', args: Tuple):
        if eventType in self.eventListeners:
            for callback in self.eventListeners[eventType]:
                callback(eventType, *args)

    def _compute_ingredient_signature(self, ingredientName):
        self.logger.debug("Computing signature for ingredient '{}'.".format(ingredientName))
        ingredient = self._get_ingredient(ingredientName)
        if ingredient is not None and ingredient.isSignatureFresh:
            self.logger.debug("Found the signature in cache: '{}'".format(ingredient.signature))
            return ingredient.signature

        reactor = self._get_producing_reactor(ingredientName)  # type: Reactor

        # If we don't know the producing reactor, or it has changed and wasn't run yet,
        # we cannot compute the signature.
        if reactor is None or not reactor.wasRun:
            self.logger.debug("Producing reactor is unknown or was never run in its current form, signature is unknown.")
            return None

        self.logger.debug("Collecting sub-ingredient signatures...")
        subingredientSignatures = []
        for subingredientName in sorted(reactor.get_inputs()):
            if subingredientName in reactor.get_outputs():  # Avoid infinite recursion on our own products.
                continue
            subsignature = self._compute_ingredient_signature(subingredientName)
            if subsignature is None:
                self.logger.debug("Signature for '{}' is unknown.".format(ingredientName))
                return None
            subingredientSignatures.append(subsignature)

        self.logger.debug("Sub-ingredients' signatures found.")

        # Compute the signature as a hash of subingredients' signatures, config parameters
        # and reactor's signature (code).
        subingredientSignature = self.stringHash(''.join(subingredientSignatures))

        # Collect all parameters that affect the ingredient. (Aux. params don't affect it, by definition.)
        try:
            parameterStrings = ['{}_{!r}'.format(name, self.config.peek(name)) for name in sorted(reactor.get_params())
                                if not self.config.is_auxiliary(name) and self.config.has(name)]
            parameterSignature = self.stringHash(''.join(parameterStrings))
        except KeyError as e:
            self.logger.info("Couldn't compute signature for ingredient '{}' because parameter '{}' is missing."
                             .format(ingredientName, e.args[0]))
            return None

        signatureParts = [reactor.get_signature(), subingredientSignature, parameterSignature, ingredientName]
        fullSignature = self.stringHash(''.join(signatureParts))

        ingredient = self._get_or_create_ingredient(ingredientName)
        ingredient.set_current_signature(fullSignature)

        self.logger.debug("Computed signature for '{}': '{}'".format(ingredient.name, ingredient.signature))
        return fullSignature

    def _compute_reactor_signature(self, reactorName: str) -> str:
        """
        Computes and *updates* the signature of a reactor and
        all its referenced subreactors.

        Computing the signature before running the reactor is valid because we have the metadata
        from the previous runs, e.g. which subreactors are being called. If the reactor has changed,
        and dependencies were added/removed, then its source code must have changed too,
        changing the signature. Thus, we can rely on the metadata, as long as the source code is unchanged.

        :param reactorName:
        :return:
        """
        assert(reactorName in self.reactors)  # Must be added already.

        reactor = self.reactors[reactorName]
        reactorBodyHash = self.functionHash(reactor.func)

        nestedHashes = []
        for subreactorName in sorted(reactor.subreactors):
            # When a subreactor is imported, its hash must be stored in a static dict.
            if subreactorName in Plant._SubreactorFunctions:
                subreactorBodyHash = self.functionHash(Plant._SubreactorFunctions[subreactorName])
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
        if reactor is not None and reactor.name not in self.executionHistory:
            self.logger.debug("Found the producing reactor '{}', scheduling.".format(reactor.name))
            self._start_reactor(reactor)
        else:
            self.logger.debug("Do not know how to produce '{}'.".format(ingredientName))

    def _start_reactor(self, reactorObject: 'Reactor') -> 'RunningReactor':
        self.logger.debug("Starting reactor '{}'.".format(reactorObject.name))
        if not reactorObject.wasRun:
            reactorObject.reset_metadata()

        if not reactorObject.func_exists():
            self.logger.info("Reactor '{}' was removed or renamed.".format(reactorObject.name))
            return None

        # No reactor should be ran twice during a session.
        assert reactorObject.name not in self.executionHistory

        # Each reactor needs a pipework to send and receive data.
        pipework = self._get_or_create_pipework(reactorObject)

        # Schedule execution.
        runningReactor = Plant.RunningReactor(reactorObject.name, reactorObject)
        self.runningReactors[reactorObject.name] = runningReactor
        self.executionHistory[reactorObject.name] = Plant.ExecutionRecord(reactorObject.name)

        # Reactors are generator functions. Obtain a generator object (also executes until the first 'yield').
        generator = reactorObject.start_func(pipework, pipework.config)

        # Save the generator object for further execution.
        runningReactor.generator = generator

        self._trigger_event(Plant.EventType.reactor_started, (reactorObject.name, reactorObject.func))

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
            nextReactor = self._get_next_reactor_to_update()
            self.logger.info("Next reactor to run: '{}'".format(nextReactor.name))

            valueToSend = None
            awaitedIngredient = nextReactor.awaitedIngredient
            # If the reactor is waiting for an ingredient, fetch and send it.
            if awaitedIngredient is not None:
                self.logger.debug("Providing reactor '{}' with ingredient '{}'".format(nextReactor.name,
                                                                                       awaitedIngredient))
                valueToSend = self.warehouse.fetch(awaitedIngredient)

            # A reactor is a generator function that yields either fetched ingredients,
            # or 'IngredientAwaited' commands, and should be sent the ingredient.
            # Keep iterating the generator until it either terminates, or requests a missing ingredient.
            missingIngredient = None
            while missingIngredient is None:
                timeBefore = time.time()
                try:
                    returnedObject = nextReactor.generator.send(valueToSend)  # Execute a step of the reactor
                    nextReactor.totalRunTime += time.time() - timeBefore  # Track the time spent in execution.
                    self._trigger_event(Plant.EventType.reactor_step, (nextReactor.name,))

                    if type(returnedObject) is Plant.IngredientAwaitedCommand:
                        # A missing ingredient is was requested, will pause the reactor.
                        nextReactor.awaitedIngredient = returnedObject.ingredientName
                        missingIngredient = returnedObject.ingredientName
                    elif type(returnedObject) is Plant.SubreactorStartedCommand:
                        # A sub-reactor is being launched. Remember the dependency and continue execution.
                        subreactorName = returnedObject.subreactorName
                        self.logger.debug("Reactor '{}' is starting subreactor '{}'.".format(nextReactor.name,
                                                                                             subreactorName))
                        nextReactor.reactorObject.register_subreactor(subreactorName)
                        self._trigger_event(Plant.EventType.subreactor_started, (subreactorName,))
                    else:
                        # A reactor successfully fetched an ingredient, no need to pause, just use it.
                        valueToSend = returnedObject

                except StopIteration:
                    nextReactor.totalRunTime += time.time() - timeBefore  # Don't miss the duration of the last step.
                    self._handle_reactor_finished(nextReactor)
                    break
                except Exception as e:
                    self.logger.critical("Encountered an exception while executing reactor '{}': {}"
                                         .format(nextReactor.name, e))
                    del self.runningReactors[nextReactor.name]

                    raise

            if missingIngredient is not None:
                # A reactor paused due to a missing ingredient, try to produce it using previous knowledge.
                self.logger.info("Will try to produce '{}' for reactor '{}'.".format(missingIngredient,
                                                                                     nextReactor.name))
                self._try_produce_ingredient(missingIngredient)

        self.logger.info("Finished running all reactors.")

    def _handle_reactor_finished(self, finishedReactor: RunningReactor):
        """
        Called from the plant execution loop when a reactor has finished.
        Performs bookkeeping by marking reactor as executed, signing
        the produced ingredients, etc.

        :param finishedReactor:
        :return:
        """

        totalRuntime = self.runningReactors[finishedReactor.name].totalRunTime
        self.logger.info("Reactor '{}' has finished running in {}"
                         .format(finishedReactor.name, self._format_duration(totalRuntime)))

        del self.runningReactors[finishedReactor.name]

        # Finished running, update the signature, since now we surely know
        # all the referenced subreactors.
        self._compute_reactor_signature(finishedReactor.name)
        finishedReactor.reactorObject.wasRun = True
        # Now that the reactor has finished running, we can compute the signatures for all
        # the ingredients that it has produced. (Now we know all the inputs and sub-reactors.)
        for outputName in finishedReactor.reactorObject.outputs:
            signature = self._compute_ingredient_signature(outputName)
            assert (signature is not None)
            self.warehouse.sign_fresh_ingredient(outputName, signature)

        # Deallocate any temp ingredients allocated by the reactor.
        for name in list(self.ingredients.keys()):  # Avoid iterating directly over the edited dict.
            ingredient = self.ingredients[name]
            if ingredient.isTemp and ingredient.producerName == finishedReactor.name:
                self.warehouse.deallocate_temp(ingredient)
                del self.ingredients[name]

        self.executionHistory[finishedReactor.name].finishTime = time.time()
        self.executionHistory[finishedReactor.name].totalRuntime = finishedReactor.totalRunTime

        # Save the current cache, so the results of this reactor
        # are safe from the potential future crashes of reactors that follow.
        self._save_cache()
        # Do the same for the warehouse, making sure that the new signatures made it to disk.
        self.warehouse.save_manifest()

        # Dispatch the event, now that all the relevant information is gathered and saved.
        self._trigger_event(Plant.EventType.reactor_finished, (finishedReactor.name, ))

    def _get_next_reactor_to_update(self) -> RunningReactor:
        """
        During the plant execution loop, figure out which reactor to update next
        based on available/missing ingredients and execution history.
        :return:
        """
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
                    if nextReactor is not None:  # Might be None if failed to start.
                        break

        if nextReactor is None:
            debugDump = "Running reactors: {} \n".format(self.runningReactors.keys())
            missingIngredients = self._get_awaited_ingredients()
            debugDump += "Missing ingredients: {} \n".format(missingIngredients)
            self.logger.info(">>> Plant state dump: \n" + debugDump)

            raise RuntimeError("Could not find a reactor that should run next. Reactors not added? Deadlock?")

        return nextReactor

    def _get_awaited_ingredients(self):
        return [reactor.awaitedIngredient for reactor in self.runningReactors.values()
                if reactor.awaitedIngredient is not None]

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

    @staticmethod
    def _format_duration(seconds: float) -> str:
        dur = relativedelta(seconds=seconds)
        bits = []
        for attr in ['days', 'hours', 'minutes']:
            attrVal = int(getattr(dur, attr))
            if attrVal > 0:
                bits.append("{:d}{}".format(attrVal, attr[0]))
        bits.append("{:.2f}s".format(dur.seconds))  # Always display seconds.
        return " ".join(bits)


class ConfigBase:
    """
    A base class for config objects that should be used with the Plant.
    Tracks the access to the config parameters and registers corresponding
    reactor dependencies.

    Provides an alternative (better) way of handling configuration,
    since IDE features for editing
    """

    def __init__(self, dictionary: Dict = None):
        if dictionary is not None:
            self.__dict__ = dictionary

        self._pipework = None  # type: Pipework
        self._auxiliaryFlags = {}  # type: Dict[str, bool]

    def peek(self, name: str):
        """
        Fetches a config parameter without registering a reactor dependency.
        Meant for backward compatibility and edge cases, mark parameters as auxiliary instead.

        :param name:
        :return:
        """
        self._throw_if_doesnt_exist(name)

        return object.__getattribute__(self, name)

    def has(self, paramName: str) -> bool:
        return paramName in self.__dict__

    def mark_auxiliary(self, params: List[str]):
        """
            @see PyPlant.mark_auxiliary
        """
        for name in params:
            self._throw_if_doesnt_exist(name)
            self._auxiliaryFlags[name] = True

    def is_auxiliary(self, name: str) -> bool:
        return name in self._auxiliaryFlags and self._auxiliaryFlags[name]

    def _getattribute_method(self, name: str):
        """
        A hack to improve PyCharm experience.

        Gets assigned to be the __get_attribute__ magic method in the module's __init__ file.
        We don't override it explicitly to hide this fact from PyCharm.
        If PyCharm sees a __getattribute__ method, it stops showing missing field warnings.
        """

        # Important to first check for the underscore to avoid recursion.
        if name.startswith('_') or name in ['peek', 'has', 'mark_auxiliary', 'is_auxiliary']:
            return object.__getattribute__(self, name)

        self._throw_if_doesnt_exist(name)

        if not self.is_auxiliary(name) and self._pipework is not None:
            self._pipework.register_config_param(name)

        return object.__getattribute__(self, name)

    def _setattr_method(self, name, value):
        if not name.startswith('_') and self._pipework is not None:
            raise RuntimeError("Editing configs from reactors is not allowed. "
                               "(Tried to change the '{}' parameter.)".format(name))

        object.__setattr__(self, name, value)

    def _clone_with_pipe(self, pipe: 'Pipework') -> 'ConfigBase':
        clone = copy.copy(self)
        clone._pipework = pipe

        return clone

    def _throw_if_doesnt_exist(self, name):
        if name not in self.__dict__:
            raise RuntimeError("The config parameter '{}' does not exist.".format(name))


class ConfigValue:
    """
    A base class for POD-style objects used as values in a Plant's config. (see ConfigBase)
    Implements proper string representation to support hashing by the plant.
    Also used as a marker by some external code to know when to expand placeholders
    in the object's fields.
    """

    def __repr__(self):
        return "{}: {}".format(self.__class__, self.__dict__)


# noinspection PyProtectedMember
class Pipework:
    """
    Used by reactors to send and receive ingredients.
    A unique instance is attached to each reactor.
    """

    def __init__(self, plant: Plant, warehouse: 'Warehouse', connectedReactor: 'Reactor', logger: logging.Logger):
        self._plant = plant  # type: Plant
        self._warehouse = warehouse
        self._connectedReactor = connectedReactor  # Which reactor this pipework is connected to.
        self._logger = logger
        self.config = plant.get_config_object()._clone_with_pipe(self)

    def receive(self, name) -> Any:
        self._logger.debug("Reactor '{}' is requesting ingredient '{}'".format(self._connectedReactor.name, name))
        self._connectedReactor.register_input(name)

        signature = self._plant._compute_ingredient_signature(name)
        # If signature is unknown, we can't fetch an ingredient (needs to be produced).
        # Exception: if fetching reactor's own product and it's fresh (though might still have no signature),
        # we can fetch it right away.
        isFetchingOwnProduct = name in self._connectedReactor.outputs and self._plant._is_ingredient_fresh(name)
        isSignatureKnown = signature is not None

        ingredientValue = None
        if isFetchingOwnProduct:
            # Ignore signature even when known: we have just produced the ingredient and can fetch it safely.
            # Otherwise, we might still be using an outdated signature (signatures update after reactor stops).
            ingredientValue = self._warehouse.fetch(name, None)
        elif isSignatureKnown:
            ingredientValue = self._warehouse.fetch(name, signature)
        else:
            self._logger.debug("Ingredient '{}' signature is not known.".format(name))

        if ingredientValue is None:
            return Plant.IngredientAwaitedCommand(name)

        return ingredientValue

    def send(self, name: str, value: Any, type: Optional['Ingredient.Type'] = None):
        self._logger.debug("Reactor '{}' is sending ingredient '{}'.".format(self._connectedReactor.name, name))

        if type is None:
            type = Ingredient.infer_type_from_value(value)

        ingredient = self._register_output(name, type)
        self._warehouse.store(ingredient, value)

    def allocate(self, name: str, type: 'Ingredient.Type', **kwargs):
        self._logger.debug("Reactor '{}' is allocating ingredient '{}'.".format(self._connectedReactor.name, name))

        ingredient = self._register_output(name, type)
        return self._warehouse.allocate(ingredient, **kwargs)

    def allocate_temp(self, name: str, type: 'Ingredient.Type', **kwargs):
        self._logger.debug("Reactor '{}' is allocating a temp ingredient '{}'.".format(self._connectedReactor.name, name))

        # Temp ingredients aren't registered as outputs.
        ingredient = self._create_ingredient(name, type)
        ingredient.isTemp = True

        return self._warehouse.allocate_temp(ingredient, **kwargs)

    def read_config(self, paramName: str) -> Any:
        """
        This method is for backward compatibility, the config object should be used instead.
        :param paramName:
        :return:
        """
        self.register_config_param(paramName)

        return self._plant._peek_config_param(paramName)

    def register_config_param(self, paramName):
        self._connectedReactor.register_parameter(paramName)

        pass

    def read_config_unregistered(self, paramName: str) -> Any:
        """
        Read a configuration parameter without registering it with the plant,
        i.e. without creating a dependency.
        (Useful for reporting/debugging code.)
        :param paramName:
        :return:
        """
        return self._plant._peek_config_param(paramName)

    def _register_output(self, name, type):
        ingredient = self._create_ingredient(name, type)
        self._connectedReactor.register_output(name)

        return ingredient

    def _create_ingredient(self, name, type):
        ingredient = self._plant._get_or_create_ingredient(name)
        ingredient.type = type
        ingredient.producerName = self._connectedReactor.name
        ingredient.isFresh = True

        return ingredient


class Reactor:

    def __init__(self, name: str, func: Callable[['Pipework', 'ConfigBase'], 'Generator']):

        # Whether the current reactor version was executed in the past.
        # If it was, we know that we can trust the metadata (inputs, outputs, etc.).
        # Subreactors don't have this flag, since they don't have the metadata,
        # they don't produce ingredients, it's all attributed to the parent reactor.
        self.wasRun = False

        self.name = name
        self.func = func
        self.generator = None       # type: Optional[Generator]
        self.inputs = set({})       # type: Set[str]
        self.outputs = set({})      # type: Set[str]
        self.params = set({})       # type: Set[str]
        self.subreactors = set([])  # type: Set[str]
        self.signature = None       # type: Optional[str]

    def get_signature(self):
        return self.signature

    def get_inputs(self):
        return self.inputs

    def get_outputs(self):
        return self.outputs

    def get_params(self):
        return self.params

    def func_exists(self) -> bool:
        """
        The function may not exist, if the reactor object was loaded from disk,
        but the corresponding reactor function was removed from the code.
        :return:
        """
        return self.func is not None

    def start_func(self, pipe: 'Pipework', config: 'ConfigBase') -> Generator:
        if len(inspect.signature(self.func).parameters) == 2:
            args = (pipe, config)
        else:
            # For backward compatibility: support reactors that don't accept a config object.
            # noinspection PyArgumentList
            args = (pipe, )

        if inspect.isgeneratorfunction(self.func):
            return self.func(*args)
        else:
            # Support reactors that are not generators by wrapping them in one.
            def _generator_wrapper():
                self.func(*args)
                yield

            return _generator_wrapper()

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
        # todo can't remember why this check is needed.
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

    def __setstate__(self, state):
        """
        When unpickling (loading from disk) make sure we're not loading
        any old runtime-dependent attributes.
        :param state:
        :return:
        """
        self.__dict__.update(state)

        self.func = None
        self.generator = None

    def __getstate__(self):
        """
        When pickling (saving Plant state to disk) ignore the runtime-dependent attributes.
        :return:
        """
        d = self.__dict__.copy()
        del d['func']
        del d['generator']

        return d


class Ingredient:

    class Type(Enum):
        unknown = 0,
        simple = 1,
        list = 2,
        array = 3,
        hdf_array = 4,
        object = 5,
        keras_model = 6,
        file = 7,
        buffered_array = 8,
        scipy_sparse = 9

    @classmethod
    def infer_type_from_value(cls, value) -> 'Ingredient.Type':
        """
        Infer the ingredient storage type from the provided object.
        """

        valueType = type(value)
        if valueType in [int, float, complex, str, bool]:
            return Ingredient.Type.simple
        elif valueType is list:
            return Ingredient.Type.list
        elif valueType is np.ndarray:
            return Ingredient.Type.array
        elif 'scipy.sparse' in sys.modules and sys.modules['scipy.sparse'].issparse(value):
            return Ingredient.Type.scipy_sparse
        elif 'h5py' in sys.modules and isinstance(value, sys.modules['h5py'].Dataset):
            return Ingredient.Type.hdf_array
        elif 'keras' in sys.modules and isinstance(value, sys.modules['keras'].models.Model):
            return Ingredient.Type.keras_model
        else:
            return Ingredient.Type.object

    def __init__(self, name: str, type: Optional['Ingredient.Type'] = None):
        self.name = name
        self.signature = None
        self.isSignatureFresh = False
        self.isFresh = False  # Whether has been produced during the current plant run (not loaded from disk).
        self.type = type or Ingredient.Type.unknown
        self.producerName = None
        # Temporary ingredients can be created to hold intermediate results, and will be removed
        # when the reactor finishes.
        self.isTemp = False

    def set_current_signature(self, signature):
        self.signature = signature
        self.isSignatureFresh = True

    def __setstate__(self, state):
        """
        When unpickling, reset the freshness flags.
        They are only valid during a single session, and the plant is being loaded from disk (new session).
        :param state:
        :return:
        """
        self.__dict__.update(state)
        self.isSignatureFresh = False
        self.isFresh = False


class Warehouse:

    class IBufferedArray:
        # todo pull BNAs into a package so we can use that type directly and avoid copy-paste

        class FileMode(Enum):
            """
            The integer value must be identical to the C++ implementation.
            """
            unknown = 0
            readonly = 1
            update = 2
            rewrite = 3

        def __init__(self, filepath: str, fileMode: FileMode, shape: Tuple,
                     dtype: np.dtype, maxBufferSize: int):
            self.filepath = filepath
            self.fileMode = fileMode
            self.dtype = dtype  # type: np.dtype
            self.ndim = len(shape)
            self.shape = shape
            self.maxBufferSize = maxBufferSize
            self.cPointer = 0

            raise RuntimeError("This class is an interface and shouldn't be instantiated.")

        def __enter__(self):
            pass

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

        def read_slice(self, index: int) -> np.ndarray:
            pass

        def destruct(self):
            pass

        def flush(self, flushOsBuffer: bool = False):
            pass

        def fill_box(self, value, cornerLow: Tuple, cornerHigh: Tuple):
            pass

    def __init__(self, baseDir, logger: logging.Logger):
        self.baseDir = baseDir
        self.cache = {}              # type: Dict[str, Any]
        self.simpleStore = {}        # type: Dict[str, Any]
        self.h5Files = {}            # type: Dict[str, h5py.File]
        self.bufferedArrays = {}     # type: Dict[str, Warehouse.IBufferedArray]
        self.customKerasLayers = {}  # type: Dict[str, Any]
        self.logger = logger
        self.manifest = {}           # type: Dict[str, Dict[str, Any]]

        try:
            manifestPath = os.path.join(os.path.join(self.baseDir, 'manifest.pyplant.pcl'))
            if os.path.exists(manifestPath):
                with open(manifestPath, 'rb') as file:
                    self.manifest = pickle.load(file)

            self.simpleStorePath = os.path.join(os.path.join(self.baseDir, 'simple.pyplant.pcl'))
            if os.path.exists(self.simpleStorePath):
                with open(self.simpleStorePath, 'rb') as file:
                    self.simpleStore = pickle.load(file)
            else:
                self.simpleStore = {}
        except pickle.UnpicklingError as e:
            self.logger.warning("Failed to load the warehouse state. Corrupted files?", exc_info=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

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

        if signature is not None and signature != self.manifest[name]['signature']:
            # The stored ingredient is outdated (Right now we only store a single version of an ingredient).
            self.logger.debug("Ingredient is outdated. Pruning from the warehouse.")
            self.logger.debug("Warehouse signature was: '{}'.".format(self.manifest[name]['signature']))
            self._prune(name, self.manifest[name]['type'])
            return None

        if name in self.cache:
            self.logger.debug("Found the ingredient in the cache.")
            return self.cache[name]

        self.logger.debug("Fetching the ingredient from disk.")
        type = self.manifest[name]['type']
        meta = self.manifest[name]['metadata']
        if type == Ingredient.Type.simple:
            return self._fetch_simple(name)
        elif type == Ingredient.Type.list:
            return self._fetch_object(name)
        elif type == Ingredient.Type.array:
            return self._fetch_array(name)
        elif type == Ingredient.Type.scipy_sparse:
            return self._fetch_scipy_sparse(name)
        elif type == Ingredient.Type.hdf_array:
            return self._fetch_hdf_array(name)
        elif type == Ingredient.Type.object:
            return self._fetch_object(name)
        elif type == Ingredient.Type.keras_model:
            return self._fetch_keras_model(name)
        elif type == Ingredient.Type.file:
            return self._fetch_file(name)
        elif type == Ingredient.Type.buffered_array:
            return self._fetch_buffered_array(name, meta)
        else:
            raise RuntimeError("This should never happen! Unsupported ingredient type: {}".format(type))

    def store(self, ingredient: Ingredient, value: Any):
        self.logger.debug("Storing ingredient '{}' in the warehouse.".format(ingredient.name))

        if ingredient.name in self.cache:
            raise RuntimeError("Ingredient is being overwritten, this is not yet supported.")

        if ingredient.name in self.manifest:
            self.logger.debug("Ingredient is already in the warehouse, pruning.")
            self._prune(ingredient.name, ingredient.type)

        metadata = {}
        if ingredient.type == Ingredient.Type.simple:
            self._store_simple(ingredient.name, value)
        elif ingredient.type == Ingredient.Type.list:
            self._store_object(ingredient.name, value)
        elif ingredient.type == Ingredient.Type.array:
            self._store_array(ingredient.name, value)
        elif ingredient.type == Ingredient.Type.scipy_sparse:
            self._store_scipy_sparse(ingredient.name, value)
        elif ingredient.type == Ingredient.Type.hdf_array:
            self._store_hdf_array(ingredient.name, value)
        elif ingredient.type == Ingredient.Type.object:
            self._store_object(ingredient.name, value)
        elif ingredient.type == Ingredient.Type.keras_model:
            self._store_keras_model(ingredient.name, value)
        elif ingredient.type == Ingredient.Type.file:
            self._store_file(ingredient.name, value)
        elif ingredient.type == Ingredient.Type.buffered_array:
            metadata = self._store_buffered_array(ingredient.name, value)
        else:
            raise RuntimeError("Unsupported ingredient type: {}".format(ingredient.type))

        self.manifest[ingredient.name] = {
            'signature': ingredient.signature,
            'type': ingredient.type,
            'metadata': metadata
        }
        self.save_manifest()

        self.cache[ingredient.name] = value

    def allocate(self, ingredient: Ingredient, **kwargs):
        self.logger.debug("Allocating storage for ingredient '{}' in the warehouse.".format(ingredient.name))
        if ingredient.type == Ingredient.Type.hdf_array:
            return self._allocate_hdf_array(ingredient.name, **kwargs)
        elif ingredient.type == Ingredient.Type.file:
            return self._allocate_file(ingredient.name, **kwargs)
        elif ingredient.type == Ingredient.Type.buffered_array:
            return self._allocate_buffered_array(ingredient.name, **kwargs)
        else:
            raise RuntimeError("Allocation is not supported for an ingredient of type {}".format(ingredient.type))

    def allocate_temp(self, ingredient: Ingredient, **kwargs):
        """
        Temp ingredients are automatically removed when the allocating reactor finishes.
        :param ingredient:
        :param kwargs:
        :return:
        """
        self.logger.debug("Allocating storage for a temp ingredient '{}' in the warehouse.".format(ingredient.name))
        if ingredient.type == Ingredient.Type.hdf_array:
            return self._allocate_hdf_array('temp_' + ingredient.name, **kwargs)
        elif ingredient.type == Ingredient.Type.buffered_array:
            return self._allocate_buffered_array('temp_' + ingredient.name, **kwargs)
        else:
            raise RuntimeError("Temp allocation is not supported for an ingredient of type {}".format(ingredient.type))

    def deallocate_temp(self, ingredient: Ingredient):
        self.logger.debug("Deallocating a temp ingredient '{}' from the warehouse.".format(ingredient.name))
        if ingredient.type == Ingredient.Type.hdf_array:
            return self._deallocate_hdf_array('temp_' + ingredient.name)
        elif ingredient.type == Ingredient.Type.buffered_array:
            return self._deallocate_buffered_array('temp_' + ingredient.name)
        else:
            raise RuntimeError("Deallocation is not supported for an ingredient of type {}".format(ingredient.type))

    def sign_fresh_ingredient(self, ingredientName: str, signature: str):
        self.logger.debug("Signing ingredient '{}' with signature '{}'.".format(ingredientName, signature))
        assert(signature is not None)
        assert(ingredientName in self.manifest)  # Did you forget to send an allocated array?
        assert(ingredientName in self.cache)  # Must be fresh, i.e. must be in cache.

        # Store the new signature.
        self.manifest[ingredientName]['signature'] = signature

    def set_custom_keras_layers(self, customLayers: Dict[str, Any]):
        self.customKerasLayers = customLayers

    def _prune(self, name: str, type: Ingredient.Type):
        pass  # We are overwriting on store, for now there's no need for explicit pruning.

    def _store_simple(self, name, value):
        self.simpleStore[name] = value
        with open(self.simpleStorePath, 'wb') as file:
            pickle.dump(self.simpleStore, file)

    def _fetch_simple(self, name):
        # The whole store is loaded on start, no need to check the disk.
        if name in self.simpleStore:
            return self.simpleStore[name]
        return None

    def _store_object(self, name, value):
        with open(os.path.join(self.baseDir, '{}.pcl'.format(name)), 'wb') as file:
            pickle.dump(value, file)

    def _fetch_object(self, name):
        path = os.path.join(self.baseDir, '{}.pcl'.format(name))
        if os.path.exists(path):
            with open(path, 'rb') as file:
                return pickle.load(file)

        return None

    def _store_array(self, name, value: np.ndarray):
        np.save(os.path.join(self.baseDir, '{}.npy'.format(name)), value)

    def _fetch_array(self, name):
        return np.load(os.path.join(self.baseDir, '{}.npy'.format(name)))

    def _store_scipy_sparse(self, name, value):
        import scipy.sparse as sp

        sp.save_npz(os.path.join(self.baseDir, '{}.npz'.format(name)), value)

    def _fetch_scipy_sparse(self, name):
        import scipy.sparse as sp

        return sp.load_npz(os.path.join(self.baseDir, '{}.npz'.format(name)))

    def _allocate_hdf_array(self, name, shape, dtype=np.float, **kwargs):
        import h5py

        dataset = self._fetch_hdf_array(name)
        h5FilePath = self._get_hdf_array_filepath(name)

        # todo should probably always recreate the array, its cheap and consistent with BNAs.

        # If the dataset already exists, but has a wrong shape/type, recreate it.
        if dataset is not None and (dataset.shape != shape or dataset.dtype != dtype):
            try:
                self.h5Files[name].close()
                os.remove(h5FilePath)
            except RuntimeError as e:
                self.logger.warning("Suppressed an error while removing dataset '{}' Details: {}"
                                    .format(name, e))
            dataset = None

        if dataset is None:
            self.h5Files[name] = h5py.File(h5FilePath, 'a')
            dataset = self.h5Files[name].create_dataset('data', shape=shape, dtype=dtype, **kwargs)

        return dataset

    def _deallocate_hdf_array(self, name):
        h5FilePath = self._get_hdf_array_filepath(name)

        if name not in self.h5Files:
            raise RuntimeError("Cannot deallocate HDF array '{}', it doesn't exist.".format(name))

        if not os.path.exists(h5FilePath):
            raise RuntimeError("Cannot deallocate HDF array '{}', the file doesn't exist: '{}'."
                               .format(name, h5FilePath))

        self.h5Files[name].close()
        del self.h5Files[name]
        os.unlink(h5FilePath)

    def _get_hdf_array_filepath(self, name):
        return os.path.join(self.baseDir, name + '.hdf')

    def _get_buffered_array_filepath(self, name):
        return os.path.join(self.baseDir, name + '.bna')

    # noinspection PyUnusedLocal
    def _store_hdf_array(self, name, value: 'h5py.Dataset'):
        # H5py takes care of storing to disk on-the-fly.
        # Just flush the data, to make sure that it is persisted.
        self.h5Files[name].flush()
        pass

    def _fetch_hdf_array(self, name):
        import h5py

        h5FilePath = self._get_hdf_array_filepath(name)
        if name not in self.h5Files:
            if not os.path.exists(h5FilePath):
                return None

            # If the array isn't in the manifest, but exists on disk, attempt to open it.
            try:
                self.h5Files[name] = h5py.File(h5FilePath, 'a')
            except OSError as e:
                self.logger.warning("Failed to open the HDF array at '{}' with error: {}"
                                    .format(h5FilePath, str(e)))
                self.logger.info("Deleting the corrupted file.")
                os.remove(h5FilePath)
                return None

        dataset = None
        try:
            if 'data' in self.h5Files[name]:
                # Try accessing the dataset. (Can crush for corrupted files.)
                dataset = self.h5Files[name]['data']
        except BaseException as e:
            self.logger.warning("Suppressed an error while accessing HDF-dataset '{}' Details: {}"
                                .format(name, e))

        if dataset is None:
            self.logger.info("The HDF file at '{}' has no dataset. Corrupted file, deleting.".format(h5FilePath))
            try:
                self.h5Files[name].close()
                os.remove(h5FilePath)
            except BaseException as e:
                self.logger.warning("Suppressed an error while removing HDF-dataset '{}' Details: {}"
                                    .format(name, e))
            return None

        return dataset

    def _store_keras_model(self, name, value):

        value.save(os.path.join(self.baseDir, '{}.keras'.format(name)), overwrite=True)
        pass

    def _fetch_keras_model(self, name):
        import keras.models

        modelPath = os.path.join(self.baseDir, '{}.keras'.format(name))
        if os.path.exists(modelPath):
            return keras.models.load_model(modelPath, custom_objects=self.customKerasLayers)

        return None

    def _allocate_file(self, name, **kwargs) -> str:
        # Create an empty file, clear if it already exists.
        filepath = self._get_filepath_from_name(name)
        open(filepath, 'w').close()

        return filepath

    def _store_file(self, name: str, value):
        expectedPath = self._get_filepath_from_name(name)
        if value != expectedPath:
            raise RuntimeError("Moving file ingredients is not allowed. Expected value: {}".format(expectedPath))

        # Don't have to do anything here, the filesystem takes care of the files.
        pass

    def _fetch_file(self, name: str) -> Union[str, None]:
        filepath = self._get_filepath_from_name(name)
        if os.path.exists(filepath):
            return filepath

        return None

    def _allocate_buffered_array(self, name, shape, dtype=np.float, **kwargs):
        if not Plant.bnaConstructor:
            raise RuntimeError("BNA constructor (class) needs to be provided to PyPlant.")

        filepath = self._get_buffered_array_filepath(name)
        # metadata = self.manifest[name]['metadata']

        if name in self.bufferedArrays:
            raise RuntimeError("Cannot allocate buffered array '{}', it's already opened!".format(name))

        # For BNAs, we simple recreate the file if it already exists.
        # If the array already exists, but has a wrong shape/type, recreate it.
        if os.path.exists(filepath):
            self.logger.debug("Found BNA '{}' on disk while allocating. Recreating the file.".format(name))
            try:
                os.remove(filepath)
            except RuntimeError as e:
                self.logger.warning("Suppressed an error while removing dataset '{}' Details: {}"
                                    .format(name, e))

        self.bufferedArrays[name] = Plant.bnaConstructor(filepath, Warehouse.IBufferedArray.FileMode.rewrite,
                                                         shape, dtype, **kwargs)
        return self.bufferedArrays[name]

    def _deallocate_buffered_array(self, name):
        bnaFilepath = self._get_buffered_array_filepath(name)

        if name not in self.bufferedArrays:
            raise RuntimeError("Cannot deallocate buffered array '{}', it doesn't exist.".format(name))

        if not os.path.exists(bnaFilepath):
            raise RuntimeError("Cannot deallocate buffered array '{}', the file doesn't exist: '{}'."
                               .format(name, bnaFilepath))

        self.bufferedArrays[name].destruct()
        del self.bufferedArrays[name]
        os.unlink(bnaFilepath)

    def _store_buffered_array(self, name, value: 'Warehouse.IBufferedArray') -> Dict[str, Any]:
        # Just flush the data, to make sure that it is persisted.
        self.bufferedArrays[name].flush(flushOsBuffer=True)
        metadata = {
            'shape': value.shape,
            'dtype_str': value.dtype.str,
            'buffer_size': value.maxBufferSize
        }

        return metadata

    def _fetch_buffered_array(self, name: str, metadata: Dict[str, Any]):
        if not Plant.bnaConstructor:
            raise RuntimeError("BNA constructor (class) needs to be provided to PyPlant.")

        filepath = self._get_buffered_array_filepath(name)
        if name not in self.bufferedArrays:
            if not os.path.exists(filepath):
                return None

            # If the array isn't in the manifest, but exists on disk, attempt to open it.
            try:
                self.bufferedArrays[name] = Plant.bnaConstructor(filepath, Warehouse.IBufferedArray.FileMode.update,
                                                                 metadata['shape'], np.dtype(metadata['dtype_str']),
                                                                 metadata['buffer_size'])
            except OSError as e:
                self.logger.warning("Failed to open the buffered array at '{}' with error: {}"
                                    .format(filepath, str(e)))
                self.logger.info("Deleting the corrupted file.")
                os.remove(filepath)
                return None

        return self.bufferedArrays[name]

    def _get_filepath_from_name(self, fileName: str) -> str:
        return os.path.join(self.baseDir, 'file_{}'.format(fileName))

    def save_manifest(self):
        manifestPath = os.path.join(self.baseDir, 'manifest.pyplant.pcl')
        with open(manifestPath, 'wb') as file:
            pickle.dump(self.manifest, file)

    def close(self):
        for name, h5File in list(self.h5Files.items()):
            h5File.close()
            del self.h5Files[name]

        for bna in self.bufferedArrays.values():
            bna.destruct()

        self.save_manifest()

        # Explicitly clear the cache. This immediately deallocates NumPy arrays, instead of waiting for GC.
        for name in list(self.cache.keys()):
            del self.cache[name]
