import os
import pickle
import inspect
import hashlib
from typing import Callable, Optional, Any
from enum import Enum
from types import SimpleNamespace

import numpy as np
import h5py

__all__ = ['Plant', 'ReactorFunc']


def ReactorFunc(func):
    print("decorating {}".format(func.__name__))

    assert('pyplant' not in func.__dict__)

    func.pyplant = SimpleNamespace()
    func.pyplant.isReactor = True

    return func


# def Produces(func, *args):
#     print("decorating {}".format(func.__name__))
#
#     assert('pyplant' in func.__dict__)
#
#     return func


class Plant:

    def __init__(self, plantDir: str):
        self.plantDir = plantDir
        self.reactors = {}
        self.ingredients = {}
        self.warehouse = Warehouse(plantDir)
        self.pipeworks = {}
        self.config = {}

        os.makedirs(self.plantDir)

        plantPath = os.path.join(self.plantDir, 'plant.pcl')
        if os.path.exists(plantPath):
            with open(plantPath, 'rb') as file:
                plantCache = pickle.load(file)
                self.reactors = plantCache['reactors']
                self.ingredients = plantCache['ingredients']

    def shutdown(self):
        print("Shutting down the plant.")

        pickle.dump({
            'reactors': self.reactors
        }, os.path.join(self.plantDir, 'plant.pcl'))

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

    def produce_ingredient(self, ingredientName):
        print("Producing ingredient '{}'.".format(ingredientName))

        self.get_or_create_ingredient(ingredientName)
        reactor = self.get_producing_reactor(ingredientName)
        if reactor is None:
            print("The producing reactor is not found. Running new reactors...")
            for reactor in self.reactors:
                if not reactor.wasRun:
                    self._run_reactor(reactor)
                    if ingredientName in reactor.inputs:
                        print("Found reactor '{}' that produces '{}'".format(reactor.name, ingredientName))
                        break  # Found a reactor that produced the ingredient.

            raise RuntimeError("Found no reactor that produces ingredient {}.".format(ingredientName))
        else:
            self._run_reactor(reactor)

    def compute_ingredient_signature(self, ingredientName):
        print("Computing signature for ingredient '{}'.".format(ingredientName))
        ingredient = self.get_ingredient(ingredientName)
        if ingredient is not None and ingredient.isSignatureFresh:
            print("Found the signature in cache: '{}'".format(ingredient.signature))
            return ingredient.signature

        reactor = self.get_producing_reactor(ingredientName)

        if reactor is None or not reactor.wasRun:
            print("Producing reactor unknown, signature unknown.")
            return None  # Don't know the reactor, or it has changed.
        
        print("Collecting sub-ingredient signatures...")
        subingredientSignatures = []
        for subingredientName in reactor.get_inputs():
            subsignature = self.compute_ingredient_signature(subingredientName)
            if subsignature is None:
                return None
            subingredientSignatures.append(subingredientName)

        print("Sub-ingredients' signatures found.")
        subingredientSignature = hashlib.sha1(''.join(subingredientSignatures).encode('utf-8')).hexdigest()

        parameterStrings = ['{}_{}'.format(name, self.config[name]) for name in reactor.get_params()]
        parameterSignature = hashlib.sha1(''.join(parameterStrings).encode('utf-8')).hexdigest()

        signatureParts = [reactor.get_signature(), subingredientSignature, parameterSignature]
        fullSignature = hashlib.sha1(''.join(signatureParts).encode('utf-8')).hexdigest()

        ingredient = self.get_or_create_ingredient(ingredientName)
        ingredient.set_current_signature(fullSignature)

        print("Computed signature: '{}'".format(ingredient.signature))
        return fullSignature

    def run_reactor(self, reactorFunc: Callable):
        assert(callable(reactorFunc))
        reactor = self.reactors[reactorFunc.__name__]

        self._run_reactor(reactor)

    def _run_reactor(self, reactorObject: Reactor):
        print("Running reactor '{}'.".format(reactorObject.name))
        if not reactorObject.wasRun:
            reactorObject.reset_metadata()

        pipework = self.get_or_create_pipework(reactorObject)
        reactorObject.func(pipework)

    def get_producing_reactor(self, name: str) -> Reactor:
        matchingReactors = (r for r in self.reactors if r.does_produce(name))
        return next(matchingReactors, None)

    def get_ingredient(self, name: str) -> Ingredient:
        matchingIngredients = (r for r in self.ingredients if r.does_produce(name))
        return next(matchingIngredients, None)

    def get_or_create_ingredient(self, name: str) -> Ingredient:
        ingredient = self.get_ingredient(name)
        if ingredient is None:
            ingredient = Ingredient(name)
            self.ingredients[name] = ingredient

        return ingredient

    def get_or_create_pipework(self, reactor: Reactor):
        if reactor.name not in self.pipeworks:
            print("Creating pipework for reactor '{}'.".format(reactor.name))
            self.pipeworks[reactor.name] = Pipework(self, self.warehouse, reactor)

        return self.pipeworks[reactor.name]

    def set_config(self, configMap):
        pass


class Pipework:

    def __init__(self, plant: Plant, warehouse: Warehouse, connectedReactor: Reactor):
        self.plant = plant
        self.warehouse = warehouse
        self.connectedReactor = connectedReactor  # Which reactor this pipework is connected to.

    def receive(self, name) -> Any:
        print("Reactor '{}' is requesting ingredient '{}'".format(self.connectedReactor.name, name))
        self.connectedReactor.register_input(name)

        signature = self.plant.compute_ingredient_signature(name)
        if signature is None:
            self.plant.produce_ingredient(name)
            return self.warehouse.fetch(name)

        ingredientValue = self.warehouse.fetch(name, signature)
        if ingredientValue is None:
            self.plant.produce_ingredient(name)
            return self.warehouse.fetch(name)

        return ingredientValue

    def send(self, name: str, value: Any, type: Ingredient.Type):
        print("Reactor '{}' is sending ingredient '{}'".format(self.connectedReactor.name, name))

        ingredient = self.plant.get_or_create_ingredient(name)
        ingredient.type = type
        ingredient.producerName = self.connectedReactor.name

        self.warehouse.store(ingredient, value)

    def allocate(self, name: str, type: Ingredient.Type, **kwargs):
        print("Reactor '{}' is allocating ingredient '{}'".format(self.connectedReactor.name, name))

        ingredient = self.plant.get_or_create_ingredient(name)
        ingredient.type = type
        ingredient.producerName = self.connectedReactor.name

        return self.warehouse.allocate(ingredient, **kwargs)

    def read_config(self, paramName):
        self.connectedReactor.register_parameter(paramName)

        #todo

        return 13


class Reactor:

    def __init__(self, name: str, func: Callable[[Pipework], None]):
        self.func = func
        self.name = name
        self.wasRun = False
        self.inputs = set({})
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

    def reset_metadata(self):
        self.inputs = set()
        self.params = set()


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
        self.type = Ingredient.Type.unknown
        self.producerName = None

    def set_current_signature(self, signature):
        self.signature = signature
        self.isSignatureFresh = True


class Warehouse:

    def __init__(self, baseDir):
        self.baseDir = baseDir
        self.cache = {}
        self.h5File = h5py.File(baseDir + 'warehouse.h5py', mode='r+')

        manifestPath = os.path.join(self.baseDir + 'manifest.pcl')
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
        elif signature is not None and signature != self.manifest[name].signature:
            # The stored ingredient is outdated (Right now we only store a single version of an ingredient).
            print("Ingredient is outdated. Pruing from the warehouse.")
            self._prune(name)
            return None

        if name in self.cache:
            print("Found the ingredient in the cache.")
            return self.cache[name]

        print("Fetching the ingredient from disk.")
        type = self.manifest[name].type
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
            print("Ingredient os already in the warehouse, pruning.")
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
            raise RuntimeError("Cannot allocate an ingredient of type {}".format(ingredient.type))

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
        np.save(os.path.join(self.baseDir, name), value, allow_pickle=False)

    def _fetch_array(self, name):
        return np.load(os.path.join(self.baseDir, name), allow_pickle=False)

    def _allocate_huge_array(self, name, shape, dtype):
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
