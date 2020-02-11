from typing import *

from pyplant.pyplant import *
# noinspection PyProtectedMember
from pyplant.pyplant import Warehouse


def store_ingredients_to_dir(plant: Plant, ingredientNames: List[str], dirPath: str):

    with Warehouse(dirPath, plant.logger) as warehouse:
        for name in ingredientNames:
            ingredientValue = plant.fetch_ingredient(name)
            ingredientObj = plant.get_ingredient_object(name)

            warehouse.store(ingredientObj, ingredientValue)


def store_reactor_inputs_to_dir(plant: Plant, reactorName: str, dirPath: str):
    reactorObj = plant.reactors[reactorName]

    return store_ingredients_to_dir(plant, list(reactorObj.inputs), dirPath)


def load_ingredients_from_dir(ingredientNames: Optional[Iterable[str]],
                              dirPath: str,
                              logger,
                              customKerasLayers: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    ingredients = {}
    with Warehouse(dirPath, logger) as warehouse:  # type: Warehouse
        if customKerasLayers:
            warehouse.set_custom_keras_layers(customKerasLayers)
        # Either load specific ingredients or all of them.
        ingredientNames = ingredientNames or warehouse.manifest.keys()
        for name in ingredientNames:
            ingredients[name] = warehouse.fetch(name, signature=None)

    return ingredients

