import logging
from typing import *

from pyplant import specs
from pyplant.pyplant import *
# noinspection PyProtectedMember
from pyplant.pyplant import Warehouse


def store_ingredients_to_dir(plant: Plant, ingredientNames: List[str], dirPath: str):
    with Warehouse(dirPath, plant.logger) as warehouse:
        # Don't take specs by ref, create new instances. Otherwise we'll close then when closing the warehouse.
        warehouse.register_ingredient_specs([type(spec)() for spec in plant.warehouse.ingredientSpecs.values()])
        for name in ingredientNames:
            ingredientValue = plant.fetch_ingredient(name)
            ingredientObj = plant.get_ingredient_object(name)

            warehouse.store(ingredientObj, ingredientValue)


def store_reactor_inputs_to_dir(plant: Plant, reactorName: str, dirPath: str):
    reactorObj = plant.reactors[reactorName]

    return store_ingredients_to_dir(plant, list(reactorObj.inputs), dirPath)


def load_ingredients_from_dir(dirPath: str,
                              ingredientNames: Optional[Iterable[str]] = None,
                              logger: Optional[logging.Logger] = None,
                              customSpecs: Optional[List[specs.IngredientTypeSpec]] = None) -> Dict[str, Any]:
    if logger is None:
        logger = logging.getLogger('_null')
        logger.setLevel(logging.CRITICAL)

    ingredients = {}
    with Warehouse(dirPath, logger) as warehouse:  # type: Warehouse
        if customSpecs:
            warehouse.register_ingredient_specs(customSpecs)
        # Either load specific ingredients or all of them.
        ingredientNames = ingredientNames or warehouse.manifest.keys()
        for name in ingredientNames:
            ingredients[name] = warehouse.fetch(name, signature=None)

    return ingredients
