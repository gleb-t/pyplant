from typing import *

from pyplant import *


# noinspection PyMissingConstructor
class PipeworkMock(Pipework):
    """
    A mock class to help with unit-testing reactors.
    """

    def __init__(self, configDict: Dict, ingredientsDict: Dict, callback: Callable[[], Any] = None):
        emptyCallback = lambda x: None

        self.config = configDict
        self.ingredients = ingredientsDict
        self.callback = callback or emptyCallback

        self.products = {}  # type: Dict[str, Ingredient]

    def read_config(self, paramName: str) -> Any:
        return self.config[paramName]

    def allocate_temp(self, name: str, type: 'Ingredient.Type', **kwargs):
        return self.callback(self.allocate_temp.__name__, name, type, **kwargs)

    def _register_output(self, name, type):
        return self.callback(self._register_output.__name__, name, type)

    def read_config_unregistered(self, paramName: str) -> Any:
        return self.config[paramName]

    def register_config_param(self, paramName):
        pass

    def send(self, name: str, value: Any, type: Optional['Ingredient.Type'] = None):
        self.products[name] = value

    def allocate(self, name: str, type: 'Ingredient.Type', **kwargs):
        return self.callback(self.allocate.__name__, name, type, **kwargs)

    def receive(self, name) -> Any:
        return self.ingredients[name]

    def get_products(self) -> Dict[str, Ingredient]:
        return self.products
