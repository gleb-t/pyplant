import logging
import tempfile
import unittest
from typing import *

import numpy as np

from pyplant import *
import pyplant.utils as utils


# noinspection PyCompatibility
class UtilsTest(unittest.TestCase):

    def test_store_to_dir_and_load(self):
        plantPath = tempfile.mkdtemp()
        dirPath = tempfile.mkdtemp()

        @ReactorFunc
        def reactor_a(pipe: Pipework, config):
            pipe.send('object', {'a': 'a', 'two': 2})
            pipe.send('object-other', {'b': 'b'})
            pipe.send('ndarray', np.arange(0, 1000, dtype=np.uint64))

            # Files are currently unsupported. (Implementation would need to be tweaked.)
            # filePath = pipe.allocate('file', Ingredient.Type.file)
            # with open(filePath, 'w') as file:
            #     file.write('file-contents')
            # pipe.send('file', filePath, Ingredient.Type.file)

        @ReactorFunc
        def reactor_b(pipe: Pipework, config):
            yield pipe.receive('object')
            yield pipe.receive('ndarray')

        with Plant(plantPath) as plant:  # type: pyplant.Plant
            plant.add_reactors(reactor_a, reactor_b)
            plant.run_reactor(reactor_b)

            utils.store_reactor_inputs_to_dir(plant, 'reactor_b', dirPath)

        ingredients = utils.load_ingredients_from_dir(dirPath, logger=logging.getLogger())

        self.assertEqual(ingredients['object'], {'a': 'a', 'two': 2})
        np.testing.assert_equal(ingredients['ndarray'], np.arange(0, 1000, dtype=np.uint64))

        # Shouldn't be exported, since it's not needed by another reactor.
        self.assertNotIn('object-other', ingredients)
