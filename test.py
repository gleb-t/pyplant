
import pyplant
import numpy as np


@pyplant.ReactorFunc
def reactor_a(pipe: pyplant.Pipework):

    maxElems = pipe.read_config('element_number')
    pipe.send('range', np.arange(0, maxElems), pyplant.Ingredient.Type.array)


@pyplant.ReactorFunc
def reactor_b(pipe: pyplant.Pipework):

    range = pipe.receive('range')
    avg = np.mean(range)

    pipe.send('avg', avg, pyplant.Ingredient.Type.simple)


@pyplant.ReactorFunc
def reactor_c(pipe: pyplant.Pipework):

    range = pipe.receive('range')
    avg = pipe.receive('avg')

    newRange = range + avg

    pipe.send('newRange', newRange, pyplant.Ingredient.Type.array)


@pyplant.ReactorFunc
def reactor_d(pipe: pyplant.Pipework):
    newRange = pipe.receive('newRange')

    print(newRange)


plant = pyplant.Plant('C:\\preloaded_data\\test')


plant.set_config({
    'element_number': 20
})
plant.add_reactors(reactor_a, reactor_b, reactor_c, reactor_d)

plant.run_reactor(reactor_d)

plant.shutdown()

