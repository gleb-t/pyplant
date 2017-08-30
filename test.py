
import pyplant
import numpy as np


@pyplant.ReactorFunc
def reactor_a(pipe: pyplant.Pipework):

    maxElems = pipe.read_config('element_number')

    pipe.send('range', np.arange(0, maxElems), pyplant.Ingredient.Type.array)

    yield


@pyplant.ReactorFunc
def reactor_b(pipe: pyplant.Pipework):

    range = yield pipe.receive('range')
    avg = np.mean(range)

    pipe.send('avg', avg, pyplant.Ingredient.Type.simple)


@pyplant.ReactorFunc
def reactor_c(pipe: pyplant.Pipework):

    range = yield pipe.receive('range')
    avg = yield pipe.receive('avg')

    newRange = range + avg

    pipe.send('newRange', newRange, pyplant.Ingredient.Type.array)


@pyplant.ReactorFunc
def reactor_d(pipe: pyplant.Pipework):
    newRange = yield pipe.receive('newRange')
    hugeArray = yield pipe.receive('huge_array')


    nonzero = np.count_nonzero(hugeArray[...])
    print(newRange)
    print(nonzero)


@pyplant.ReactorFunc
def reactor_huge_array(pipe: pyplant.Pipework):
    array = pipe.allocate('huge_array', pyplant.Ingredient.Type.huge_array, shape=(100, 255, 255))
    array[...] = 1.0
    array[:50, ...] = 0.0


    pipe.send('huge_array', array, pyplant.Ingredient.Type.huge_array)

    yield


plant = pyplant.Plant('C:\\preloaded_data\\test')


plant.set_config({
    'element_number': 20
})
plant.add_reactors(reactor_a, reactor_b, reactor_c, reactor_d, reactor_huge_array)

plant.run_reactor(reactor_d)

plant.shutdown()

# def simple_generator(a):
#     print(a)
#     a = yield a
#     print(a)
#     a = yield a
#     print(a)
#     a = yield a
#     print(a)
#
# gen = simple_generator(1)
# val = None
# while True:
#     try:
#         val = gen.send(val * 2 if val is not None else val)
#     except StopIteration:
#         print('Stop.')
#         break