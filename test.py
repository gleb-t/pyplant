
from pyplant import *
import numpy as np


@ReactorFunc
def reactor_a(pipe: Pipework):

    maxElems = pipe.read_config('element_number')

    pipe.send('range', np.arange(0, maxElems), Ingredient.Type.array)

    yield


@ReactorFunc
def reactor_b(pipe: Pipework):

    range = yield pipe.receive('range')
    avg = np.mean(range)

    pipe.send('avg', avg, Ingredient.Type.simple)


@ReactorFunc
def reactor_c(pipe: Pipework):

    range = yield pipe.receive('range')
    avg = yield pipe.receive('avg')

    newRange = range + avg

    pipe.send('newRange', newRange, Ingredient.Type.array)


@ReactorFunc
def reactor_d(pipe: Pipework):
    newRange = yield pipe.receive('newRange')
    hugeArray = yield pipe.receive('huge_array')
    doubleRange = yield pipe.receive('doubleRange')


    nonzero = np.count_nonzero(hugeArray[...])
    print(newRange)
    print(nonzero)
    print(doubleRange)


@ReactorFunc
def reactor_huge_array(pipe: Pipework):
    array = pipe.allocate('huge_array', Ingredient.Type.huge_array, shape=(100, 255, 255))
    array[...] = 1.0
    array[:50, ...] = 0.0

    pipe.send('huge_array', array, Ingredient.Type.huge_array)

    yield

@ReactorFunc
def reactor_with_subreactors(pipe: Pipework):

    doubledRange = yield from subreactor_a(pipe)

    pipe.send('doubleRange', doubledRange, Ingredient.Type.array)


@SubreactorFunc
def subreactor_a(pipe: Pipework):
    yield from subreactor_b(pipe)
    result = yield from subreactor_c(pipe)
    return result


@SubreactorFunc
def subreactor_b(pipe: Pipework):

    yield from subreactor_c(pipe)

@SubreactorFunc
def subreactor_c(pipe: Pipework):
    result = yield from subreactor_d(pipe)
    return result

@SubreactorFunc
def subreactor_d(pipe: Pipework):
    range = yield pipe.receive('range')

    return range * 2


plant = Plant('C:\\preloaded_data\\test')


plant.set_config({
    'element_number': 20
})
plant.add_reactors(reactor_a, reactor_b, reactor_c, reactor_d, reactor_huge_array, reactor_with_subreactors)

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