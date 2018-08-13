import inspect

from pyplant import *
from pyplant.test import PipeworkMock


def run_reactor_to_completion(reactorFunc, pipework: Pipework):
    """
    A helper method for running a single isolated reactor to the end.
    Based on the PyPlant's Plant class.

    :param reactorFunc:
    :param pipework:
    :return:
    """
    generator = reactorFunc(pipework)
    if not inspect.isgenerator(generator):
        raise RuntimeError("Reactor '{}' is not a generator function!".format(reactorFunc.__name__))

    valueToSend = None
    while True:
        try:
            returnedObject = generator.send(valueToSend)
            valueToSend = None

            if type(returnedObject) is Plant.IngredientAwaitedCommand:
                raise RuntimeError("Reactor requested a missing ingredient: {}"
                                   .format(returnedObject.ingredientName))
            elif type(returnedObject) is Plant.SubreactorStartedCommand:
                # A sub-reactor is being launched. Remember the dependency and continue execution.
                subreactorName = returnedObject.subreactorName
                print("Reactor '{}' is starting subreactor '{}'.".format(reactorFunc.__name__, subreactorName))
            else:
                # A reactor successfully fetched an ingredient, no need to pause, just use it.
                valueToSend = returnedObject
        except StopIteration:
            break

    return


def run_subreactor_to_completion(subreactorGen, pipework: Pipework):
    @ReactorFunc
    def subreactor_wrapper(pipe: Pipework):
        yield from subreactorGen

    pipework = PipeworkMock({}, {})
    run_reactor_to_completion(subreactor_wrapper, pipework)
