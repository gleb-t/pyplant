
import pyplant


@pyplant.ReactorFunc
def some_reactor(pipe: pyplant.Pipework):




plant = pyplant.Plant('C:\\preloaded_data\\test')


plant.add_reactors(some_reactor)


plant.shutdown()