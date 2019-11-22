# PyPlant
> Out-of-core function pipelines

[![Build Status](https://travis-ci.com/gleb-t/pyplant.svg?branch=master)](https://travis-ci.com/gleb-t/pyplant)

## Introduction
PyPlant's goal is to simplify writing data processing pipelines. 
It helps avoid re-running expensive early stages of the pipeline, when only the later stages have changed.

Given a set of Python functions that consume and produce data, it automatically runs them in a correct order and caches intermediate results. 
When the pipeline is executed again, only the necessary parts are re-run.

Here is a minimal example. We load a large data array and then want to plot some examples:
```python
from pyplant import *

@ReactorFunc
def load_data(pipe):
    data = expensive_operation()
    
    pipe.send('data', data, Ingredient.Type.array)

@ReactorFunc
def plot_data(pipe, config):
    data = yield pipe.receive('data')
    
    plot_examples(data, config.color)

with Plant('/tmp/plant') as plant:
    plant.run_reactor(plot_data)
```

We structure the code as two PyPlant "reactors", so if we change the plotting function or the configuration and re-run, 
only the plotting will be re-executed, because the data loading part has not changed.


## Motivation

Why another Python pipeline/workflow library? 
When writing ML-related scripts, it is often a pain that the whole script needs to re-run, when only a small cosmetic change was made.
So every time the final plot is tweaked, the data is re-loaded and the model is re-trained.


In fact, this is so common that there is a zoo of various data pipelining libraries out there.
However, I still haven't found something that's simple, automatic and can handle large data sizes.
Here's a list of "wants" that PyPlant is made to fulfill:
  - **Simple**
    Quick to learn, no custom language and workflow design programs. Start prototyping right away.
  - **DRY**
    Function code is metadata. No need to write execution graphs or external metadata. It just works.
  - **Automatic**
    No need to manually re-run outdated parts.
  - **Large data**
    Handle data that doesn't fit into memory. Persist between runs.


## Installation

Install from PyPI
```bash
pip install pyplant
```
or clone directly from Github:
```bash
pip install git+https://github.com/gleb-t/pyplant.git
```

## The Guide


### Plant
The `Plant` object needs to be created in order to run code with PyPlant. 
It manages metadata, schedules execution and stores/loads ingredients as needed. 
The only required parameter for a plant is `plantDir`, which specifies the location where ingredients and plant metadata will be stored.
Usually, each project will have its own unique `plantDir`.

The recommended way to create a `Plant` is using the `with` statement, making sure that the plant will be gracefully terminated:
```python
with Plant('/tmp/pyplant/my-project') as plant:
    # Run reactors here.
```

### Reactors

Reactors are units of execution in PyPlant. To specify that a function is a PyPlant reactor, use the `@ReactorFunc` decorator.
Reactors must have one positional argument, through which a `Pipework` object will be provided:
```python
@ReactorFunc
def plot_data(pipe: Pipework):
    # ...
```
`Pipework` is a small object that is used to send and receive ingredients (more below). To run the reactor, construct a plant and call 
```python
plant.run_reactor(plot_data)
```
Most of the time you should run the reactors that produce the final artifacts you're interested in (images, metrics, etc.) and let PyPlant decide when to run the rest of the pipeline.

Technically, before running any reactors, all reactors need to be added to the plant. 
However, all `@ReactorFunc` functions are automatically registered when you run the first reactor.
If this is not desired (e.g. you manage several different plants), add the reactors explicitly before running any:
```python
plant.add_reactor([load_data, plot_data])
```


### Ingredients

Ingredients are the data/objects that are sent and received by the reactors. 
PyPlant tracks ingredient dependencies to schedule reactor execution.
To receive an ingredient use `ingredient = yield pipe.receive('name')`.
The yield keyword is there to pause the reactor execution in case the ingredient is not yet ready.
When an ingredient is sent to the plant, note that its type has to be provided, affecting how it will be serialized and stored:
`pipe.send('name', ingredient, Ingredient.Type.object)`


Here are the possible types:
- `simple` Primitive Python types: `int`, `str`, etc.
- `list`   Python list of primitive type objects.
- `object` Any Python object that can be pickled.
- `array`  NumPy array.
- `hdf_array` HDF array. Needs to be allocated. H5Py needs to be installed.
- `keras_model` Keras model, written using `save_model()`. Keras needs to be installed.
- `file`        File stored in the plant dir, passed as a path. Needs to be allocated.

Some ingredient types need to be allocated first, because PyPlant manages their creation.
Additional arguments can be provided, depending on the type. Here is an example for an HDF-array:

```python
hdf_array = pipe.allocate('hdf-array', Ingredient.Type.hdf_array,
                          shape=(500, 1000, 1000), dtype=np.float32)
```

### Configuration

PyPlant can help manage project configuration and track reactor dependencies on configuration parameters.
So, when a parameter is changed, only the affected reactors will be re-ran and not the whole plant.
The configuration can be either a `dict` or a `ConfigBase`-derived object.
It should be provided to the plant before running the reactors:
```python
plant.set_config(config)
```
When using a `dict` config, any reactor can read parameters through its `Pipework`:
```python
color = pipe.read_config('color')
```
Using the `ConfigBase` object (recommended), the config should be declared as a class inheriting from `ConfigBase`:
```python
class MyConfig(ConfigBase):

    def __init__(self):
        super(self).__init__()
        self.color = 'red'
```
Any reactor can declare a second argument (besides the pipe) where it will receive the config object:
```python
@ReactorFunc
def plot_data(pipe, config):
    color = config.color
```
Upon reading a configuration parameter the plant will automatically register the dependency. 
Object-based configurations are superior to `dict` because of their cleaner syntax and lack of "magic strings" that hinder code analysis tools.

Sometimes, parameters shouldn't trigger re-execution (e.g. temporary dirs, logging level). 
This can be done by marking the parameter as auxiliary to the plant or in the configuration constructor:
```python
    # In the plant:
    plant.mark_params_as_auxiliary(['log-level'])
    # ... or in the config constructor:
    def __init__(self):
        # ...
        self.mark_auxiliary(['log-level'])
```

### Subreactors

Subreactors are reusable procedures whose code is also tracked by PyPlant.
They can be called from reactors to make sure, that changes to their code trigger re-execution of the dependent reactor.
So, project code that is prone to implementation changes (keeping the caller code the same) should be wrapped into subreactors.
Furthermore, subreactors are also generators managed by PyPlant and can receive and wait for ingredients, just like reactors.

A subreactor function has to be decorated with `@SubreactorFunc`. 
To call a subreactor, use the `yield from` keyword: 

```python 
returnValue = yield from my_subreactor(argA, argB)
```


Here's is a complete example. Notice, that subreactors can be nested:
```python
@SubreactorFunc
def subreactor_a(pipe: Pipework):
    return 3 + 10  # Do stuff.

@SubreactorFunc
def subreactor_b(pipe: Pipework):
    value = yield from subreactor_a(pipe)
    return value

@ReactorFunc
def reactor_a(pipe: Pipework):
    value = yield from subreactor_b(pipe)
    pipe.send('value', value, Ingredient.Type.simple)
```


