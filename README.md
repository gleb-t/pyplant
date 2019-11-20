# PyPlant
> Out-of-core function pipelines


## Introduction
PyPlant's goal is to simplify writing data processing pipelines. 
It helps avoid re-running expensive early stages of the pipeline, when only the later stages have changed.

Given a set of Python functions that consume and produce data, it automatically runs them in a correct order and caches intermediate results. 
When the pipeline is executed again, only the necessary parts are re-run.

Here is a minimal example. We load a large data array and then want to plot some examples:
```python
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

We structure the code as two PyPlant "reactors", so if we change the plotting function or configuration and re-run, 
only the plotting will be re-executed, because the data loading part has not changed.


## Motivation

Why another Python pipeline/workflow library? 
When writing ML-related scripts, it is often a pain that the whole script needs to re-run, when only a small cosmetic change was made.
So every time the final plot is tweaked, the data is re-loaded and the model is re-trained.


In fact, this is so common that there is a zoo of various data pipelining libraries out there.
However, I still haven't found something that simple, automatic and can handle large data sizes.
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

TODO

## Walk through

TODO

