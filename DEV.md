# Development

## Implement a new component
```
# define a propagate method. first argument must be neutron
# other paramters must match that defined in the component
# __init__ method
@cuda.jit()
def propagate(neutron, param1, param2, ...):

# must inherint from appropriate base class
# Normally it is mcvine.acc.components.ComponentBase.ComponentBase
# For components needing random numbers, use
# mcvine.acc.components.StochasticComponentBase.StochasticComponentBase
# For source components, use
# mcvine.acc.components.sources.SourceBase.SourceBase
# For example
from mcvine.acc.components.ComponentBase import ComponentBase
class NewComponent(ComponentBase):

    def __init__(self, name, *args):
        self.propagate_params = [param1_value, param2_value, ...]
#
# after class definition, register the propagate method
NewComponent.register_propagate_method(propagate)
```

## Run acc instrument
To run a mcvine instrument with all components being acc components, 
instead of using `mcvine.run_script.run*` methods, use `mcvine.acc.run_script.run`.

To run a mcvine instrument with a mixture of acc and non-acc components, or
all non-acc components, use `mcvine.run_script.run*` methods.

## Testing

If a test only works with CUDA, use the following decorator:

```
import pytest
from mcvine.acc.test import USE_CUDA
@pytest.mark.skipif(not USE_CUDA, reason='No CUDA')
```

If a test only works with CUDASIM, use the following decorator:

```
import pytest
from mcvine.acc.test import USE_CUDASIM
@pytest.mark.skipif(not USE_CUDASIM, reason='no CUDASIM')
```
