# Development

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
