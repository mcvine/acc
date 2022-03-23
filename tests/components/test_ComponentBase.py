#!/usr/bin/env python

import pytest

import numba
from numba import cuda, void, float32
from numba.core.types import Array, Float
from mcvine.acc.components.ComponentBase import ComponentBase
from mcvine.acc import test


def test_no_propagate_raises():
    with pytest.raises(TypeError):
        # check creating a component with no propagate method raises error
        class Component(ComponentBase):
            def __init__(self, **kwargs):
                # super().__init__(__class__, **kwargs)
                return
        component = Component()


@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_propagate_no_signature_raises():
    with pytest.raises(RuntimeError):
        # check that defining a propagate function with no signature fails
        class Component(ComponentBase):
            def __init__(self, **kwargs):
                super().__init__(__class__, **kwargs)

            @cuda.jit(device=True)
            def propagate():
                pass
        component = Component()


@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_set_float():
    class Component(ComponentBase):
        def __init__(self, **kwargs):
            # super().__init__(__class__, **kwargs)
            return

        @cuda.jit(void(), device=True)
        def propagate():
            pass

    component = Component()
    Component.change_floattype("float32")

    # Both the instance and class attribute should have changed
    assert component.floattype == "float32"
    assert Component.get_floattype() == "float32"


@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_propagate_args_changed():
    # check that propagate arguments are changed from float64 -> float32
    NB_FLOAT = getattr(numba, "float64")
    class Component(ComponentBase):
        def __init__(self, **kwargs):
            #super().__init__(__class__, **kwargs)
            return

        @cuda.jit(void(NB_FLOAT, NB_FLOAT[:]), device=True)
        def propagate(x, y):
            y[0] = x + 1.0

    component = Component()
    Component.change_floattype("float32")
    assert component.floattype == "float32"

    # check that the class wide attributes are changed
    assert Component.get_floattype() == "float32"
    assert Component.process_kernel is not None
    args = Component.propagate.args
    assert len(args) == 2

    assert isinstance(args[0], Float)
    assert args[0].bitwidth == 32
    assert isinstance(args[1], Array)
    assert args[1].dtype == float32

