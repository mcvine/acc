#!/usr/bin/env python

import pytest

import numba
from numba import cuda, void, float32, float64
from numba.core.types import Array, Float
from mcvine.acc.components.ComponentBase import ComponentBase
#from mcvine.acc.neutron import absorb
from mcvine.acc import test


@cuda.jit(float64(float64), device=True)
def global_kernel(x):
    return x

@cuda.jit(float64(float64), device=True)
def nested_global_kernel(x):
    y = 1.0 + global_kernel(x)
    return y


def test_no_propagate_raises():
    with pytest.raises(TypeError):
        # check creating a component with no propagate method raises error
        class Component(ComponentBase):
            def __init__(self, **kwargs):
                # super().__init__(__class__, **kwargs)
                return
        component = Component()


@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
@pytest.mark.skipif(int(numba.__version__.split(".")[1]) >= 56, reason="Only for Numba <56")
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
@pytest.mark.skipif(int(numba.__version__.split(".")[1]) >= 56, reason="Only for Numba <56")
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


@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_propagate_global_function_changed():
    # check that propagate arguments are changed from float64 -> float32
    NB_FLOAT = getattr(numba, "float64")
    class Component(ComponentBase):
        def __init__(self, **kwargs):
            return

        @cuda.jit(void(NB_FLOAT, NB_FLOAT[:]), device=True)
        def propagate(x, y):
            y[0] = global_kernel(x)

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

    # check that the global kernel function args are changed
    args = global_kernel.args
    assert len(args) == 1

    assert isinstance(args[0], Float)
    assert args[0].bitwidth == 32


@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_propagate_nested_global_function_changed():
    # check that propagate arguments are changed from float64 -> float32
    NB_FLOAT = getattr(numba, "float64")
    class Component(ComponentBase):
        def __init__(self, **kwargs):
            return

        @cuda.jit(void(NB_FLOAT, NB_FLOAT[:]), device=True)
        def propagate(x, y):
            y[0] = nested_global_kernel(x)

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

    # check that the nested kernel function args are changed
    args = nested_global_kernel.args
    assert len(args) == 1

    assert isinstance(args[0], Float)
    assert args[0].bitwidth == 32


@pytest.mark.skipif(not test.USE_CUDA, reason='No CUDA')
def test_propagate_local_function_changed():
    # check that propagate arguments are changed from float64 -> float32
    NB_FLOAT = getattr(numba, "float64")
    @cuda.jit(NB_FLOAT(NB_FLOAT, NB_FLOAT), device=True)
    def helper_kernel(x, y):
        return x * y

    class Component(ComponentBase):
        def __init__(self, **kwargs):
            return

        @cuda.jit(void(NB_FLOAT, NB_FLOAT[:]), device=True)
        def propagate(x, y):
            y[0] = helper_kernel(x, x)

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

    # check that the local kernel function args are changed
    args = helper_kernel.args
    assert len(args) == 2
    for arg in args:
        assert isinstance(arg, Float)
        assert arg.bitwidth == 32
