#!/usr/bin/env python
#
# Copyright (c) 2021-2022 by UT-Battelle, LLC.

from numba import cuda, void

category = 'optics'

from ...config import get_numba_floattype
NB_FLOAT = get_numba_floattype()

@cuda.jit(
    void(
        NB_FLOAT[:]
    ), device=True
)
def propagate(
        in_neutron
):
    pass


from ..ComponentBase import ComponentBase
class Arm(ComponentBase):

    def __init__(
            self, name):
        """
        Initialize this Arm component.

        Parameters:
        name (str): the name of this component
        """
        self.name = name
        self.propagate_params = ()

        # Aim a neutron at this arm to cause JIT compilation.
        import mcni
        neutrons = mcni.neutron_buffer(1)
        neutrons[0] = mcni.neutron(r=(0, 0, -1), v=(0, 0, 1), prob=1, time=0)
        self.process(neutrons)


Arm.register_propagate_method(propagate)
