#!/usr/bin/env python
#
# Copyright (c) 2021-2022 by UT-Battelle, LLC.

from numba import cuda, void

category = 'optics'

from ...config import get_numba_floattype
NB_FLOAT = get_numba_floattype()


from ..ComponentBase import ComponentBase
class Arm(ComponentBase):

    def __init__(
            self, name, **kwargs):
        """
        Initialize this Arm component.

        Parameters:
        name (str): the name of this component
        """
        super().__init__(__class__, **kwargs)

        self.name = name
        self.propagate_params = ()

        # Aim a neutron at this arm to cause JIT compilation.
        import mcni
        neutrons = mcni.neutron_buffer(1)
        neutrons[0] = mcni.neutron(r=(0, 0, -1), v=(0, 0, 1), prob=1, time=0)
        self.process(neutrons)

    @cuda.jit(
        void(
            NB_FLOAT[:]
        ), device=True
    )
    def propagate(
            in_neutron
    ):
        pass

