# -*- python -*-
#
# Jiao Lin <jiao.lin@gmail.com>
#

import numpy as np, numba
from numba import cuda, void, int64, int32
from numba.cuda.random import xoroshiro128p_type
from ..._numba import xoroshiro128p_uniform_float32
from math import sqrt, exp

from ... import test
from .SampleBase import SampleBase
from ...neutron import absorb, prop_dt_inplace, clone, is_moving
from ... import vec3
from ...geometry.arrow_intersect import max_intersections
from ...geometry.location import inside

from numba.core import config
if not config.ENABLE_CUDASIM:
    from numba.cuda.compiler import Dispatcher, DeviceFunction

from ...config import get_numba_floattype
NB_FLOAT = get_numba_floattype()


category = 'samples'

@cuda.jit(device=True)
def dummy_absorb(neutron):
    return

def factory(shape, kernel, max_scattered_neutrons=10, max_ms_loops=3, max_ms_loops_path1=2, minimum_neutron_event_probability=1e-20):
    """
    Usage
    -----
    * create a new python module with code to define a new scatterer class
      that inherits from a base class created using this factory method.
      see tests/components/samples/HSS_isotropic_sphere.py for an example
      - ...load shape and kernel...
      - HSSbase = factory(shape = shape, kernel = None)
      - class HSS(HSSbase): pass

    Parameters
    ----------
    max_scattered_neutrons: int
      max number of scattered neutrons to compute. this is used to set the limit of neutron cache
    max_ms_loops: int
      max number of multiple scattering loops running interactM_path1
    max_ms_loops_path1: int
      max number of multiple scattering loops in first encounter when the neutron never left the scatterer
    """
    # worst case it can be (max_ms_loops_path1+1) ^ max_ms_loops but that is almost impossible to happen
    assert max_scattered_neutrons > max_ms_loops*(max_ms_loops_path1 + 1)
    from ...geometry import arrow_intersect
    intersect = arrow_intersect.arrow_intersect_func_factory.render(shape)
    locate = arrow_intersect.locate_func_factory.render(shape)
    from ...geometry import propagation
    propagators = propagation.makePropagateMethods(intersect, locate)
    from . import getAbsScttCoeffs
    mu, sigma = getAbsScttCoeffs(kernel)
    from ...kernels import scatter_func_factory
    scatter, calc_scattering_coeff, absorb = scatter_func_factory.render(kernel)
    if calc_scattering_coeff is None:
        @cuda.jit(device=True)
        def calc_scattering_coeff(neutron):
            return sigma
    if absorb is None:
        absorb = dummy_absorb

    # propagate methods
    propagate_to_next_incident_surface = propagators['propagate_to_next_incident_surface']
    propagate_to_next_exiting_surface = propagators['propagate_to_next_exiting_surface']
    tof_before_first_exit = propagators['tof_before_first_exit']
    is_exiting = propagators['is_exiting']
    propagate_out = propagators['propagate_out']

    @cuda.jit(NB_FLOAT(NB_FLOAT[:]), device=True, inline=True)
    def _calc_atten_coeff(neutron):
        sigma = calc_scattering_coeff(neutron)
        vx,vy,vz = neutron[3:6]
        v = sqrt(vx*vx+vy*vy+vz*vz)
        mu1 = mu/v*2200
        return sigma+mu1

    @cuda.jit(void(NB_FLOAT[:]), device=True)
    def _propagate_to_next_exiting_surface_with_attenuation(neutron):
        vx, vy, vz = neutron[3:6]
        tof = tof_before_first_exit(neutron)
        v = sqrt(vx*vx+vy*vy+vz*vz)
        distance = tof*v
        atten = _calc_atten_coeff(neutron)
        transmission_prob = exp(-atten * distance)
        propagate_to_next_exiting_surface(neutron)
        neutron[-1] *= transmission_prob
        return

    @cuda.jit(
        int32(int64, xoroshiro128p_type[:], NB_FLOAT[:, :], NB_FLOAT[:]
    ) , device=True)
    def _interactM1(threadindex, rng_states, out_neutrons, neutron):
        """neutron interact with the scatterer once and generate scattered and transmitted neutrons.
        also run the absorption mechanism.
        out_neutrons should be at least size of 2
        """
        x, y, z, vx, vy, vz = neutron[:6]
        loc = locate(x,y,z)
        if loc!= inside:
            propagate_to_next_incident_surface(neutron)
        tof = tof_before_first_exit(neutron)
        v = sqrt(vx*vx+vy*vy+vz*vz)
        distance = tof*v
        sigma = calc_scattering_coeff(neutron)
        mu1 = mu/v*2200
        transmission_prob = exp(-(mu1+sigma) * distance)
        absorption_prob = (1-transmission_prob)*(mu1/(mu1+sigma))
        # transmission
        transmitted = out_neutrons[0]
        clone(neutron, transmitted)
        propagate_to_next_exiting_surface(transmitted)
        transmitted[-1] *= transmission_prob
        # absorption
        temp = out_neutrons[1]
        clone(neutron, temp)
        temp[-1] *= absorption_prob
        absorb(temp)
        # scattering
        rand = xoroshiro128p_uniform_float32(rng_states, threadindex)
        x = distance*rand
        prob = distance * exp( -(mu1+sigma) * x );
        scattered = out_neutrons[1]
        clone(neutron, scattered)
        scattered[-1] *= prob
        prop_dt_inplace(scattered, x/v)
        scatter(threadindex, rng_states, scattered)
        # packing_factor
        return 2

    @cuda.jit(
        int32(int64, xoroshiro128p_type[:], NB_FLOAT[:, :], NB_FLOAT[:],
              NB_FLOAT[:, :], NB_FLOAT[:, :], NB_FLOAT[:, :]),
        device=True
    )
    def _interactM_path1(
            threadindex, rng_states, out_neutrons, neutron,
            to_be_scattered, to_be_scattered2, scattered,
    ):
        """all interactions between a neutron and a scatterer during its first encounter,
        the neutron never left the scatterer
        """
        # neutron too weak, bail out
        if neutron[-1] < minimum_neutron_event_probability: return 0
        # init neutron array to be scattered
        N_to_be_scattered = 1
        clone(neutron, to_be_scattered[0])
        out_index = 0
        for iloop in range(max_ms_loops_path1):
            scattered2_index = 0 # index for to_be_scattered2
            for ineutron in range(N_to_be_scattered):
                neutron1 = to_be_scattered[ineutron]
                nscattered1 = _interactM1(threadindex, rng_states, scattered, neutron1)
                for iscattered in range(nscattered1):
                    scattered1 = scattered[iscattered]
                    if scattered1[-1] < minimum_neutron_event_probability: continue
                    # save neutron at border or outside to output neutron array
                    x,y,z = scattered1[:3]
                    if locate(x,y,z) != inside:
                        clone(scattered1, out_neutrons[out_index])
                        out_index+=1
                        if out_index >= min(max_scattered_neutrons, len(out_neutrons)):
                            return out_index
                        continue
                    # neutron inside shape need to be scattered again
                    clone(scattered1, to_be_scattered2[scattered2_index])
                    scattered2_index+=1
                    if scattered2_index >= max_scattered_neutrons:
                        break
                    continue
                if scattered2_index >= max_scattered_neutrons:
                    break
            # swap to_be_scattered and to_be_scattered2
            tmp = to_be_scattered2
            to_be_scattered2 = to_be_scattered
            to_be_scattered = tmp
            N_to_be_scattered = scattered2_index
            if not N_to_be_scattered: break
        # leftover neutrons not yet scattered
        for i in range(N_to_be_scattered):
            if out_index >= min(max_scattered_neutrons, len(out_neutrons)):
                break
            neutron = to_be_scattered[i]
            clone(neutron, out_neutrons[out_index])
            out_index+=1
        return out_index

    if not config.ENABLE_CUDASIM:
        @cuda.jit(
            int32(int64, xoroshiro128p_type[:], NB_FLOAT[:, :], NB_FLOAT[:]),
            device=True
        )
        def interactM_path1(
                threadindex, rng_states, out_neutrons, neutron,
        ):
            # temp data
            to_be_scattered = cuda.local.array((max_scattered_neutrons, 10), dtype=numba.float64)
            to_be_scattered2 = cuda.local.array((max_scattered_neutrons, 10), dtype=numba.float64)
            scattered = cuda.local.array((2, 10), dtype=numba.float64)
            return _interactM_path1(
                threadindex, rng_states, out_neutrons, neutron,
                to_be_scattered, to_be_scattered2, scattered,
            )
    else:
        def interactM_path1(
                threadindex, rng_states, out_neutrons, neutron,
        ):
            # temp data
            to_be_scattered = np.zeros((max_scattered_neutrons, 10), dtype=float)
            to_be_scattered2 = np.zeros((max_scattered_neutrons, 10), dtype=float)
            scattered = np.zeros((2, 10), dtype=float)
            return _interactM_path1(
                threadindex, rng_states, out_neutrons, neutron,
                to_be_scattered, to_be_scattered2, scattered,
            )

    @cuda.jit(
        int32(int64, xoroshiro128p_type[:], NB_FLOAT[:, :], NB_FLOAT[:],
              NB_FLOAT[:, :], NB_FLOAT[:, :], NB_FLOAT[:, :]),
        device=True
    )
    def _scatterM(
            threadindex, rng_states, out_neutrons, neutron,
            to_be_scattered, to_be_scattered2, scattered):
        """scatter a neutron to multiple output neutrons"""
        # init neutron array to be scattered
        N_to_be_scattered = 1
        clone(neutron, to_be_scattered[0])
        out_index = 0
        for iloop in range(max_ms_loops):
            scattered2_index = 0 # index for to_be_scattered2
            for ineutron in range(N_to_be_scattered):
                neutron1 = to_be_scattered[ineutron]
                nscattered1 = interactM_path1(threadindex, rng_states, scattered, neutron1)
                for iscattered in range(nscattered1):
                    scattered1 = scattered[iscattered]
                    if scattered1[-1] < minimum_neutron_event_probability: continue
                    if not is_moving(scattered1): continue
                    # if neutron is still inside propagate it out
                    x,y,z = scattered1[:3]
                    if locate(x,y,z) == inside:
                        _propagate_to_next_exiting_surface_with_attenuation(scattered1)
                    # if neutron is exiting, then we are done
                    if is_exiting(scattered1):
                        clone(scattered1, out_neutrons[out_index])
                        out_index+=1
                        if out_index >= min(max_scattered_neutrons, len(out_neutrons)):
                            return out_index
                        continue
                    # neutron that still has intersections with the shape need to be scattered again
                    clone(scattered1, to_be_scattered2[scattered2_index])
                    scattered2_index+=1
                    if scattered2_index >= max_scattered_neutrons:
                        break
                    continue
                if scattered2_index >= max_scattered_neutrons:
                    break
            # swap to_be_scattered and to_be_scattered2
            tmp = to_be_scattered2
            to_be_scattered2 = to_be_scattered
            to_be_scattered = tmp
            N_to_be_scattered = scattered2_index
            if not N_to_be_scattered: break
        # leftover neutrons not yet scattered
        for i in range(N_to_be_scattered):
            if out_index >= min(max_scattered_neutrons, len(out_neutrons)):
                break
            neutron = to_be_scattered[i]
            x,y,z = neutron[:3]
            propagate_out(neutron)
            xo,yo,zo = neutron[:3]
            dist = sqrt((xo-x)*(xo-x)+(yo-y)*(yo-y)+(zo-z)*(zo-z))
            atten = _calc_atten_coeff(neutron)
            transmission_prob = exp(-atten * dist)
            neutron[-1]*= transmission_prob
            clone(neutron, out_neutrons[out_index])
            out_index+=1
        return out_index

    if not config.ENABLE_CUDASIM:
        @cuda.jit(
            int32(int64, xoroshiro128p_type[:], NB_FLOAT[:, :], NB_FLOAT[:]),
            device=True
        )
        def scatterM(threadindex, rng_states, out_neutrons, neutron):
            """scatter a neutron to multiple output neutrons"""
            # temp data
            to_be_scattered = cuda.local.array((max_scattered_neutrons, 10), dtype=numba.float64)
            to_be_scattered2 = cuda.local.array((max_scattered_neutrons, 10), dtype=numba.float64)
            scattered = cuda.local.array((max_scattered_neutrons, 10), dtype=numba.float64)
            return _scatterM(
                threadindex, rng_states, out_neutrons, neutron,
                to_be_scattered, to_be_scattered2, scattered,
            )
    else:
        @cuda.jit(
            int32(int64, xoroshiro128p_type[:], NB_FLOAT[:, :], NB_FLOAT[:]),
            device=True
        )
        def scatterM(threadindex, rng_states, out_neutrons, neutron):
            """scatter a neutron to multiple output neutrons"""
            # temp data
            to_be_scattered = np.zeros((max_scattered_neutrons, 10), dtype=float)
            to_be_scattered2 = np.zeros((max_scattered_neutrons, 10), dtype=float)
            scattered = np.zeros((max_scattered_neutrons, 10), dtype=float)
            return _scatterM(
                threadindex, rng_states, out_neutrons, neutron,
                to_be_scattered, to_be_scattered2, scattered,
            )
    class HomogeneousMultipleScatterer(SampleBase):

        is_multiscattering = True
        NUM_MULTIPLE_SCATTER = max_scattered_neutrons

        def __init__(self, name):
            """
            Initialize the isotropicbox component.

            Parameters:
            name (str): the name of this component
            """
            self.name = name
            self.propagate_params = ()

            # Aim neutrons toward the sample to cause JIT compilation.
            import mcni
            neutrons = mcni.neutron_buffer(1)
            neutrons[0] = mcni.neutron(r=(0, 0, -1), v=(0, 0, 1), prob=1, time=0)
            self.process(neutrons)

        @cuda.jit(void(int64, xoroshiro128p_type[:], NB_FLOAT[:, :], NB_FLOAT[:]) , device=True)
        def propagate(threadindex, rng_states, out_neutrons, in_neutron):
            #x, y, z, vx, vy, vz = in_neutron[:6]
            scatterM(threadindex, rng_states, out_neutrons, in_neutron)
            return

        @cuda.jit(void(int64, xoroshiro128p_type[:], NB_FLOAT[:], NB_FLOAT[:, :]), device=True)
        def propagateM(threadindex, rng_states, in_neutron, out_neutrons):
            x, y, z, vx, vy, vz = in_neutron[:6]
            return
    HomogeneousMultipleScatterer._interactM1 = _interactM1
    HomogeneousMultipleScatterer.interactM_path1 = interactM_path1
    HomogeneousMultipleScatterer.scatterM = scatterM
    HomogeneousMultipleScatterer.max_scattered_neutrons = max_scattered_neutrons
    HomogeneousMultipleScatterer.max_ms_loops = max_ms_loops
    HomogeneousMultipleScatterer.max_ms_loops_path1 = max_ms_loops_path1
    HomogeneousMultipleScatterer.minimum_neutron_event_probability = minimum_neutron_event_probability
    return HomogeneousMultipleScatterer
