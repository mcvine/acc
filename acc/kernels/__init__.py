import numba
import numpy as np
from numba import cuda
from mcni import units

from ..components.samples import getAbsScttCoeffs

class ScatterFuncFactory:

    def render(self, kernel):
        """returns cuda device function tuple (scatter, scattering_coeff, absorb)
        absorb function is used typically by detectors to handle neutron detection.
        for other kernels, just use None
        """
        scatter, calc_scattering_coeff, absorb, calc_absorption_coeff =  kernel.identify(self)
        if calc_scattering_coeff is None or calc_absorption_coeff is None:
            mu, sigma = getAbsScttCoeffs(kernel)
            calc_scattering_coeff = calc_scattering_coeff or make_calc_sctt_coeff_func(sigma)
            calc_absorption_coeff = calc_absorption_coeff or make_calc_abs_coeff_func(mu)
        if absorb is None:
            absorb = dummy_absorb
        return scatter, calc_scattering_coeff, absorb, calc_absorption_coeff

    def onCompositeKernel(self, composite):
        elements = composite.elements()
        weights = []
        for element in elements:
            weights.append(getattr(element, 'weight', 1.))
            continue
        s = sum(weights)
        for w, element in zip(weights, elements):
            element.weight = w/s
        from .composite import makeKernelMethods
        return makeKernelMethods(composite)

    def onIsotropicKernel(self, kernel):
        from ..components.samples import getAbsScttCoeffs
        mu, sigma = getAbsScttCoeffs(kernel)
        from .isotropic import S
        @cuda.jit(device=True, inline=True)
        def isotropic_scatter(threadindex, rng_states, neutron):
            neutron[-1] *= sigma
            return S(threadindex, rng_states, neutron)
        return isotropic_scatter, None, None, None

    def onSimplePowderDiffractionKernel(self, kernel):
        from .powderdiffr import scatter, scattering_coefficient, PowderDiffraction
        # the data translation copied from mccomponents.sample.diffraction.ComputationEngineRendererExtension.onSimplePowderDiffractionKernel
        xs = kernel.cross_sections
        pd = PowderDiffraction(
            kernel.peaks, kernel.unitcell_volume,
            xs.abs, xs.inc, xs.coh
        )
        w_v, q_v, my_s_v2 = pd.w_v, pd.q_v, pd.my_s_v2
        Npeaks = pd.Npeaks
        d_phi = pd.d_phi
        ucvol = pd.unitcell_volume
        @cuda.jit(device=True)
        def simplepowderdiffraction_scatter(threadindex, rng_states, neutron):
            n = cuda.local.array(3, dtype=numba.float64)
            vtmp = cuda.local.array(3, dtype=numba.float64)
            vout = cuda.local.array(3, dtype=numba.float64)
            return scatter(
                threadindex, rng_states, neutron,
                w_v, q_v, my_s_v2, Npeaks,
                d_phi, n, vtmp, vout, ucvol,
            )
        @cuda.jit(device=True)
        def simplepowderdiffraction_scattering_coefficient(neutron):
            return scattering_coefficient(neutron, ucvol, Npeaks, q_v, my_s_v2)
        return simplepowderdiffraction_scatter, simplepowderdiffraction_scattering_coefficient, None, None

    def onConstantQEKernel(self, kernel):
        from ..components.samples import getAbsScttCoeffs
        mu, sigma = getAbsScttCoeffs(kernel)
        Q = _units_remover.remove_unit(kernel.Q, 1/units.length.angstrom)
        E = _units_remover.remove_unit(kernel.E, units.energy.meV)

        from .constant_qe import S
        @cuda.jit(device=True)
        def constantqe_scatter(threadindex, rng_states, neutron):
            neutron[-1] *= sigma
            return S(threadindex, rng_states, neutron, Q, E)

        return constantqe_scatter, None, None, None

    def onE_Q_Kernel(self, kernel):
        from ..components.samples import getAbsScttCoeffs
        mu, sigma = getAbsScttCoeffs(kernel)

        Qmin = _units_remover.remove_unit(kernel.Qmin, 1/units.length.angstrom)
        Qmax = _units_remover.remove_unit(kernel.Qmax, 1/units.length.angstrom)
        E_Q = kernel.E_Q
        S_Q = kernel.S_Q

        from .E_Q import makeS
        S = makeS(E_Q, S_Q, Qmin, Qmax, max_iter=100)
        @cuda.jit(device=True)
        def E_Q_scatter(threadindex, rng_states, neutron):
            neutron[-1] *= sigma
            return S(threadindex, rng_states, neutron)
        return E_Q_scatter, None, None, None

    def onSANS2D_ongrid_Kernel(self, kernel):
        from ..components.samples import getAbsScttCoeffs
        mu, sigma = getAbsScttCoeffs(kernel)

        S_QxQy = np.load(kernel.S_QxQy)
        Qx_min = _units_remover.remove_unit(kernel.Qx_min, 1/units.length.angstrom)
        Qx_max = _units_remover.remove_unit(kernel.Qx_max, 1/units.length.angstrom)
        Qy_min = _units_remover.remove_unit(kernel.Qy_min, 1/units.length.angstrom)
        Qy_max = _units_remover.remove_unit(kernel.Qy_max, 1/units.length.angstrom)

        from .E_Q import makeS
        S = makeS(S_QxQy, Qx_min, Qx_max, Qy_min, Qy_max)
        @cuda.jit(device=True)
        def scatter(threadindex, rng_states, neutron):
            neutron[-1] *= sigma
            return S(threadindex, rng_states, neutron)
        return scatter, None, None, None

    def onDGSSXResKernel(self, kernel):
        from ..components.samples import getAbsScttCoeffs
        mu, sigma = getAbsScttCoeffs(kernel)

        target_position = _units_remover.remove_unit(kernel.target_position, units.length.meter)
        target_radius = _units_remover.remove_unit(kernel.target_radius, units.length.meter)
        tof_target = _units_remover.remove_unit(kernel.tof_at_target, units.time.second)
        dtof = _units_remover.remove_unit(kernel.dtof, units.time.second)

        target_position = np.asarray(target_position, dtype=float)
        from .DGSSXResKernel import scatter

        @cuda.jit(device=True)
        def dgssxres_scatter(threadindex, rng_states, neutron):
            neutron[-1] *= sigma
            return scatter(threadindex, rng_states, neutron, target_position, target_radius, tof_target, dtof)
        return dgssxres_scatter, None, None, None

scatter_func_factory = ScatterFuncFactory()

def make_calc_sctt_coeff_func(sigma):
    @cuda.jit(device=True)
    def calc_scattering_coeff(neutron):
        return sigma
    return calc_scattering_coeff

from ..vec3 import length
def make_calc_abs_coeff_func(mu):
    @cuda.jit(device=True)
    def calc_absorption_coeff(neutron):
        v = length(neutron[3:6])
        return mu/v*2200
    return calc_absorption_coeff

@cuda.jit(device=True)
def dummy_absorb(threadindex, rng_states, neutron):
    return

from mccomposite.units_utils import UnitsRemover
from mccomposite import units
_units_remover = UnitsRemover(
    length_unit=units.length.meter, angle_unit=units.angle.degree)

from . import xml
