import numba
from numba import cuda
from mcni import units

class ScatterFuncFactory:

    def render(self, kernel):
        """returns function tuple (scatter, scattering_coeff, absorb)
        absorb function is used typically by detectors to handle neutron detection.
        for other kernels, just use None
        """
        return kernel.identify(self)

    def onIsotropicKernel(self, kernel):
        from ..components.samples import getAbsScttCoeffs
        mu, sigma = getAbsScttCoeffs(kernel)
        from .isotropic import S
        @cuda.jit(device=True, inline=True)
        def isotropic_scatter(threadindex, rng_states, neutron):
            neutron[-1] *= sigma
            return S(threadindex, rng_states, neutron)
        return isotropic_scatter, None, None

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
        return simplepowderdiffraction_scatter, simplepowderdiffraction_scattering_coefficient, None

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

        return constantqe_scatter, None, None

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
        return E_Q_scatter, None, None


scatter_func_factory = ScatterFuncFactory()

from mccomposite.units_utils import UnitsRemover
from mccomposite import units
_units_remover = UnitsRemover(
    length_unit=units.length.meter, angle_unit=units.angle.degree)
