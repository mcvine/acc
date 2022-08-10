import numba
from numba import cuda
from mcni import units

class ScatterFuncFactory:

    def render(self, kernel):
        "returns function tuple (scatter, scattering_coeff)"
        return kernel.identify(self)

    def onIsotropicKernel(self, kernel):
        from ..components.samples import getAbsScttCoeffs
        mu, sigma = getAbsScttCoeffs(kernel)
        from .isotropic import S
        @cuda.jit(device=True, inline=True)
        def isotropic_scatter(threadindex, rng_states, neutron):
            neutron[-1] *= sigma
            return S(threadindex, rng_states, neutron)
        return isotropic_scatter, None

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
        return simplepowderdiffraction_scatter, simplepowderdiffraction_scattering_coefficient

    def onConstantQEKernel(self, kernel):
        from ..components.samples import getAbsScttCoeffs
        mu, sigma = getAbsScttCoeffs(kernel)
        from .constant_qe import S
        @cuda.jit(device=True)
        def constantqe_scatter(threadindex, rng_states, neutron):
            neutron[-1] *= sigma
            return S(threadindex, rng_states, neutron, kernel.Q, kernel.E)

        return constantqe_scatter, None


scatter_func_factory = ScatterFuncFactory()
