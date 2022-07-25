import numba
from numba import cuda
from mcni import units

class ScatterFuncFactory:

    def render(self, kernel):
        return kernel.identify(self)

    def onIsotropicKernel(self, kernel):
        from .isotropic import scatter
        return scatter

    def onSimplePowderDiffractionKernel(self, kernel):
        from .powderdiffr import scatter, PowderDiffraction
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
        return simplepowderdiffraction_scatter

scatter_func_factory = ScatterFuncFactory()
