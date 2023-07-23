#!/usr/bin/env python
#

from .KernelNode import KernelNode as base, debug

class SANS2D_ongrid_Kernel(base):

    tag = "SANS2D_ongrid_Kernel"

    def createKernel( self, **kwds ):
        Qx_min = self._parse( kwds['Qx_min'] )
        Qx_max = self._parse( kwds['Qx_max'] )
        Qy_min = self._parse( kwds['Qy_min'] )
        Qy_max = self._parse( kwds['Qy_max'] )
        import numpy as np
        S_QxQy = np.load( kwds['S_QxQy'] )
        from ...SANS2D_ongrid import SANS2D_ongrid_Kernel as f
        return f(S_QxQy, Qx_min, Qx_max, Qy_min, Qy_max)

    pass # end of SANS2D_ongrid_Kernel


from .HomogeneousScatterer import HomogeneousScatterer
HomogeneousScatterer.onSANS2D_ongrid_Kernel = HomogeneousScatterer.onKernel

# End of file
