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

        # check if S_QxQy is a .npy or .h5 file
        if kwds['S_QxQy'].endswith(".npy"):
            import numpy as np
            S_QxQy = np.load( kwds['S_QxQy'] )
        elif kwds['S_QxQy'].endswith(".h5"):
            import h5py
            input_file = h5py.File(kwds['S_QxQy'], 'r')
            # Load the intensity dataset "I" from file
            if "I" not in input_file.keys():
                raise RuntimeError("Error loading intensity from S_QxQy file, could not find dataset 'I'")
            S_QxQy = input_file["I"][()]
            input_file.close()
        else:
            raise RuntimeError("Unrecognized file type for S_QxQy. Expected .npy or .h5")

        from ...SANS2D_ongrid import SANS2D_ongrid_Kernel as f
        return f(S_QxQy, Qx_min, Qx_max, Qy_min, Qy_max)

    pass # end of SANS2D_ongrid_Kernel


from .HomogeneousScatterer import HomogeneousScatterer
HomogeneousScatterer.onSANS2D_ongrid_Kernel = HomogeneousScatterer.onKernel

# End of file
