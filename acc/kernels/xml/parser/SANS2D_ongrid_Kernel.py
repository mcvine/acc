#!/usr/bin/env python
#

import numpy as np
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
            S_QxQy = np.load( kwds['S_QxQy'] )
        elif kwds['S_QxQy'].endswith(".h5"):
            import h5py
            input_file = h5py.File(kwds['S_QxQy'], 'r')

            # check that the file contains the necessary data
            datasets = ['I', 'qx', 'qy']
            for dataset in datasets:
                if dataset not in input_file.keys():
                    raise RuntimeError(
                        "Error loading from S_QxQy file, could not find dataset '{}'".format(dataset))

            # Load the intensity dataset 'I', and 'qx', 'qy' from file
            S_QxQy = input_file['I'][()]
            Qx = input_file['qx'][()]
            Qy = input_file['qy'][()]
            input_file.close()

            # make sure the Qx/Qy arrays match the intensity size
            assert S_QxQy.shape[1] == len(Qx)
            assert S_QxQy.shape[0] == len(Qy)

            # overwrite the Qx,Qy min and max parameters from the input file
            Qx_min = np.min(Qx)
            Qx_max = np.max(Qx)
            Qy_min = np.min(Qy)
            Qy_max = np.max(Qy)
        else:
            raise RuntimeError("Unrecognized file type for S_QxQy. Expected .npy or .h5")

        from ...SANS2D_ongrid import SANS2D_ongrid_Kernel as f
        return f(S_QxQy, Qx_min, Qx_max, Qy_min, Qy_max)

    pass # end of SANS2D_ongrid_Kernel


from .HomogeneousScatterer import HomogeneousScatterer
HomogeneousScatterer.onSANS2D_ongrid_Kernel = HomogeneousScatterer.onKernel

# End of file
