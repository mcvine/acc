import os
thisdir = os.path.abspath(os.path.dirname(__file__))
import numpy as np

def make_sans2d_data(plot=False):
    Qx_min=-0.0047176
    Qx_max=0.0047176
    Qy_min=-0.0035176
    Qy_max=0.0035176

    nx = 1475
    ny = 1100

    dQx = (Qx_max-Qx_min)/nx
    Qx = np.arange(Qx_min+dQx/2, Qx_max, dQx)

    dQy = (Qy_max-Qy_min)/ny
    Qy = np.arange(Qy_min+dQy/2, Qy_max, dQy)

    Qxgrid, Qygrid = np.meshgrid(Qx, Qy)

    Q = np.sqrt(Qxgrid*Qxgrid + Qygrid*Qygrid)

    I = np.sin(Q*3000)**2 * np.exp(-Q*300)

    if plot:
        from matplotlib import pyplot as plt
        plt.figure()
        plt.pcolormesh(Qxgrid, Qygrid, I)
        plt.colorbar()
        plt.show()

    path = os.path.join(thisdir, "./sampleassemblies/sans2d/I.npy")
    np.save(path, I)
    return
