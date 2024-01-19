# -*- python -*-
#

import numpy as np
from math import sqrt
from numba import cuda, void
from mcni.utils.conversion import V2K

category = 'optics'

from ...config import get_numba_floattype
NB_FLOAT = get_numba_floattype()
from ._guide_utils import calc_reflectivity
from ...neutron import absorb

max_bounces = 100000


from ..ComponentBase import ComponentBase
class Guide_anyshape(ComponentBase):

    def __init__(
            self,
            name,
            xwidth=0, yheight=0, zdepth=0, center=False,
            R0=0.99, Qc=0.0219, alpha=3, m=2, W=0.003,
            geometry="",
            **kwargs):
        """
        Initialize this Guide component.
        The guide is centered on the z-axis with the entrance at z=0.

        Parameters:
        name (str): the name of this component
        xwidth (m): resize the bounding box on X axis
        yheight (m): resize the bounding box on y axis
        zdepth (m): resize the bounding box on z axis
        center (boolean): object will be centered w.r.t. the local coord system if true

        R0: low-angle reflectivity
        Qc: critical scattering vector
        alpha: slope of reflectivity
        m: m-value of material (0 is complete absorption)
        W: width of supermirror cutoff

        geometry (str): path of the OFF/PLY geometry file for the guide shape
        """
        self.name = name
        self.propagate_params = (
            float(ww), float(hh), float(hw1), float(hh1), float(l),
            float(R0), float(Qc), float(alpha), float(m), float(W),
        )

        # Aim a neutron at the side of this guide to cause JIT compilation.
        import mcni
        velocity = ((w1 + w2) / 2, 0, l / 2)
        neutrons = mcni.neutron_buffer(1)
        neutrons[0] = mcni.neutron(r=(0, 0, 0), v=velocity, prob=1, time=0)
        self.process(neutrons)

    @cuda.jit(
        void(
            NB_FLOAT[:],
            NB_FLOAT, NB_FLOAT, NB_FLOAT, NB_FLOAT, NB_FLOAT,
            NB_FLOAT, NB_FLOAT, NB_FLOAT, NB_FLOAT, NB_FLOAT,
        ), device=True
    )
    def propagate(
            in_neutron,
            ww, hh, hw1, hh1, l,
            R0, Qc, alpha, m, W,
    ):
        x, y, z, vx, vy, vz = in_neutron[:6]
        t = in_neutron[-2]
        prob = in_neutron[-1]
        # propagate to z=0
        dt = -z / vz
        x += vx * dt;
        y += vy * dt;
        z = 0.;
        t += dt
        in_neutron[:6] = x, y, z, vx, vy, vz
        in_neutron[-2] = t
        in_neutron[-1] = prob

def load_scaled_centered_faces(path, xwidth=0, yheight=0, zdepth=0, center=False):
    from . import offio
    vertices, faces = offio.load(path)
    assert len(vertices.shape) == 2
    nvertices, _ = vertices.shape
    assert _ == 3
    assert len(faces.shape) == 2
    nfaces, n_vertices_per_face = faces.shape
    # scale and center
    ratiox = ratioy = ratioz = 0
    resize = True
    cx = cy = cz = 0
    xmax = np.max(vertices[:, 0])
    xmin = np.min(vertices[:, 0])
    cx = (xmin+xmax)/2
    if xwidth:
        ratiox = xwidth/(xmax-xmin)
    ymax = np.max(vertices[:, 1])
    ymin = np.min(vertices[:, 1])
    cy = (ymin+ymax)/2
    if yheight:
        ratioy = yheight/(ymax-ymin)
    zmax = np.max(vertices[:, 2])
    zmin = np.min(vertices[:, 2])
    cz = (zmin+zmax)/2
    if zdepth:
        ratioz = zdepth/(zmax-zmin)
    ratios = ratiox, ratioy, ratioz
    nratios = sum(int(bool(r)) for r in ratios)
    if nratios == 2:
        raise ValueError("Please provide all of xwidth, yheight, zdepth, or none of them, or one of them")
    if nratios == 1:
        ratiox = ratioy = ratioz = sum(r for r in ratios if r)
    if nratios == 0:
        ratiox = ratioy = ratioz = 1
    vertices[:, 0] = (vertices[:, 0]-cx)*ratiox + (0 if center else cx)
    vertices[:, 1] = (vertices[:, 1]-cy)*ratioy + (0 if center else cy)
    vertices[:, 2] = (vertices[:, 2]-cz)*ratioz + (0 if center else cz)
    faces = np.array([[vertices[i] for i in face] for face in faces])
    return faces
