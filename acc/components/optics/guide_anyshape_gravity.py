# -*- python -*-
#

from math import sqrt, fabs
import numpy as np, numba
from numba import cuda, void
from mcni.utils.conversion import V2K

category = 'optics'

from ...config import get_numba_floattype
NB_FLOAT = get_numba_floattype()
from ._guide_utils import calc_reflectivity
from ...neutron import absorb
from ...geometry._utils import insert_into_sorted_list_with_indexes
from .geometry2d import inside_convex_polygon
from ... import vec3

max_bounces = 1000
max_numfaces = 100


from .guide_anyshape import calc_face_normal, get_faces_data

@cuda.jit(NB_FLOAT(NB_FLOAT[:], NB_FLOAT[:], NB_FLOAT[:], NB_FLOAT[:],  NB_FLOAT[:]),
          device=True, inline=True)
def intersect_plane(position, velocity, gravity, center, normal):
    A = vec3.dot(gravity, normal)/2
    B = vec3.dot(velocity, normal)
    C = vec3.dot(position, normal) - vec3.dot(center, normal)
    if fabs(A) < 1E-10:
        return -C/B
    B2_4AC = B*B-4*A*C
    if B2_4AC < 0: return -np.inf
    sqrt_B2_4AC = sqrt(B2_4AC)
    t1 = (-B-sqrt_B2_4AC)/2/A
    t2 = (-B+sqrt_B2_4AC)/2/A
    if t2 < t1:
       tmp=t2; t2=t1; t1=tmp
    if t1>0: return t1
    return t2

@cuda.jit(void(NB_FLOAT[:], NB_FLOAT[:], NB_FLOAT[:], NB_FLOAT, NB_FLOAT[:], NB_FLOAT[:], NB_FLOAT[:]),
          device=True, inline=True)
def propagate_with_gravity(position, velocity, gravity, t, new_position, new_velocity, tmp):
    vec3.copy(gravity, tmp)
    vec3.scale(tmp, t)
    vec3.add(velocity, tmp, new_velocity)
    vec3.add(velocity, new_velocity, tmp)
    vec3.scale(tmp, 0.5*t)
    vec3.add(position, tmp, new_position)
    return

@cuda.jit(
    void(
        NB_FLOAT[:],
        NB_FLOAT[:, :, :], NB_FLOAT[:, :], NB_FLOAT[:, :, :], NB_FLOAT[:, :, :],
        NB_FLOAT, NB_FLOAT, NB_FLOAT, NB_FLOAT, NB_FLOAT,
        NB_FLOAT[:], numba.int32[:],
        NB_FLOAT[:],
        NB_FLOAT[:], NB_FLOAT[:], NB_FLOAT[:],
    ), device=True
)
def _propagate(
        in_neutron,
        faces, centers, unitvecs, faces2d,
        R0, Qc, alpha, m, W,
        intersections, face_indexes,
        gravity,
        tmp1, tmp2, tmp3,
):
    nfaces = len(faces)
    for nb in range(max_bounces):
        # calc intersection with each face and save positive ones in a list with increasing order
        ninter = 0
        for iface in range(nfaces):
            intersection = intersect_plane(
                in_neutron[:3], in_neutron[3:6], gravity,
                centers[iface], unitvecs[iface, 2],
            )
            if intersection>0:
                ninter = insert_into_sorted_list_with_indexes(iface, intersection, face_indexes, intersections, ninter) 
        if not ninter:
            break
        # find the smallest intersection that is inside the mirror 
        found = False
        for iinter in range(ninter):
            face_index = face_indexes[iinter]
            intersection = intersections[iinter]
            # calc position and velocity at intersection
            propagate_with_gravity(in_neutron[:3], in_neutron[3:6], gravity, intersection, tmp1, tmp2, tmp3)
            # calc 2d coordinates and use it to check if it is inside the mirror
            vec3.subtract(tmp1, centers[face_index], tmp3)
            e0 = unitvecs[face_index, 0, :]
            e1 = unitvecs[face_index, 1, :]
            e2 = unitvecs[face_index, 2, :]
            face2d = faces2d[face_index]
            if inside_convex_polygon((vec3.dot(tmp3, e0), vec3.dot(tmp3, e1)), face2d):
                found = True
                break
        if not found:
            break
        t = in_neutron[-2]
        prob = in_neutron[-1]
        # propagate to intersection
        intersection -= intersection * 1E-14
        t += intersection
        in_neutron[-2] = t
        vec3.copy(tmp1, in_neutron[:3])
        vec3.copy(tmp2, in_neutron[3:6])
        #
        vq = -vec3.dot(tmp2, e2)*2
        R = calc_reflectivity(fabs(vq)*V2K, R0, Qc, alpha, m, W)
        prob *= R
        if prob <= 0:
            absorb(in_neutron)
            break
        # tmp1 = velocity change vector
        vec3.copy(e2, tmp1)
        vec3.scale(tmp1, vq)
        # change direction
        vec3.add(in_neutron[3:6], tmp1, in_neutron[3:6])
        in_neutron[-1] = prob
    return

from ..ComponentBase import ComponentBase
class Guide_anyshape_gravity(ComponentBase):

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
        faces, centers, unitvecs, faces2d = get_faces_data(geometry, xwidth, yheight, zdepth, center)
        faces2d = np.array([
            [
                [np.dot(vertex-center, ex), np.dot(vertex-center, ey)]
                for vertex in face
            ]
            for face, center, (ex,ey,ez) in zip(faces, centers, unitvecs)
        ]) # nfaces, nverticesperface, 2
        self.propagate_params = (
            faces, centers, unitvecs, faces2d,
            float(R0), float(Qc), float(alpha), float(m), float(W),
        )
        return
    
    @property
    def gravity(self):
        return self._gravity
    
    @gravity.setter
    def gravity(self, g):
        print("set my gravity to", g)
        self._gravity = g
        self.propagate_params = self.propagate_params + (g,)

    @property
    def abs_orientation(self):
        return self._abs_orientation
    
    @abs_orientation.setter
    def abs_orientation(self, orientation):
        self._abs_orientation = orientation
        self.gravity = np.dot(orientation, [0, -9.80665, 0])

    @cuda.jit(
        void(
            NB_FLOAT[:],
            NB_FLOAT[:, :, :], NB_FLOAT[:, :], NB_FLOAT[:, :, :], NB_FLOAT[:, :, :],
            NB_FLOAT, NB_FLOAT, NB_FLOAT, NB_FLOAT, NB_FLOAT,
            NB_FLOAT[:]
        ), device=True, inline=True,
    )
    def propagate(
            in_neutron,
            faces, centers, unitvecs, faces2d,
            R0, Qc, alpha, m, W,
            g,
    ):
        tmp1 = cuda.local.array(3, dtype=numba.float64)
        tmp2 = cuda.local.array(3, dtype=numba.float64)
        tmp3 = cuda.local.array(3, dtype=numba.float64)
        nfaces = len(faces)
        assert nfaces < max_numfaces
        intersections = cuda.local.array(max_numfaces, dtype=numba.float64)
        face_indexes = cuda.local.array(max_numfaces, dtype=numba.int32)
        return _propagate(
            in_neutron,
            faces, centers, unitvecs, faces2d,
            R0, Qc, alpha, m, W,
            intersections, face_indexes,
            g,
            tmp1, tmp2, tmp3,
        )
