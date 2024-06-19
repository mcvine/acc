# -*- python -*-
#

import logging
logger = logging.getLogger(__name__)
from math import sqrt, fabs
import numpy as np, numba
from numba import cuda, void
from mcni.utils.conversion import V2K

category = 'optics'

from ...config import get_numba_floattype
NB_FLOAT = get_numba_floattype()
from ._guide_utils import calc_reflectivity
from ...neutron import absorb, clone, prop_dt_inplace
from ...geometry._utils import insert_into_sorted_list_with_indexes
from .geometry2d import inside_convex_polygon
from ... import vec3

max_bounces = 30
max_numfaces = 5000
t_epsilon = 1E-14
l_epsilon = 1E-11

@cuda.jit(NB_FLOAT(NB_FLOAT[:], NB_FLOAT[:], NB_FLOAT[:], NB_FLOAT[:, :],  NB_FLOAT[:]),
          device=True, inline=True)
def intersect_plane(position, velocity, center, plane_unitvecs, rtmp):
    vec3.subtract(center, position, rtmp)
    normal = plane_unitvecs[2, :]
    dist = vec3.dot(rtmp, normal)
    vtmp = vec3.dot(velocity, normal)
    return dist/vtmp

def calc_face_normal(face):
    seg1 = face[1]-face[0]
    seg2 = face[2]-face[0]
    n = np.cross(seg1, seg2)
    return n/np.linalg.norm(n)

def calc_face_unit_vectors(face):
    seg1 = face[1]-face[0]
    seg2 = face[2]-face[0]
    n = np.cross(seg1, seg2)
    e3 = n/np.linalg.norm(n)
    e1 = seg1/np.linalg.norm(seg1)
    e2 = np.cross(e3, e1)
    return np.array([e1, e2, e3])

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
    elif nratios == 1:
        ratiox = ratioy = ratioz = sum(r for r in ratios if r)
    elif nratios == 0:
        ratiox = ratioy = ratioz = 1
    vertices[:, 0] = (vertices[:, 0]-cx)*ratiox + (0 if center else cx)
    vertices[:, 1] = (vertices[:, 1]-cy)*ratioy + (0 if center else cy)
    vertices[:, 2] = (vertices[:, 2]-cz)*ratioz + (0 if center else cz)
    faces = np.array([[vertices[i] for i in face] for face in faces])
    return faces

@cuda.jit(numba.boolean(NB_FLOAT[:], NB_FLOAT[:],  NB_FLOAT[:, :], NB_FLOAT[:, :]),
          device=True, inline=True)
def likely_inside_face(postmp, face_center, face_uvecs, face2d_bounds):
    vec3.subtract(postmp, face_center, postmp)
    ex = face_uvecs[0] 
    ey = face_uvecs[1] 
    x = vec3.dot(ex, postmp)
    y = vec3.dot(ey, postmp)
    return (
        (x>face2d_bounds[0, 0]-l_epsilon) and (x<face2d_bounds[1, 0]+l_epsilon)
        and (y>face2d_bounds[0, 1]-l_epsilon) and (y<face2d_bounds[1, 1]+l_epsilon)
    )  

@cuda.jit(numba.boolean(NB_FLOAT[:], NB_FLOAT[:],  NB_FLOAT[:]),
          device=True, inline=True)
def on_the_other_side(pos, face_center, face_norm):
    l0 = vec3.dot(face_center, face_norm)
    l = vec3.dot(pos, face_norm)
    # logger.debug(f"  on_the_other_side: pos={pos}, face_center={face_center}, face_norm={face_norm}, l={l}, l0={l0}")
    if l*l0 < 0: return False
    if abs(l0) < abs(l):
        # import pdb; pdb.set_trace()
        return True
    return False

@cuda.jit(
    void(
        NB_FLOAT[:],
        NB_FLOAT[:, :, :], NB_FLOAT[:, :], NB_FLOAT[:, :, :], NB_FLOAT[:, :, :], NB_FLOAT[:, :, :],
        NB_FLOAT, NB_FLOAT, NB_FLOAT, NB_FLOAT, NB_FLOAT,
        NB_FLOAT[:], numba.int32[:],
        NB_FLOAT[:], NB_FLOAT[:], numba.int32[:],
    ), device=True
)
def _propagate(
        in_neutron,
        faces, centers, unitvecs, faces2d, bounds2d,
        R0, Qc, alpha, m, W,
        intersections, face_indexes,
        tmpv3, tmp_neutron, tmp_face_hist,
):
    nfaces = len(faces)
    # logger.debug(f"in_neutron={in_neutron}, faces {faces}")
    N_tmp_face_hist = len(tmp_face_hist)
    face_hist_start_ind = 0
    face_hist_size = 0
    for nb in range(max_bounces):
        # logger.debug(f"bounce {nb}: in_neutron={in_neutron}, faces {faces}")
        # calc intersection with each face and save positive ones in a list with increasing order
        ninter = 0
        for iface in range(nfaces):
            found_in_history = False
            # logger.debug(f"  face {iface}, {face_hist_start_ind}, {face_hist_start_ind+face_hist_size}")
            for ifh in range(face_hist_start_ind, face_hist_start_ind+face_hist_size):
                # logger.debug(f"  face {iface}, {ifh}, {tmp_face_hist[ifh]}")
                if iface == tmp_face_hist[ifh]:
                    found_in_history = True
                    break
            # logger.debug(f"  face {iface} of {nfaces} faces: found_in_history={found_in_history}")
            if found_in_history: continue
            x0 = in_neutron[:3]; v0 = in_neutron[3:6]
            face_center = centers[iface]
            face_uvecs = unitvecs[iface]
            intersection = intersect_plane( x0, v0, face_center, face_uvecs, tmpv3)
            # logger.debug(f"  face {iface}: x0={x0}, v0={v0}, face_center={face_center}, face_uvecs={face_uvecs}, intersection={intersection}")
            # if intersection<=-t_epsilon:
            # if intersection<0:
            if intersection<=-t_epsilon/5:
                continue
            face2d_bounds = bounds2d[iface]
            clone(in_neutron, tmp_neutron)
            prop_dt_inplace(tmp_neutron, intersection)
            vec3.copy(tmp_neutron[:3], tmpv3)
            if not likely_inside_face(tmpv3, face_center, face_uvecs, face2d_bounds):
                continue
            # logger.debug(f"  face {iface}: intersection likely inside face")
            vec3.copy(tmp_neutron[:3], tmpv3)
            if intersection < t_epsilon and on_the_other_side(tmpv3, face_center, face_uvecs[2]):
                continue
            # logger.debug(f"  face {iface}: intersection on the right side")
            ninter = insert_into_sorted_list_with_indexes(iface, intersection, face_indexes, intersections, ninter) 
            # logger.debug(f"  face {iface}: new intersection list {intersections[:ninter]} for faces {face_indexes[:ninter]}")
        # logger.debug(f"bounce {nb}: ninter={ninter}")
        # import pdb; pdb.set_trace()
        if not ninter:
            break
        # find the smallest intersection that is inside the mirror 
        found = -1
        for iinter in range(ninter):
            face_index = face_indexes[iinter]
            intersection = intersections[iinter]
            # calc position of intersection
            vec3.copy(in_neutron[3:6], tmpv3)
            vec3.scale(tmpv3, intersection)
            vec3.add(tmpv3, in_neutron[:3], tmpv3)
            # calc 2d coordinates and use it to check if it is inside the mirror
            vec3.subtract(tmpv3, centers[face_index], tmpv3)
            e0 = unitvecs[face_index, 0, :]
            e1 = unitvecs[face_index, 1, :]
            e2 = unitvecs[face_index, 2, :]
            face2d = faces2d[face_index]
            # logger.debug(f"check intersection {iinter}: face={face_center}, {face_uvecs[2]}, intersection={intersection}")
            if inside_convex_polygon((vec3.dot(tmpv3, e0), vec3.dot(tmpv3, e1)), face2d, l_epsilon):
                found = face_index
                # logger.debug(f"check intersection {iinter}: found={found}")
                to_travel_after_bounce = l_epsilon/vec3.length(in_neutron[3:6]) # move a tiny distance
                if iinter<ninter-1:
                    next_intersection = intersections[iinter+1]
                    to_travel_after_bounce = min((next_intersection-intersection)/2, to_travel_after_bounce)
                break
        if found < 0:
            break
        if intersection<t_epsilon:
            if face_hist_size < N_tmp_face_hist:
                tmp_face_hist[(face_hist_start_ind+face_hist_size) % N_tmp_face_hist] = found
                face_hist_size += 1
            else:
                face_hist_start_ind = (face_hist_start_ind+1) % N_tmp_face_hist
                tmp_face_hist[(face_hist_start_ind-1) % N_tmp_face_hist] = found
        else:
            tmp_face_hist[face_hist_start_ind] = found
            face_hist_size = 1
            
        x, y, z, vx, vy, vz = in_neutron[:6]
        t = in_neutron[-2]
        prob = in_neutron[-1]
        # propagate to intersection
        # intersection -= intersection * 1E-14
        x += vx * intersection
        y += vy * intersection
        z += vz * intersection
        t += intersection
        #
        vq = -vec3.dot(in_neutron[3:6], e2)*2
        R = calc_reflectivity(fabs(vq)*V2K, R0, Qc, alpha, m, W)
        prob *= R
        if prob <= 0:
            absorb(in_neutron)
            break
        # tmpv3 = velocity change vector
        vec3.copy(e2, tmpv3)
        vec3.scale(tmpv3, vq)
        # change direction
        vec3.add(in_neutron[3:6], tmpv3, in_neutron[3:6])
        in_neutron[:3] = x, y, z
        in_neutron[-2] = t
        in_neutron[-1] = prob
        prop_dt_inplace(in_neutron, to_travel_after_bounce)
        # logger.debug(f"bounce {nb}: out_neutron={in_neutron}")
    return

def get_faces_data(geometry, xwidth, yheight, zdepth, center):
    faces = load_scaled_centered_faces(geometry, xwidth=xwidth, yheight=yheight, zdepth=zdepth, center=center)
    nfaces = len(faces)
    centers = faces.mean(axis=1)
    unitvecs = np.array([calc_face_unit_vectors(f) for f in faces]) # nfaces, 3, 3
    faces2d = np.array([
        [
            [np.dot(vertex-center, ex), np.dot(vertex-center, ey)]
            for vertex in face
        ]
        for face, center, (ex,ey,ez) in zip(faces, centers, unitvecs)
    ]) # nfaces, nverticesperface, 2
    bounds2d = np.array([
        [np.min(face2d, axis=0), np.max(face2d, axis=0)]
        for face2d in faces2d
    ]) # nfaces, 2, 2. bounds2d[iface][0]: minx, miny; bounds2[iface][1]: maxx, maxy
    return faces, centers, unitvecs, faces2d, bounds2d

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
        faces, centers, unitvecs, faces2d, bounds2d = get_faces_data(geometry, xwidth, yheight, zdepth, center)
        assert len(faces) < max_numfaces
        self.propagate_params = (
            faces, centers, unitvecs, faces2d, bounds2d,
            float(R0), float(Qc), float(alpha), float(m), float(W),
        )

        # Aim a neutron at the side of this guide to cause JIT compilation.
        import mcni
        velocity = (0, 0, 1000)
        neutrons = mcni.neutron_buffer(1)
        neutrons[0] = mcni.neutron(r=(0, 0, 0), v=velocity, prob=1, time=0)
        self.process(neutrons)

    @cuda.jit(
        void(
            NB_FLOAT[:],
            NB_FLOAT[:, :, :], NB_FLOAT[:, :], NB_FLOAT[:, :, :], NB_FLOAT[:, :, :], NB_FLOAT[:, :, :],
            NB_FLOAT, NB_FLOAT, NB_FLOAT, NB_FLOAT, NB_FLOAT,
        ), device=True, inline=True,
    )
    def propagate(
            in_neutron,
            faces, centers, unitvecs, faces2d, bounds2d,
            R0, Qc, alpha, m, W,
    ):
        intersections = cuda.local.array(max_numfaces, dtype=numba.float64)
        face_indexes = cuda.local.array(max_numfaces, dtype=numba.int32)
        tmpv3 = cuda.local.array(3, dtype=numba.float64)
        tmp_neutron = cuda.local.array(10, dtype=numba.float64)
        tmp_face_hist = cuda.local.array(2, dtype=numba.int32)
        return _propagate(
            in_neutron, faces, centers, unitvecs, faces2d, bounds2d,
            R0, Qc, alpha, m, W,
            intersections, face_indexes,
            tmpv3,  tmp_neutron, tmp_face_hist,
        )
