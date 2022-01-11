#!/usr/bin/env python

# Copyright (c) 2021 by UT-Battelle, LLC.

from math import inf, isnan, tanh
from numba import cuda, guvectorize
from numpy import array, count_nonzero, empty, empty_like

from mcni import neutron_buffer
from mcni.AbstractComponent import AbstractComponent
from mcni.neutron_storage import neutrons_as_npyarr, ndblsperneutron
import mcvine.acc.geometry.plane as plane

category = 'optics'


# mcni.utils.conversion.v2k
@cuda.jit(device=True, inline=True)
def v2k(n):
    """
    Reimplement the McStas V2K conversion.

    Parameters:
    n (float): V

    Returns:
    float: K
    """
    mN = 1.6749286e-27
    hbar = 1.054571628e-34
    return n * 1e-10 * mN / hbar


@cuda.jit(device=True, inline=True)
def guide_construct(
        dimensions,
        R0=0.99, Qc=0.0219, alpha=6.07, m=2, W=0.003):
    """
    Initialize this Guide component.
    The guide is centered on the z-axis with the entrance at z=0.

    Parameters:
    dimensions (tuple): width, height of guide entry, exit, and guide length
    R0: low-angle reflectivity
    Qc: critical scattering vector
    alpha: slope of reflectivity
    m: m-value of material (0 is complete absorption)
    W: width of supermirror cutoff

    Returns a tuple:
    guide_nature (tuple): various characteristics of the guide
    guide_sides (tuple): the planes forming the sides of the guide
    """
    (w1, h1, w2, h2, l) = dimensions
    return ((w1, h1, R0, Qc, alpha, m, W),
            (  # entrance
             plane.plane_from_points((+w1 / 2, +h1 / 2, 0),
                                     (+w1 / 2, -h1 / 2, 0),
                                     (-w1 / 2, -h1 / 2, 0)),
               # exit
             plane.plane_from_points((+w2 / 2, +h2 / 2, l),
                                     (+w2 / 2, -h2 / 2, l),
                                     (-w2 / 2, -h2 / 2, l)),
               # sides
             plane.plane_from_points((+w1 / 2, +h1 / 2, 0),
                                     (+w1 / 2, -h1 / 2, 0),
                                     (+w2 / 2, +h2 / 2, l)),
             plane.plane_from_points((-w1 / 2, +h1 / 2, 0),
                                     (-w1 / 2, -h1 / 2, 0),
                                     (-w2 / 2, +h2 / 2, l)),
             plane.plane_from_points((+w1 / 2, +h1 / 2, 0),
                                     (-w1 / 2, +h1 / 2, 0),
                                     (+w2 / 2, +h2 / 2, l)),
             plane.plane_from_points((+w1 / 2, -h1 / 2, 0),
                                     (-w1 / 2, -h1 / 2, 0),
                                     (+w2 / 2, -h2 / 2, l))))


@cuda.jit(device=True, inline=True)
def guide_reflectivity(guide_nature, incident, reflected):
    (entrance_width, entrance_height, R0, Qc, alpha, m, W) = guide_nature
    """
    Calculate the mirror reflectivity for a neutron.

    Parameters:
    guide_nature (tuple): various characteristics of the guide
    incident (tuple): initial velocity before reflecting
    reflected (tuple): final velocity after reflecting

    Returns:
    float: the reflectivity for the neutron's given momentum change
    """
    (ix, iy, iz) = incident
    (rx, ry, rz) = reflected
    # see eq 5.2 in mcstas manual, sec 5.1.1
    Q = v2k(plane.size_of_vector((ix - rx, iy - ry, iz - rz)))
    reflectivity = R0
    if Q > Qc:
        reflectivity *= (1 - tanh((Q - m * Qc) / W)) * \
                        (1 - alpha * (Q - Qc)) / 2
    return reflectivity


@cuda.jit(device=True, inline=True)
def guide_propagate(guide_nature, guide_sides,
                    position, velocity, duration, weight):
    """
    Propagate a particle through a guide.
    If the weight returned is 0 then the content of the other
    values is undefined.

    Parameters:
    guide_nature (tuple): various characteristics of the guide
    guide_sides (tuple): the planes forming the sides of the guide
    position (tuple): x,y,z of the particle's initial position
    velocity (tuple): x,y,z of the particle's initial velocity
    duration (float): for how long the particle has traveled already
    weight (float): the particle's initial weight factor

    Returns a tuple:
    tuple: x,y,z of the particle's exit position
    tuple: x,y,z of the particle's exit velocity
    float: for how long the particle has traveled when it exits the guide
    float: the particle's weight factor on exit
    """
    (entrance_width, entrance_height, R0, Qc, alpha, m, W) = guide_nature
    (pos_curr, vel_curr) = (position, velocity)
    if vel_curr[2] <= 0:
        # The particle is not traveling from the entrance to the exit.
        return position, velocity, duration, 0
    # The first side is the entrance to the guide.
    plane_point = (guide_sides[0][0][0],
                   guide_sides[0][0][1],
                   guide_sides[0][0][2])
    plane_normal = (guide_sides[0][1][0],
                    guide_sides[0][1][1],
                    guide_sides[0][1][2])
    (pos_curr, dur_curr) = plane.plane_intersection_duration(
        plane_point, plane_normal, pos_curr, vel_curr)
    if isnan(dur_curr):
        # No intersection with the entrance plane could be determined.
        return position, velocity, duration, 0
    if dur_curr < 0 or abs(pos_curr[0]) > entrance_width / 2 or \
            abs(pos_curr[1]) > entrance_height / 2:
        # The particle does not enter the guide.
        return position, velocity, duration, 0
    duration += dur_curr

    ind_prev = inf
    while True:
        # Find which side is hit next.
        dur_min = inf
        for index in range(1, len(guide_sides)):
            if index == ind_prev:
                continue
            plane_point = (guide_sides[index][0][0],
                           guide_sides[index][0][1],
                           guide_sides[index][0][2])
            plane_normal = (guide_sides[index][1][0],
                            guide_sides[index][1][1],
                            guide_sides[index][1][2])
            (pos_next, dur_next) = plane.plane_intersection_duration(
                plane_point, plane_normal, pos_curr, vel_curr)
            if not isnan(dur_next) and 0 < dur_next < dur_min:
                ind_min = index
                (pos_min, dur_min) = (pos_next, dur_next)
        if dur_min == inf:
            # No new side was hit.
            return position, velocity, duration, 0
        duration += dur_min
        if ind_min == 1:
            # The second side is the exit from the guide.
            return pos_min, vel_curr, duration, weight
        # Update particle to be reflecting from the new side.
        pos_curr = pos_min
        vel_prev = vel_curr
        plane_normal = (guide_sides[ind_min][1][0], guide_sides[ind_min][1][1],
                        guide_sides[ind_min][1][2])
        vel_curr = plane.plane_reflect(plane_normal, vel_prev)
        ind_prev = ind_min
        # Update particle weight from reflection.
        reflectivity = guide_reflectivity(guide_nature, vel_prev, vel_curr)
        weight *= reflectivity
        if weight < 1e-10:
            # For efficiency, discard particles of very low weight.
            return position, velocity, duration, 0


@guvectorize(
    ["float32, float32, float32, float32, float32, float32, float32, "
     "float32[:, :, :], float64[:, :], float64[:, :]"],
    "(),(),(),(),(),(),(),(p,q,r),(m,n)->(m,n)",
    target="cuda")
def guide_process(entrance_width, entrance_height,
                  R0, Qc, alpha, m, W,
                  sides, neutrons_in, neutrons_out):
    """
    Vectorized kernel for propagation of neutrons through guide via GPU.
    Neutrons with a weight of 0 set in neutrons_out are absorbed and any
    other written characteristics are undefined.

    Parameters:
    entrance_width (m): width at the guide entry
    entrance_height (m): height at the guide entry
    R0: low-angle reflectivity
    Qc: critical scattering vector
    alpha: slope of reflectivity
    m: m-value of material (0 is complete absorption)
    W: width of supermirror cutoff
    sides (array): the planes forming the sides of the guide
    neutrons_in (array): neutrons to propagate through the guide
    neutrons_out (array): write target,
        returns the neutrons as they emerge from the exit of the guide,
        indexed identically as from neutrons_in
    """
    guide_nature = (entrance_width, entrance_height, R0, Qc, alpha, m, W)
    guide_sides = (((sides[0][0][0], sides[0][0][1], sides[0][0][2]),
                    (sides[0][1][0], sides[0][1][1], sides[0][1][2])),
                   ((sides[1][0][0], sides[1][0][1], sides[1][0][2]),
                    (sides[1][1][0], sides[1][1][1], sides[1][1][2])),
                   ((sides[2][0][0], sides[2][0][1], sides[2][0][2]),
                    (sides[2][1][0], sides[2][1][1], sides[2][1][2])),
                   ((sides[3][0][0], sides[3][0][1], sides[3][0][2]),
                    (sides[3][1][0], sides[3][1][1], sides[3][1][2])),
                   ((sides[4][0][0], sides[4][0][1], sides[4][0][2]),
                    (sides[4][1][0], sides[4][1][1], sides[4][1][2])),
                   ((sides[5][0][0], sides[5][0][1], sides[5][0][2]),
                    (sides[5][1][0], sides[5][1][1], sides[5][1][2])))
    for index in range(neutrons_in.shape[0]):
        position = (neutrons_in[index][0],
                    neutrons_in[index][1],
                    neutrons_in[index][2])
        velocity = (neutrons_in[index][3],
                    neutrons_in[index][4],
                    neutrons_in[index][5])
        duration = neutrons_in[index][8]
        weight = neutrons_in[index][9]
        for i in range(10):
            neutrons_out[index][i] = 7
        (position, velocity, duration, weight) = guide_propagate(
            guide_nature, guide_sides, position, velocity, duration, weight)
        neutrons_out[index][9] = weight
        if weight > 0:
            (neutrons_out[index][0],
             neutrons_out[index][1],
             neutrons_out[index][2]) = position
            (neutrons_out[index][3],
             neutrons_out[index][4],
             neutrons_out[index][5]) = velocity
            neutrons_out[index][8] = duration


@cuda.jit
def guide_construct_kernel(
        dimensions,
        R0, Qc, alpha, m, W,
        guide_nature, guide_sides):
    """
    GPU kernel wrapping guide_construct.

    Parameters:
    dimensions (tuple): width, height of guide entry, exit, and guide length
    R0: low-angle reflectivity
    Qc: critical scattering vector
    alpha: slope of reflectivity
    m: m-value of material (0 is complete absorption)
    W: width of supermirror cutoff
    guide_nature (array): write target,
        returns various characteristics of the guide
    guide_sides (array): write target,
        returns the planes forming the sides of the guide
    """
    (nature, sides) = guide_construct(
        dimensions,
        R0, Qc, alpha, m, W)
    (guide_nature[0], guide_nature[1], guide_nature[2], guide_nature[3],
     guide_nature[4], guide_nature[5], guide_nature[6]) = nature
    (((guide_sides[0][0][0], guide_sides[0][0][1], guide_sides[0][0][2]),
      (guide_sides[0][1][0], guide_sides[0][1][1], guide_sides[0][1][2])),
     ((guide_sides[1][0][0], guide_sides[1][0][1], guide_sides[1][0][2]),
      (guide_sides[1][1][0], guide_sides[1][1][1], guide_sides[1][1][2])),
     ((guide_sides[2][0][0], guide_sides[2][0][1], guide_sides[2][0][2]),
      (guide_sides[2][1][0], guide_sides[2][1][1], guide_sides[2][1][2])),
     ((guide_sides[3][0][0], guide_sides[3][0][1], guide_sides[3][0][2]),
      (guide_sides[3][1][0], guide_sides[3][1][1], guide_sides[3][1][2])),
     ((guide_sides[4][0][0], guide_sides[4][0][1], guide_sides[4][0][2]),
      (guide_sides[4][1][0], guide_sides[4][1][1], guide_sides[4][1][2])),
     ((guide_sides[5][0][0], guide_sides[5][0][1], guide_sides[5][0][2]),
      (guide_sides[5][1][0], guide_sides[5][1][1], guide_sides[5][1][2]))) = \
        sides


@cuda.jit
def guide_reflectivity_kernel(guide_nature, incident, reflected, reflectivity):
    """
    GPU kernel wrapping guide_reflectivity.

    Parameters:
    guide_nature (vector): various characteristics of the guide
    incident (vector): initial velocity before reflecting
    reflected (vector): final velocity after reflecting
    reflectivity (array): write target, returns the reflectivity
    """
    reflectivity[0] = guide_reflectivity(guide_nature, incident, reflected)


@cuda.jit
def guide_propagate_kernel(
        guide_nature, guide_sides,
        position_in, velocity_in, duration_in, weight_in,
        position_out, velocity_out, duration_out, weight_out):
    """
    GPU kernel wrapping guide_propagate.
    If weight_out is set to 0 then the content of the other write targets is
    undefined.

    Parameters:
    guide_nature (vector): various characteristics of the guide
    guide_sides (array): the planes forming the sides of the guide
    position_in (vector): the particle's initial position
    velocity_in (vector): the particle's initial velocity
    duration_in (float): for how long the particle has traveled already
    weight_in (float): the particle's initial weight factor
    position_out (array): write target, returns the particle's final position
    velocity_out (array): write target, returns the particle's final velocity
    duration_out (array): write target, returns the particle's new travel time
    weight_out (array): write target, returns the particle's final weight factor
    """
    (position, velocity, duration, weight) = \
        guide_propagate(guide_nature, guide_sides,
                        position_in, velocity_in, duration_in, weight_in)
    weight_out[0] = weight
    if weight > 0:
        (position_out[0], position_out[1], position_out[2]) = \
            (position[0], position[1], position[2])
        (velocity_out[0], velocity_out[1], velocity_out[2]) = \
            (velocity[0], velocity[1], velocity[2])
        duration_out[0] = duration


class Guide(AbstractComponent):

    def __init__(
            self, name,
            w1, h1, w2, h2, l,
            R0=0.99, Qc=0.0219, alpha=6.07, m=2, W=0.003):
        """
        Initialize this Guide component.
        The guide is centered on the z-axis with the entrance at z=0.

        Parameters:
        name (str): the name of this component
        w1 (m): width at the guide entry
        h1 (m): height at the guide entry
        w2 (m): width at the guide exit
        h2 (m): height at the guide exit
        l (m): length of guide
        R0: low-angle reflectivity
        Qc: critical scattering vector
        alpha: slope of reflectivity
        m: m-value of material (0 is complete absorption)
        W: width of supermirror cutoff
        """
        self.name = name  # TODO: pass into GPU too
        self.nature = empty(7)
        self.sides = empty((6, 2, 3))
        guide_construct_kernel[1, 1](
            (w1, h1, w2, h2, l),
            R0, Qc, alpha, m, W,
            self.nature, self.sides)

    def reflectivity(self, velocity_i, velocity_f):
        """
        Calculate the mirror reflectivity for a neutron.

        Parameters:
        velocity_i (vector): initial velocity before reflecting
        velocity_f (vector): final velocity after reflecting

        Returns:
        float: the reflectivity for the neutron's given momentum change
        """
        reflectivity = empty(1)
        guide_reflectivity_kernel[1, 1](self.nature, velocity_i, velocity_f,
                                        reflectivity)
        return reflectivity[0]

    def propagate(self, position, velocity, duration, weight):
        """
        Propagate a particle through this guide.

        Parameters:
        position (vector): x,y,z of the particle's initial position
        velocity (vector): x,y,z of the particle's initial velocity
        duration (float): for how long the particle has traveled already
        weight (float): the particle's initial weight factor

        Returns a tuple (or None if particle does not exit):
        vector: x,y,z of the particle's exit position
        vector: x,y,z of the particle's exit velocity
        float: for how long the particle has traveled when it exits the guide
        float: the particle's weight factor on exit
        """
        final_position = empty(3)
        final_velocity = empty(3)
        final_duration = empty(1)
        final_weight = empty(1)
        guide_propagate_kernel[1, 1](
            self.nature, self.sides,
            position, velocity, duration, weight,
            final_position, final_velocity, final_duration, final_weight)
        if final_weight[0] == 0:
            return None
        else:
            return (final_position, final_velocity,
                    final_duration, final_weight)

    def process(self, neutrons):
        """
        Propagate a buffer of particles through this guide.
        Adjusts the buffer to include only the particles that exit,
        at the moment of exit.

        Parameters:
        neutrons: a buffer containing the particles
        """
        (entrance_width, entrance_height, R0, Qc, alpha, m, W) = self.nature
        neutron_array = neutrons_as_npyarr(neutrons)
        neutron_array.shape = -1, ndblsperneutron
        neutrons_out = empty_like(neutron_array)
        guide_process(entrance_width, entrance_height,
                      R0, Qc, alpha, m, W,
                      self.sides, neutron_array, neutrons_out)
        mask = array(list(map(lambda weight: weight > 0, neutrons_out.T[9])),
                     dtype=bool)
        neutrons.resize(count_nonzero(mask), neutrons[0])
        neutrons.from_npyarr(neutrons_out[mask])
