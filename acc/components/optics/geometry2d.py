# -*- python -*-
#

import numpy as np
from math import sqrt, fabs
from numba import cuda, void

from ...config import get_numba_floattype
NB_FLOAT = get_numba_floattype()

# modified from
# https://stackoverflow.com/questions/1119627/how-to-test-if-a-point-is-inside-of-a-convex-polygon-in-2d-integer-coordinates#:~:text=If%20it%20is%20convex%2C%20a,traversed%20in%20the%20same%20order)
@cuda.jit(device=True)
def inside_convex_polygon(point, vertices, l_epsilon):
    # previous_side = get_side(v_sub(vertices[1], vertices[0]), v_sub(point, vertices[0]))
    previous_sign = cosine_sign(v_sub(vertices[1], vertices[0]), v_sub(point, vertices[0]))
    previous_side = previous_sign <= 0
    n_vertices = len(vertices)
    for n in range(1, n_vertices):
        a, b = vertices[n], vertices[(n+1)%n_vertices]
        affine_segment = v_sub(b, a)
        affine_point = v_sub(point, a)
        sign = cosine_sign(affine_segment, affine_point)
        if fabs(sign) < l_epsilon:
            continue
        # current_side = get_side(affine_segment, affine_point)
        current_side = sign <= 0
        if fabs(previous_sign) < l_epsilon:
            previous_side = current_side
            continue
        if previous_side != current_side:
            return False
    return True

@cuda.jit(device=True, inline=True)
def get_side(a, b):
    x = cosine_sign(a, b)
    return x <=0

@cuda.jit(device=True, inline=True)
def v_sub(a, b):
    return a[0]-b[0], a[1]-b[1]

@cuda.jit(device=True, inline=True)
def cosine_sign(a, b):
    return a[0]*b[1]-a[1]*b[0]

