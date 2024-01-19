# -*- python -*-

import os
import numpy as np

def load(path):
    lines = open(path, 'rt').readlines()
    assert lines[0].strip() == 'OFF'
    nvertices, nfaces, _ = list(map(int, lines[1].split()))
    vertices = []
    for iv in range(nvertices):
        line = lines[2+iv]
        v = list(map(float, line.split()))
        vertices.append(v)
    faces = []
    for i in range(nfaces):
        line = lines[2+nvertices+i]
        numbers = list(map(int, line.split()))
        nv = numbers[0]
        face = [vertices[iv] for iv in numbers[1:]]
        assert nv == len(face)
        faces.append(face)
    return np.array(faces)
