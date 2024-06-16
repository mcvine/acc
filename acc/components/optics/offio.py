# -*- python -*-

import os
import warnings
import numpy as np

def load(path):
    lines = open(path, 'rt').readlines()
    lines = [l.strip() for l in lines if not l.strip().startswith('#')]
    assert lines[0].strip() == 'OFF'
    nvertices, nfaces, _ = list(map(int, lines[1].split()))
    vertices = []
    for iv in range(nvertices):
        line = lines[2+iv]
        v = list(map(float, line.split()))
        vertices.append(v)
    faces = []
    nvertices_per_face = None
    for i in range(nfaces):
        line = lines[2+nvertices+i]
        numbers = list(map(float, line.split()))
        nv = numbers[0]
        assert int(nv) == nv
        nv = int(nv)
        if nvertices_per_face is None:
            nvertices_per_face = nv
        else:
            assert nvertices_per_face == nv
        face = numbers[1:]
        if nv != len(face):
            msg = f"Extra parameters disregarded: {face[nv:]}"
            warnings.warn(msg)
            face = face[:nv]
        iface = [int(i) for i in face]
        assert iface == face
        faces.append(iface)
    return np.array(vertices), np.array(faces)
