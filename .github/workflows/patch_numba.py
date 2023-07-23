import os, subprocess as sp
import numba

this_dir = os.path.dirname(__file__)
numba_dir = os.path.dirname(numba.__file__)

patch_file = os.path.join(this_dir, "numba-0.53.1.patch")

cmd = f"patch -p1 < {patch_file}"
rt = sp.call(cmd, shell=True, cwd=numba_dir)
if rt:
    raise RuntimeError("Failed to patch numba")
