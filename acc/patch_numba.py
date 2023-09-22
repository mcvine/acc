import os, subprocess as sp
import numba

this_dir = os.path.dirname(__file__)
numba_dir = os.path.dirname(numba.__file__)

def main():
    patch_file = os.path.abspath(os.path.join(this_dir, "_numba_patches", f"numba-{numba.__version__}.patch"))
    cmd = f"patch -p1 < {patch_file}"
    print(f"Patching numba at {numba_dir} using {patch_file}")
    rt = sp.call(cmd, shell=True, cwd=numba_dir)
    if rt:
        raise RuntimeError("Failed to patch numba")

if __name__ == '__main__': main()
