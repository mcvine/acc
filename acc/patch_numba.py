import os, subprocess as sp
import numba

this_dir = os.path.dirname(__file__)
numba_dir = os.path.dirname(numba.__file__)

def main():
    patch_file = os.path.abspath(os.path.join(this_dir, "_numba_patches", f"numba-{numba.__version__}.patch"))
    cmd = f"patch -R -p1 -s -f --dry-run < {patch_file}"
    # print(f"Running {cmd} at {numba_dir}")
    rt = sp.call(cmd, shell=True, cwd=numba_dir)
    if rt == 0:
        print("numba patch already applied")
        return
    cmd = f"patch -p1 < {patch_file}"
    print(f"Patching numba at {numba_dir} using {patch_file}")
    print("by running the following command")
    print(cmd)
    rt = sp.call(cmd, shell=True, cwd=numba_dir)
    if rt:
        raise RuntimeError("Failed to patch numba")

if __name__ == '__main__': main()
