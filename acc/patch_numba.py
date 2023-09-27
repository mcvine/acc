import os, subprocess as sp
import numba

this_dir = os.path.dirname(__file__)
numba_dir = os.path.dirname(numba.__file__)
patch_file = os.path.abspath(os.path.join(this_dir, "_numba_patches", f"numba-{numba.__version__}.patch"))
indent = ' '*4

def patch():
    cmd = f"patch -R -p1 -s -f --dry-run < {patch_file}"
    print(f"Running\n{indent}{cmd}\nat {numba_dir}")
    print("to check if patch was already applied")
    rt = sp.call(cmd, shell=True, cwd=numba_dir)
    if rt == 0:
        print("\nnumba patch already applied")
        return
    print("\nnumba patch not yet applied")
    cmd = f"patch -p1 < {patch_file}"
    print(f"Patching numba at {numba_dir} using {patch_file}")
    print("by running the following command")
    print(indent + cmd)
    rt = sp.call(cmd, shell=True, cwd=numba_dir)
    if rt:
        raise RuntimeError("Failed to patch numba")

def reverse_patch():
    cmd = f"patch -p1 -s -f --dry-run < {patch_file}"
    print(f"Running\n{indent}{cmd}\nat {numba_dir}")
    print("to check if patch was applied or not")
    rt = sp.call(cmd, shell=True, cwd=numba_dir)
    if rt == 0:
        print("\nnumba patch not applied yet")
        return
    print("\nnumba patch applied.")
    cmd = f"patch -R -p1 < {patch_file}"
    print(f"Reversing numba patch at {numba_dir} using {patch_file}")
    print("by running the following command")
    print(indent + cmd)
    rt = sp.call(cmd, shell=True, cwd=numba_dir)
    if rt:
        raise RuntimeError("Failed to reverse numba patch")

def main():
    import sys
    if len(sys.argv)==2 and sys.argv[1] == 'reverse':
        reverse_patch()
    elif len(sys.argv) == 1:
        patch()
    else:
        print(help)
        sys.exit(1)

help = """** mcvine.acc numba patching tool

Either run

  $ python -m mcvine.acc.patch_numba

to apply the patch, or run

  $ python -m mcvine.acc.patch_numba reverse

to reverse the patch
"""

if __name__ == '__main__': main()
