#!/usr/bin/env python

from mcvine import run_script
def test():
    run_script.run1(
        "./guide_cpu_instrument.py", 'out.debug-guide_cpu_instrument',
        ncount=100000, overwrite_datafiles=True)
    return

def main():
    test()
    return

if __name__ == '__main__': main()
