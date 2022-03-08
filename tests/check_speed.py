#!/usr/bin/env python

import json
import numpy as np
import os
import shutil
import sys
import time
import warnings
import yaml

# ignore the FutureWarnings given from histogram when saving hdf output
# set the environment variable since these occur in the child process launched
# by the run_script call
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["PYTHONWARNINGS"] = "ignore"

from mcvine import run_script
from mcvine.acc import run_script as run_acc_script

thisdir = os.path.dirname(__file__)


def run(instrument_script, ncount, niters, **kwds):
    outdir = 'out.debug-check-speed'
    if os.path.exists(outdir):
        shutil.rmtree(outdir)

    # TODO: this stdout file gets overwritten on each call, move or use append?
    stdout = open("./check-speed-stdout.txt", "w")
    sys.stderr = stdout

    avg_times = []
    for n in ncount:
        print(" Running '{}' with n={}".format(instrument_script, n))

        times = []
        # redirect script output to reduce terminal clutter
        for iter in range(niters + 1):
            sys.stdout = stdout
            buffer_size = int(n) if n > 1e6 else int(1e6)

            # get the runtime for the script. Note: this probably isn't the
            # best timing, since it will include the overhead of launching the
            # script and any file IO done inside the script, but there is no
            # current way to time individual components.
            # TODO: the stdout could be parsed to pull the process timings for acc components
            # TODO: there is still mcvine component output, likely from a subprocess in which this redirection has no effect
            time_before = time.time_ns()
            run_script.run1(
                instrument_script, outdir, buffer_size=buffer_size, ncount=n,
                overwrite_datafiles=True, **kwds)
            time_after = time.time_ns()

            # skip first run
            if iter < 1:
                continue

            delta = time_after - time_before
            times.append(delta)

        sys.stdout = sys.__stdout__

        time_avg = np.sum(np.asarray(times)) / len(times)
        avg_times.append(time_avg)
        print(" TIME = {} ms ({} s)".format(time_avg * 1e-6, time_avg * 1e-9))

    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    stdout.close()

    return avg_times


def main():
    ncounts = [1e6]
    iters = 1

    # usage: check_speed.py input_file
    if len(sys.argv) < 2:
        print("Expected an input config file")
        exit(-1)
    filename = sys.argv[1]

    if filename.endswith(".json"):
        opts = json.load(open(filename, 'r'))
    elif filename.endswith(".yml") or filename.endswith(".yaml"):
        opts = yaml.safe_load(open(filename, 'r'))
    else:
        print("Unrecognized input file type, expected .json, .yml, or .yaml")
        exit(-1)

    if "output_file" not in opts:
        raise RuntimeError("Expected an 'output_file' entry")
    output_file = opts["output_file"]

    if "ncounts" in opts:
        if isinstance(opts["ncounts"], str):
            ncounts_tmp = np.fromstring(opts["ncounts"], dtype=np.float,
                                        sep=',')
        else:
            ncounts_tmp = opts["ncounts"]
        ncounts = []
        for n in ncounts_tmp:
            ncounts.append(int(float(n)))

    if "iterations" in opts:
        iters = int(opts["iterations"])

    if "scripts" not in opts:
        raise RuntimeError("Expected a 'scripts' entry in the input file")

    runs = dict()
    scripts = opts["scripts"]
    for script in scripts:
        kwds = dict()
        if "file" not in script:
            raise RuntimeError("Expected script entry to have a 'file' entry")
        if "kwds" in script and len(script["kwds"]) > 0:
            kwds = dict(script["kwds"])

        # check for any duplicate script names
        script_file = script["file"]
        i = 2
        while script_file in runs:
            # if we already have the same name, append a number to the results
            script_file += "_{}".format(i)

        runs[script_file] = run(script["file"], ncounts, iters, **kwds)

    result_file = open(output_file, "w")
    header = "ncount\t"
    for key in runs.keys():
        header += "{}\t".format(key)
    header += "\n"
    result_file.write(header)

    for i in range(len(ncounts)):
        n = ncounts[i]
        line = "{}\t".format(n)
        for r in runs:
            line += "{:.4f}\t".format(runs[r][i] * 1e-9)
        line += "\n"
        result_file.write(line)
    result_file.close()

    print("Wrote timing results to '{}'".format(output_file))


if __name__ == '__main__':
    main()
