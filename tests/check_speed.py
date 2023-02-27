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


def run(instrument_script, ncount, niters, mpi=False, nodes=1, acc_run=False, skip_ncounts=[], **kwds):
    outdir = 'out.debug-check-speed'
    if os.path.exists(outdir):
        shutil.rmtree(outdir)

    # TODO: this stdout file gets overwritten on each call, move or use append?
    stdout = open("./check-speed-stdout.txt", "w")
    sys.stderr = stdout

    avg_times = []
    for n in ncount:
        if n in skip_ncounts:
            # insert element to preserve ordering
            avg_times.append(np.nan)
            continue

        print(" Running '{}' with n={}".format(instrument_script, n))

        times = []
        # redirect script output to reduce terminal clutter
        for iter in range(niters + 1):
            sys.stdout = stdout
            buffer_size = int(n) if n > 1e6 else int(1e6)
            if buffer_size > 1e9:
                buffer_size = int(1e9)

            # get the runtime for the script. Note: this probably isn't the
            # best timing, since it will include the overhead of launching the
            # script and any file IO done inside the script, but there is no
            # current way to time individual components.
            # TODO: the stdout could be parsed to pull the process timings for acc components
            # TODO: there is still mcvine component output, likely from a subprocess in which this redirection has no effect
            if mpi:
                # run script in MPI mode
                # NOTE: with mcvine=0.44, pyyaml 5.3 must be used to avoid yaml load error. This is fixed in newer versions of mcvine.
                time_before = time.time_ns()
                run_script.run_mpi(instrument_script, outdir, nodes=nodes, buffer_size=buffer_size, ncount=n,
                                   overwrite_datafiles=True, **kwds)
                time_after = time.time_ns()
            elif acc_run:
                # use accelerated run script
                time_before = time.time_ns()
                run_acc_script.run(
                    instrument_script, outdir, buffer_size=buffer_size, ncount=n,
                    overwrite_datafiles=True, **kwds)
                time_after = time.time_ns()
            else:
                # use default run script (mcvine, single node)
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


def parse_array_opt(array):
    tmp = array
    if isinstance(array, str):
        tmp = np.fromstring(array, dtype=np.float, sep=',')
    # convert sci notation number (e.g, 1e5) to int
    output = []
    for n in tmp:
        output.append(int(float(n)))
    return output


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
        ncounts = parse_array_opt(opts["ncounts"])

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

        mpi = False
        nodes = 1
        acc_run = False
        name = script_file
        if "mpi" in script:
            mpi = script["mpi"]
        if mpi and "nodes" in script:
            nodes = int(script["nodes"])
            if nodes <= 0:
                raise RuntimeError("Nodes count must be >= 1")
        if "acc_run" in script:
            acc_run = script["acc_run"]

        if mpi and acc_run:
            raise RuntimeError("Cannot run script in MPI mode and accelerated mode, only one must be enabled.")

        if "name" in script:
            # use this name for the output instead of the script filename
            name = script["name"]

        skip = []
        if "skip_for" in script:
            skip = parse_array_opt(script["skip_for"])

        try:
            runs[name] = run(script["file"], ncounts, iters, mpi, nodes, acc_run, skip, **kwds)
        except Exception as e:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            print("Error running {}-'{}': {}: {}".format(name, script["file"], type(e), e))

    result_file = open(output_file, "w")
    header = "ncount\t"
    for key in runs.keys():
        header += "{}\t".format(key)
    header += "\n"
    result_file.write(header)

    for i in range(len(ncounts)):
        n = ncounts[i]
        line = "{:1.0e}\t".format(n)
        for r in runs:
            line += "{:.4f}\t".format(runs[r][i] * 1e-9)
        line += "\n"
        result_file.write(line)
    result_file.close()

    print("Wrote timing results to '{}'".format(output_file))


if __name__ == '__main__':
    main()
