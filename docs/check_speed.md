# `check_speed.py`

`check_speed.py` can be used to get the wallclock time of one or more instrument scripts over a range of neutron counts with different configurations.

`check_speed` takes either a `.json` or `.yml` configuration file as input and will generate an output text file containing a table of timings (in seconds) for each script specified in the config file.

## Example usage:

The following shows the simplest `.yml` file for running an instrument script with two different neutron counts:

`config.yml`
```
ncounts: [1e6, 1e7]
output_file: "./speeds.txt"
scripts:
  - name: "verdi_acc_run"
    file: "instruments/VERDI/VERDI_base.py"
    kwds:
      { use_gpu: true }
    acc_run: true
```

This will run the VERDI instrument script with `mcvine.acc` (GPU) components using `mcvine.acc.run_script`.

`python check_speed.py config.yml`

Running the above will output `./speed.txt` which will contain timing data similar to
```
ncount	verdi_acc_run	
1000000	29.6919	
10000000	43.5030	
```

## Configuration Options:

Below is a list of all possible configuration options with a brief description. Mandatory options are followed with a *.

- `ncounts`*: Integer array of neutron counts to run each script with.
- `output_file`*: String containing name of the output file. *This will get overwritten if it already exists.*
- `iterations`: Integer specifying how many times to run each script to get a timing. The time for each iteration is averaged to obtain the final time. Default: 1.
- `scripts`*: Collection (YAML block sequence) of scripts to run. Each script has a mandatory `file` option, and several optional ones to control how it is run:
    - `file`*: MCViNE Python instrument script
    - `name`: Sets the column name for this script in the output file. Defaults to the script name if not provided.
    - `kwds`: Defines any parameters used in the instrument script and is passed to `run_script` as `**kwargs`. This is a YAML mapping, i.e, `kwds: { key: value }`
    - `skip_for`: Optional integer array used to indicate neutron counts values for which this script should be skipped. This can be useful if a particularly slow script (such as a CPU-only run) should not be timed for high neutron counts.
    - `acc_run`: Boolean, if `true`, then `mcvine.acc.run_script` is used to run this script. `false` uses the original `mcvine.run_script`. Mutually exclusive with the `mpi` flag. Default: `false`.
    - `mpi`: Boolean, if `true`, then `mcvine.run_mpi` is used. Mutually exclusive with the `acc_run` option. Default: `false`.
    - `nodes`: Integer specifying the number of MPI nodes to be used. Only used when `mpi` is `true`, otherwise has no effect. Default: `1`.

## Advanced example:

An example configuration is shown below that runs the VERDI instrument for several configurations: GPU, CPU, and MPI with 4 and 8 nodes.

```
ncounts: [1e6, 1e7, 1e8]
iterations: 2
output_file: "./verdi_timings.txt"
scripts:
  - name: "acc_run"
    file: "instruments/VERDI/VERDI_base.py"
    kwds: { use_gpu: true,
            ntotalthreads: 524288,
            threads_per_block: 256
          }
    acc_run: true
  - name: "cpu_run"
    file: "instruments/VERDI/VERDI_base.py"
    kwds: { use_gpu: false }
    acc_run: false
    skip_for: [1e8]
  - name: "MPI_4Core"
    file: "instruments/VERDI/VERDI_base.py"
    kwds:
      { use_gpu: false }
    mpi: true
    nodes: 4
    acc_run: false
    skip_for: [ ]
  - name: "MPI_8Core"
    file: "instruments/VERDI/VERDI_base.py"
    kwds:
      { use_gpu: false }
    mpi: true
    nodes: 8
    acc_run: false
    skip_for: [ ]
```

Here `use_gpu` is an option for the `VERDI_base.py` instrument script.
