[![CI](https://github.com/mcvine/acc/actions/workflows/CI.yml/badge.svg)](https://github.com/mcvine/acc/actions/workflows/CI.yml)

# acc
Accelerated mcvine engine

## Develop

Install mcvine
```
$ conda config --add channels conda-forge 
$ conda config --add channels mcvine 
$ conda install python=3.8 mcvine=1.4.5 
```

Install dependencies
```
$ conda install numba=0.53.1 cupy cudatoolkit=11.2.2
```

Build mcvine.acc
```
$ python setup.py sdist
$ pip install --no-deps .
```
