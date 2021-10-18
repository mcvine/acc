# acc
Accelerated mcvine engine

## Develop

Install mcvine
```
$ conda config --add channels conda-forge 
$ conda config --add channels diffpy 
$ conda config --add channels mantid 
$ conda config --add channels mcvine 
$ conda install python=3.8 mcvine=1.4.4 
```

Install dependencies
```
$ conda install numba=0.53.1 cupy cudatoolkit=11.2.2
```

Build mcvine.acc
```
$ pip install --no-deps .
```
