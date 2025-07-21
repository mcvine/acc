[![CI](https://github.com/mcvine/acc/actions/workflows/CI.yml/badge.svg)](https://github.com/mcvine/acc/actions/workflows/CI.yml)

# acc - Accelerated mcvine engine

Speedup varies for different MCViNE applications.

1000X speedup achieved for a simple virtual experiment with an isotropic spherical sample component.

<img src="https://user-images.githubusercontent.com/1796155/222188657-1c6a4a5a-6970-4516-b51a-ba0329f56dae.png"  width="500">

<img src="https://user-images.githubusercontent.com/1796155/222188086-3156e883-8691-4178-905a-be3f1c48dd1a.png"  width="500">

More details:

<img width="2638" height="1370" alt="image" src="https://github.com/user-attachments/assets/b4bf1948-da47-4c50-8b67-d33716037e9b" />


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
