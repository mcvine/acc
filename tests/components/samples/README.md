# Tests for sample components

* `test_isotropic_box.py`: test the first sample implementation "isotropic_box". This implementation is not generic and only works for a box shape sample with isotropic kernel
* `test_homogeneous_single_scatterer`: test the first generic sample implementation "homogeneous_single_scatterer". Check the machinery for cuda code generation.
* `test_isotropic_sphere.py`: compare the accelerated version of isotropic sphere sample using the homogeneous_single_scatterer class with the non-accelerated version of isotropic sphere.
