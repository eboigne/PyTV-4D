# PyTV
A set of Python routines to compute the Total Variation (TV) of 2D and 3D images on CPU & GPU, in application to iterative Computed Tomography (CT) reconstructions.

# Current features

- Explicit functions to compute the total variation of 2D & 3D images.
- Functions provide subgradients for easy implementation of gradient descent.
- Different spatial discretization schemes available: upwind, downwind, centered, and hybrid.
- Efficient GPU implementations using PyTorch tensors and convolution kernels.
- Operator-form implementation compatible with primal-dual and  formulations.

# Installation

Install PyTorch following the guidelines on the official website: https://pytorch.org/. Make sure to install the correct version for your setup to enable GPU computations.  

Then, the PyTV files can be downloaded directly from the Github repository, or installed as a package using anaconda:  

`conda install -c eboigne pytv`

# Getting started

Below is a simple example to compute the total variation and sub-gradient on CPU and GPU:

```
import pytv  
import numpy as np

Nz, N = 20, 1000 # 3D Image dimensions
np.random.seed(0)
img = np.random.rand(Nz, N, N)

tv1, G1 = pytv.tv.tv_hybrid(img)
tv2, G2 = pytv.tv_pyTorch.tv_hybrid(img)

print('TV value from CPU: '+str(tv1))
print('TV value from GPU: '+str(tv2))
print('Sub-gradients from CPU and GPU are equal: '+str(np.prod(np.abs(G1-G2)<1e-5)>0))
```

Output is:

```
TV value from CPU: 12763241.060426874
TV value from GPU: 12763241.0
Sub-gradients from CPU and GPU are equal: True
```


# TV Gradient discretization


# Comments

- Nz = 2 is a troublesome case, either send data as 2D images, or a 3D chunk of more than 2 images.
- Time discretization in the operator forms: the discretization scheme used is the same as the spatial scheme for each discretization. For the `centered` scheme that require M>2, the `upwind` scheme is used instead for the time discretization for cases with M=2.
