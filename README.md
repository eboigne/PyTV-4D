# PyTV
A set of Python routines to compute the Total Variation (TV) of 2D, 3D and 4D images on CPU & GPU, in application to iterative Computed Tomography (CT) reconstructions.

- [Current features](#current-features)
- [Installation](#installation)
    + [CPU Only](#cpu-only)
    + [CPU & GPU](#cpu---gpu)
    + [Testing](#testing)
- [Getting started](#getting-started)
    + [Computing TV and subgradient](#computing-tv-and-subgradient)
    + [Denoizing an image](#denoizing-an-image)
    + [Accelerated convergence using gradient operators](#accelerated-convergence-using-gradient-operators)
- [PyTV functions overview](#pytv-functions-overview)
- [TV definition](#tv-definition)
- [Comments](#comments)

# Current features

- Explicit functions to compute the total variation of 2D & 3D images.
- Functions provide subgradients for easy implementation of (sub)-gradient descent.
- Efficient GPU implementations using PyTorch tensors and convolution kernels.
- Operator-form implementation compatible with primal-dual and proximal formulations.
- Four different spatial discretization schemes are available: upwind, downwind, centered, and hybrid.


# Installation

### CPU Only

For a quick installation running the CPU routines only, install numpy and PyTV using anaconda, skipping the PyTorch dependency for PyTV:

`conda install numpy && conda install --no-deps -c eboigne pytv`


### CPU & GPU

##### Conda
First, install PyTorch (version at least 1.5.0) following the guidelines [on the official website](https://pytorch.org/). Make sure to install the correct version for your setup to enable GPU computations.  

Then, the PyTV files can be installed as a package using anaconda:  

`conda install -c eboigne pytv`

##### Pip
Alternatively, PyTV can be installed using pip. To do so, install numpy and PyTorch and download the [latest tar release ](https://github.com/eboigne/PyTV/releases) of PyTV. Then, using the downloaded file, run:

`pip install ./PyTV-X.X.X.tar.gz`

If you have trouble with installed dependencies not being recognized with pip, run `pip install --no-deps ./PyTV-X.X.X.tar.gz`. 

##### Manual installation
PyTV can also be installed manually with (dependencies need to be set properly):

`python setup.py install`

### Testing

Once installed, you can run some basic tests on CPU and GPU:

```python
import pytv

pytv.run_CPU_tests()
pytv.run_GPU_tests()
```

Note that the tests may fail because of bad rng, so try running it a couple times.


# Getting started

### Computing TV and subgradient

Below is a simple example to compute the total variation and sub-gradient on CPU and GPU:

```python
import pytv  
import numpy as np

Nz, N = 20, 1000 # 3D Image dimensions
np.random.seed(0)
img = np.random.rand(Nz, N, N)

tv1, G1 = pytv.tv_CPU.tv_hybrid(img)
tv2, G2 = pytv.tv_GPU.tv_hybrid(img)

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

### Denoizing an image

A simple example of image denoizing using the total variation. The following loss function is minimized:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?{\color{Gray}\Large&space;\frac{1}{2}||x-x_0||_2^2+\lambda\text{TV}(x)}"/>
</p>

where <img src="https://latex.codecogs.com/svg.latex?{\color{Gray}\Large&space;x"/> is the current image, <img src="https://latex.codecogs.com/svg.latex?{\color{Gray}\Large&space;x_0"/> is the input noisy image, and <img src="https://latex.codecogs.com/svg.latex?{\color{Gray}\Large&space;\lambda"/> is a regularization parameter.
Because the TV is not everywhere differentiable, the sub-gradient descent method is used to minimize this loss function:

```python
import matplotlib.pyplot as plt

noise_level = 100
nb_it = 150
regularization = 25
step_size = 5e-3 # If step size is too large, loss function may not decrease at every step

cameraman_truth = pytv.utils.cameraman() # Open the cameraman's grayscale image
cameraman_noisy = cameraman_truth + noise_level * np.random.rand(*cameraman_truth.shape) # Add noise
cameraman_estimate = np.copy(cameraman_noisy)

loss_fct = np.zeros([nb_it,])
for it in range(nb_it): # A simple sub-gradient descent algorithm for image denoising
    tv, G = pytv.tv.tv_hybrid(cameraman_estimate)
    cameraman_estimate += - step_size * ((cameraman_estimate - cameraman_noisy) + regularization * G)
    loss_fct[it] = 0.5 * np.sum(np.square(cameraman_estimate - cameraman_noisy)) + regularization * tv
```

<p align="center">
<img src="https://raw.githubusercontent.com/eboigne/PyTV/main/pytv/media/img_denoising_cameraman.png" alt="Images of the cameraman"/>
<img src="https://raw.githubusercontent.com/eboigne/PyTV/main/pytv/media/img_denoising_loss_fct.png" alt="Loss function"/>
</p>



### Accelerated convergence using gradient operators
Because the loss function with total variation is non-smooth, it is challenging the achieve sufficient convergence with the gradient descent algorithm. 
Instead, the primal-dual algorithm from Chambolle and Pock (https://doi.org/10.1007/s10851-010-0251-1) achieves faster convergence. 
To enable easy implementation of such proximal-based algorithm, the calculations of image gradients are available in PyTV. 
A simple example is presented below in the case of the denoising of the cameraman image:   

```python
# A simple version of the Chambolle & Pock algorithm for image denoising
# Ref: Chambolle, Antonin, and Thomas Pock. "A first-order primal-dual algorithm for convex problems with applications to imaging." Journal of mathematical imaging and vision 40.1 (2011): 120-145.

sigma_D = 0.5
sigma_A = 1.0
tau = 1 / (8 + 1)

for it in range(nb_it):
    
    # Dual update
    dual_update_fidelity = (dual_update_fidelity + sigma_A * (cameraman_estimate - cameraman_noisy))/(1.0+sigma_A)
    D_x = pytv.tv_operators_GPU.D_hybrid(cameraman_estimate)
    prox_argument = dual_update_TV + sigma_D * D_x
    dual_update_TV = prox_argument / np.maximum(1.0, np.sqrt(np.sum(prox_argument**2, axis = 1)) / regularization)

    # Primal update
    cameraman_estimate = cameraman_estimate - tau * dual_update_fidelity - tau * pytv.tv_operators_GPU.D_T_hybrid(dual_update_TV)
    
    # Loss function update
    loss_fct[it] = 0.5 * np.sum(np.square(cameraman_estimate - cameraman_noisy)) + regularization * pytv.tv_operators_GPU.compute_L21_norm(D_x)
```

<p align="center">
<img src="https://raw.githubusercontent.com/eboigne/PyTV/main/pytv/media/img_denoising_loss_fct_CP_GD.png" alt="Loss function"/>
</p>

# PyTV functions overview

PyTV provides the following functions:

- Direct CPU and GPU, for quick (sub)-gradient descent algorithms (time not currently supported):
```python
use_GPU = True

import numpy as np
if use_GPU:
  import pytv.tv_GPU as tv
else:
  import pytv.tv_CPU as tv

Nz, N = 10, 100 # Image size 
img = np.random.rand(Nz, N, N)

# TV values, and sub-gradient arrays
tv1, G1 = tv.tv_upwind(img)
tv2, G2 = tv.tv_downwind(img)
tv3, G3 = tv.tv_centered(img)
tv4, G4 = tv.tv_hybrid(img)
```
- CPU and GPU operators, useful for proximal algorithms (supports time term):
```python
use_GPU = True

import numpy as np
if use_GPU:
  import pytv.tv_operators_GPU as tv
else:
  import pytv.tv_operators_CPU as tv

Nz, N = 10, 100 # Image size 
M = 2 # Time size
reg_time = 2**(-5) # Time regularization (lambda_t)
img = np.random.rand(Nz, M, N, N)

# Discrete gradient: D_img has size (Nz, Nd, M, N, N) where Nd is the number of difference terms
D_img1 = tv.D_upwind(img, reg_time = reg_time)
D_img2 = tv.D_downwind(img, reg_time)
D_img3 = tv.D_centered(img, reg_time)
D_img4 = tv.D_hybrid(img, reg_time)

# Transposed discrete gradient: D_T_D_img has size (Nz, M, N, N)
D_T_D_img1 = tv.D_T_upwind(D_img1, reg_time)
D_T_D_img2 = tv.D_T_downwind(D_img2, reg_time)
D_T_D_img3 = tv.D_T_centered(D_img3, reg_time)
D_T_D_img4 = tv.D_T_hybrid(D_img4, reg_time)

# TV values: obtained by computing the L2,1 norm of the image gradient D(img) 
tv1 = tv.compute_L21_norm(D_img1)
tv2 = tv.compute_L21_norm(D_img2)
tv3 = tv.compute_L21_norm(D_img3)
tv4 = tv.compute_L21_norm(D_img4)
```

# TV definition


<p align="center">
<img src="https://raw.githubusercontent.com/eboigne/PyTV/main/pytv/media/TV_def.png" alt="TV definition"/>
<img src="https://raw.githubusercontent.com/eboigne/PyTV/main/pytv/media/TV_table_schemes.png" alt="TV discretization"/>
</p>


# Comments

- Nz = 2 is a troublesome case, either send data as 2D images, or a 3D chunk of more than 2 images. The different TV implementations will not give the same results in the case Nz = 2
- Time discretization in the operator forms: the discretization scheme used is the same as the spatial scheme for each discretization. For the `centered` scheme that require M>2, the `upwind` scheme is used instead for the time discretization for cases with M=2.
- The (Nz, M, N, N) data order is prefered to (M, Nz, N, N) since the CT operations can be decomposed easily along z for parallel beam configurations. 

# To implement

- 3D+t for tv_GPU and tv_CPU
