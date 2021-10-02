# PyTV
A set of Python routines to compute the Total Variation (TV) of 2D, 3D and 4D images on CPU & GPU, in application to iterative Computed Tomography (CT) reconstructions.

# Current features

- Explicit functions to compute the total variation of 2D & 3D images.
- Functions provide subgradients for easy implementation of (sub)-gradient descent.
- Efficient GPU implementations using PyTorch tensors and convolution kernels.
- Operator-form implementation compatible with primal-dual and proximal formulations.
- Four different spatial discretization schemes are available: upwind, downwind, centered, and hybrid.

# TV definition


<p align="center">
<img src="https://raw.githubusercontent.com/eboigne/PyTV/main/pytv/media/TV_def.png" alt="TV definition"/>
<img src="https://raw.githubusercontent.com/eboigne/PyTV/main/pytv/media/TV_table_schemes.png" alt="TV discretization"/>
</p>



# Installation

### CPU Only

For a quick installation running the CPU routines only, install numpy and PyTV using anaconda, skipping the PyTorch dependency for PyTV:

`conda install numpy && conda install --no-deps -c eboigne pytv`


### CPU & GPU
First, install PyTorch following the guidelines on the official website: https://pytorch.org/. Make sure to install the correct version for your setup to enable GPU computations.  

Then, the PyTV files can installed as a package using anaconda:  

`conda install -c eboigne pytv`

### Testing

Once installed, you can run some basic tests on CPU and GPU:

```
import pytv

pytv.run_CPU_tests()
pytv.run_GPU_tests()
```

Note that the tests may fail because of bad rng, so try running it a couple times.

# Getting started

### Computing TV and subgradient

Below is a simple example to compute the total variation and sub-gradient on CPU and GPU:

```
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

```
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

plt.figure(1, figsize=[9.5, 3], dpi = 150)
plt.subplot(1,3,1, title='Truth (no noise)')
plt.imshow(cameraman_truth, cmap = plt.get_cmap('gray'))
plt.axis('off')
plt.subplot(1,3,2, title='Noisy input')
plt.axis('off')
plt.imshow(cameraman_noisy, cmap = plt.get_cmap('gray'))
plt.subplot(1,3,3, title='Algorithm output')
plt.imshow(cameraman_estimate, cmap = plt.get_cmap('gray'))
plt.axis('off')
plt.tight_layout(pad=0.5)

plt.figure(2, figsize=[3, 2], dpi = 75)
plt.plot(loss_fct)
plt.xlabel('Iteration')
plt.ylabel('Loss function')
plt.tight_layout(pad=0.5)
plt.show()
```

<p align="center">
<img src="https://raw.githubusercontent.com/eboigne/PyTV/main/pytv/media/img_denoising_cameraman.png" alt="Images of the cameraman"/>
<img src="https://raw.githubusercontent.com/eboigne/PyTV/main/pytv/media/img_denoising_loss_fct.png" alt="Loss function"/>
</p>




# Comments

- Nz = 2 is a troublesome case, either send data as 2D images, or a 3D chunk of more than 2 images.
- Time discretization in the operator forms: the discretization scheme used is the same as the spatial scheme for each discretization. For the `centered` scheme that require M>2, the `upwind` scheme is used instead for the time discretization for cases with M=2.
- The (Nz, M, N, N) data order is prefered to (M, Nz, N, N) since the CT operations can be decomposed easily along z for parallel beam configurations. 
