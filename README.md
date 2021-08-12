# PyTV
A set of Python routines to compute the total variation of images using PyTorch convolution kernels

# Current features

- Explicit functions to compute the total variation of 2D & 3D images. 
- Functions provide subgradients for easy implementation of gradient descent.
- Different spatial discretization schemes available: upwind, downwind, centered, and hybrid.
- Efficient GPU implementations using PyTorch convolution kernels.
- Operator-form implementation compatible with primal-dual formulations.

# Example

`import PyTV as tv`

# Visualizing output data

I recommend using Fiji to visualize the output data from PyRAMID. It enables a quick drag and drop from the .tif stack stored in folders.