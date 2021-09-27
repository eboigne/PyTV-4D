# PyTV
A set of Python routines to compute the total variation of images using PyTorch convolution kernels for different gradient discretization.

# Current features

- Explicit functions to compute the total variation of 2D & 3D images. 
- Functions provide subgradients for easy implementation of gradient descent.
- Different spatial discretization schemes available: upwind, downwind, centered, and hybrid.
- Efficient GPU implementations using PyTorch convolution kernels.
- Operator-form implementation compatible with primal-dual formulations.

# Example

`import PyTV as tv`

# Choice of gradient discretization
