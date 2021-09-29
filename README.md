# PyTV
A set of Python routines to compute the Total Variation (TV) of 2D and 3D images on CPU & GPU, in application to iterative Computed Tomography (CT) reconstructions.

# Current features

- Explicit functions to compute the total variation of 2D & 3D images.
- Functions provide subgradients for easy implementation of gradient descent.
- Different spatial discretization schemes available: upwind, downwind, centered, and hybrid.
- Efficient GPU implementations using PyTorch tensors and convolution kernels.
- Operator-form implementation compatible with primal-dual and  formulations.

# Example

`import PyTV as tv`

# Choice of gradient discretization


# Comments

- Nz = 2 is a troublesome case, either send data as 2D images, or a 3D chunk of more than 2 images.
- Time discretization in the operator forms: the discretization scheme used is the same as the spatial scheme for each discretization. For the `centered` scheme that require M>2, the `upwind` scheme is used instead for the time discretization for cases with M=2.
