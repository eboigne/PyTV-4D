import numpy as np
import tv_2d

def tv_centered(img, mask = []):
    # Return the total variation of the 2D image img. If mask is specified, only accounts for the value inside the mask

    if mask != []:
        img[~mask] = 0

    if len(img) == 2 or (len(img.shape) == 3 and img.shape[0] < 3):
        return(tv_2d.tv_centered(img[0]))

    # The intensity differences across rows.
    row_diff = 0.5 * ( img[1:-1, 2:, 1:-1] - img[1:-1, :-2, 1:-1] )

    # The intensity differences across columns.
    col_diff = 0.5 * ( img[1:-1, 1:-1, 2:] - img[1:-1, 1:-1, :-2] )

    # The intensity differences across slices.
    slice_diff = 0.5 * ( img[2:, 1:-1, 1:-1] - img[:-2, 1:-1, 1:-1] )

    #  Compute the total variation.
    eps = 0
    grad_norms = np.sqrt(np.square(row_diff)+np.square(col_diff)+np.square(slice_diff)+eps)
    tv = np.sum(grad_norms)

    # Find a subgradient.
    G = np.zeros_like(img)
    # When non-differentiable, set to 0.
    grad_norms[grad_norms == 0] = np.inf # not necessary if eps > 0

    G[1:-1, 2:, 1:-1] += row_diff/grad_norms
    G[1:-1, :-2, 1:-1] += - row_diff/grad_norms
    G[1:-1, 1:-1, 2:] += col_diff/grad_norms
    G[1:-1, 1:-1, :-2] += - col_diff/grad_norms
    G[2:, 1:-1, 1:-1] += slice_diff/grad_norms
    G[:-2, 1:-1, 1:-1] += - slice_diff/grad_norms

    return (tv, G)

def tv_hybrid(img, mask = []):

    if mask != []:
        img[~mask] = 0

    if len(img) == 2 or (len(img.shape) == 3 and img.shape[0] == 1):
        return(tv_2d.tv_hybrid(img[0]))

    # The intensity differences across rows.
    row_diff = np.zeros_like(img)
    row_diff[:-1, :-1, :-1] = img[:-1, 1:, :-1] - img[:-1, :-1, :-1]

    # The intensity differences across columns.
    col_diff = np.zeros_like(img)
    col_diff[:-1, :-1, :-1] = img[:-1, :-1, 1:] - img[:-1, :-1, :-1]

    # The intensity differences across slices.
    slice_diff = np.zeros_like(img)
    slice_diff[:-1, :-1, :-1] = img[1:, :-1, :-1] - img[:-1, :-1, :-1]

    #  Compute the total variation.
    eps = 0
    grad_norms = np.zeros_like(img)
    grad_norms[:-1, :-1, :-1] = np.sqrt(np.square(row_diff[:-1, :-1, :-1]) + np.square(col_diff[:-1, :-1, :-1])
                                        + np.square(slice_diff[:-1, :-1, :-1]) + np.square(row_diff[1:, :-1, 1:])
                                        + np.square(col_diff[1:, 1:, :-1]) + np.square(slice_diff[:-1, 1:, 1:]) + eps) / np.sqrt(2)
    tv = np.sum(grad_norms)

    # Find a subgradient.
    G = np.zeros_like(img)
    # When non-differentiable, set to 0.
    grad_norms[grad_norms == 0] = np.inf # not necessary if eps > 0

    G[:-1, :-1, :-1] = - (row_diff+col_diff+slice_diff)[:-1, :-1, :-1]/grad_norms[:-1, :-1, :-1]

    G[:-1, :-1, 1:] += col_diff[:-1, :-1, :-1]/grad_norms[:-1, :-1, :-1]
    G[:-1, 1:, :-1] += row_diff[:-1, :-1, :-1]/grad_norms[:-1, :-1, :-1]
    G[1:, :-1, :-1] += slice_diff[:-1, :-1, :-1]/grad_norms[:-1, :-1, :-1]

    G[1:, 1:, :-1] += -col_diff[1:, 1:, :-1]/grad_norms[:-1, :-1, :-1]
    G[1:, :-1, 1:] += -row_diff[1:, :-1, 1:]/grad_norms[:-1, :-1, :-1]
    G[:-1, 1:, 1:] += -slice_diff[:-1, 1:, 1:]/grad_norms[:-1, :-1, :-1]

    G[1:, 1:, 1:] += (row_diff[1:, :-1, 1:] + col_diff[1:, 1:, :-1] + slice_diff[:-1, 1:, 1:])/grad_norms[:-1, :-1, :-1]

    return (tv, G)

def tv_downwind(img, mask = []):

    if mask != []:
        img[~mask] = 0

    if len(img) == 2 or (len(img.shape) == 3 and img.shape[0] == 1):
        return(tv_2d.tv_downwind(img[0]))
    
    # The intensity differences across rows.
    row_diff = np.zeros_like(img)
    row_diff[:-1, :-1, :-1] = img[:-1, 1:, :-1] - img[:-1, :-1, :-1]

    # The intensity differences across columns.
    col_diff = np.zeros_like(img)
    col_diff[:-1, :-1, :-1] = img[:-1, :-1, 1:] - img[:-1, :-1, :-1]

    # The intensity differences across slices.
    slice_diff = np.zeros_like(img)
    slice_diff[:-1, :-1, :-1] = img[1:, :-1, :-1] - img[:-1, :-1, :-1]

    #  Compute the total variation.
    eps = 0
    grad_norms = np.zeros_like(img)
    grad_norms[:-1, :-1, :-1] = np.sqrt(np.square(row_diff[1:, :-1, 1:])+np.square(col_diff[1:, 1:, :-1])+np.square(slice_diff[:-1, 1:, 1:])+eps)
    tv = np.sum(grad_norms)

    # Find a subgradient.
    G = np.zeros_like(img)
    # When non-differentiable, set to 0.
    grad_norms[grad_norms == 0] = np.inf # not necessary if eps > 0
    
    G[1:, 1:, 1:] = (row_diff[1:, :-1, 1:] + col_diff[1:, 1:, :-1] + slice_diff[:-1, 1:, 1:]) / grad_norms[:-1, :-1, :-1]
    G[1:, :-1, 1:] += - row_diff[1:, :-1, 1:] / grad_norms[:-1, :-1, :-1]
    G[1:, 1:, :-1] += - col_diff[1:, 1:, :-1] / grad_norms[:-1, :-1, :-1]
    G[:-1, 1:, 1:] += - slice_diff[:-1, 1:, 1:] / grad_norms[:-1, :-1, :-1]

    return (tv, G)

def tv_upwind(img, mask = []):

    if mask != []:
        img[~mask] = 0

    if len(img) == 2 or (len(img.shape) == 3 and img.shape[0] == 1):
        return(tv_2d.tv_upwind(img[0]))
    
    # The intensity differences across rows.
    row_diff = np.zeros_like(img)
    row_diff[:-1, :-1, :-1] = img[:-1, 1:, :-1] - img[:-1, :-1, :-1]

    # The intensity differences across columns.
    col_diff = np.zeros_like(img)
    col_diff[:-1, :-1, :-1] = img[:-1, :-1, 1:] - img[:-1, :-1, :-1]

    # The intensity differences across slices.
    slice_diff = np.zeros_like(img)
    slice_diff[:-1, :-1, :-1] = img[1:, :-1, :-1] - img[:-1, :-1, :-1]

    #  Compute the total variation.
    eps = 0
    grad_norms = np.sqrt(np.square(row_diff)+np.square(col_diff)+np.square(slice_diff)+eps)
    tv = np.sum(grad_norms)

    # Find a subgradient.
    G = np.zeros_like(img)
    # When non-differentiable, set to 0.
    grad_norms[grad_norms == 0] = np.inf # not necessary if eps > 0

    G[:-1, :-1, :-1] =  - (row_diff+col_diff+slice_diff)[:-1, :-1, :-1]/grad_norms[:-1, :-1, :-1]
    G[:-1, 1:, :-1] += row_diff[:-1, :-1, :-1]/grad_norms[:-1, :-1, :-1]
    G[:-1, :-1, 1:] += col_diff[:-1, :-1, :-1]/grad_norms[:-1, :-1, :-1]
    G[1:, :-1, :-1] += slice_diff[:-1, :-1, :-1]/grad_norms[:-1, :-1, :-1]

    return (tv, G)

# def tv_mixture(img, mask = []):
#
#     (tv1, G1) = tv_downwind(img, mask = mask)
#     (tv2, G2) = tv_upwind(img, mask = mask)
#     tv = (tv1+tv2)/2.0
#     G = (G1+G2)/2.0
#
#     return(tv, G)
