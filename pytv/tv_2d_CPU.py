import numpy as np

def tv_hybrid(img, mask = []):
    '''
    Calculates the total variation and a subgradient of the input image img using the hybrid gradient discretization

    Parameters
    ----------
    img : np.ndarray
        The array of the input image data of dimensions N x N.
    mask : np.ndarray
        A mask used to specify regions of the image ot skip for the TV calculation.

    Returns
    -------
    tv : float
        The value of the total variation.
    G : np.ndarray
        A sub-gradient array of the total variation term of dimensions N x N.
    '''

    if mask != []:
        img[~mask] = 0

    # The intensity differences across rows.
    row_diff = np.zeros_like(img)
    row_diff[:-1, :-1] = img[1:, :-1] - img[:-1, :-1]

    # The intensity differences across columns.
    col_diff = np.zeros_like(img)
    col_diff[:-1, :-1] = img[:-1, 1:] - img[:-1, :-1]

    #  Compute the total variation.
    eps = 0
    grad_norms = np.zeros_like(img)
    grad_norms[:-1, :-1] = np.sqrt(np.square(row_diff[:-1, :-1])
                                   + np.square(col_diff[:-1, :-1]) + np.square(row_diff[:-1, 1:])
                                   +np.square(col_diff[1:, :-1])+eps) / np.sqrt(2)
    tv = np.sum(grad_norms)

    # Find a subgradient.
    G = np.zeros_like(img)
    # When non-differentiable, set to 0.
    grad_norms[grad_norms == 0] = np.inf # not necessary if eps > 0

    G[:-1, :-1] = - (row_diff+col_diff)[:-1, :-1]/grad_norms[:-1, :-1]
    G[:-1, 1:] += (col_diff[:-1, :-1] - row_diff[:-1, 1:])/grad_norms[:-1, :-1]
    G[1:, :-1] += (row_diff[:-1, :-1] - col_diff[1:, :-1])/grad_norms[:-1, :-1]
    G[1:, 1:] += (row_diff[:-1, 1:]+col_diff[1:, :-1])/grad_norms[:-1, :-1]

    return (tv, G)

def tv_downwind(img, mask = []):
    '''
    Calculates the total variation and a subgradient of the input image img using the downwind gradient discretization

    Parameters
    ----------
    img : np.ndarray
        The array of the input image data of dimensions N x N.
    mask : np.ndarray
        A mask used to specify regions of the image ot skip for the TV calculation.

    Returns
    -------
    tv : float
        The value of the total variation.
    G : np.ndarray
        A sub-gradient array of the total variation term of dimensions N x N.
    '''

    if mask != []:
        img[~mask] = 0
    
    # The intensity differences across rows.
    row_diff = np.zeros_like(img)
    row_diff[:-1, :-1] = img[1:, :-1] - img[:-1, :-1]

    # The intensity differences across columns.
    col_diff = np.zeros_like(img)
    col_diff[:-1, :-1] = img[:-1, 1:] - img[:-1, :-1]

    #  Compute the total variation.
    eps = 0
    grad_norms = np.zeros_like(img)
    grad_norms[:-1, :-1] = np.sqrt(np.square(row_diff[:-1, 1:])+np.square(col_diff[1:, :-1])+eps)
    tv = np.sum(grad_norms)

    # Find a subgradient.
    G = np.zeros_like(img)
    # When non-differentiable, set to 0.
    grad_norms[grad_norms == 0] = np.inf # not necessary if eps > 0
    
    G[1:, 1:] =  (row_diff[:-1,1:]+col_diff[1:,:-1])/grad_norms[:-1, :-1]
    G[:-1, 1:] += - row_diff[:-1, 1:]/grad_norms[:-1, :-1]
    G[1:, :-1] += - col_diff[1:, :-1]/grad_norms[:-1, :-1]

    return (tv, G)

def tv_upwind(img, mask = []):
    '''
    Calculates the total variation and a subgradient of the input image img using the upwind gradient discretization

    Parameters
    ----------
    img : np.ndarray
        The array of the input image data of dimensions N x N.
    mask : np.ndarray
        A mask used to specify regions of the image ot skip for the TV calculation.

    Returns
    -------
    tv : float
        The value of the total variation.
    G : np.ndarray
        A sub-gradient array of the total variation term of dimensions N x N.
    '''

    if mask != []:
        img[~mask] = 0
    
    # The intensity differences across rows.
    row_diff = np.zeros_like(img)
    row_diff[:-1, :-1] = img[1:, :-1] - img[:-1, :-1]

    # The intensity differences across columns.
    col_diff = np.zeros_like(img)
    col_diff[:-1, :-1] = img[:-1, 1:] - img[:-1, :-1]

    #  Compute the total variation.
    eps = 0
    grad_norms = np.sqrt(np.square(row_diff)+np.square(col_diff)+eps)
    tv = np.sum(grad_norms)

    # Find a subgradient.
    G = np.zeros_like(img)
    # When non-differentiable, set to 0.
    grad_norms[grad_norms == 0] = np.inf # not necessary if eps > 0

    G[:-1, :-1] =  - (row_diff+col_diff)[:-1, :-1]/grad_norms[:-1, :-1]
    G[1:, :-1] += row_diff[:-1, :-1]/grad_norms[:-1, :-1]
    G[:-1, 1:] += col_diff[:-1, :-1]/grad_norms[:-1, :-1]

    return (tv, G)

def tv_centered(img, mask = []):
    '''
    Calculates the total variation and a subgradient of the input image img using the centered gradient discretization

    Parameters
    ----------
    img : np.ndarray
        The array of the input image data of dimensions N x N.
    mask : np.ndarray
        A mask used to specify regions of the image ot skip for the TV calculation.

    Returns
    -------
    tv : float
        The value of the total variation.
    G : np.ndarray
        A sub-gradient array of the total variation term of dimensions N x N.
    '''

    if mask != []:
        img[~mask] = 0

    # The intensity differences across rows.
    col_diff = 0.5 * ( img[1:-1, 2:] - img[1:-1, :-2] )
    # The intensity differences across columns.
    row_diff = 0.5 * ( img[2:, 1:-1] - img[:-2, 1:-1] )

    #  Compute the total variation.
    eps = 0
    grad_norms = np.sqrt(np.square(row_diff)+np.square(col_diff)+eps)
    tv = np.sum(grad_norms)

    # Find a subgradient.
    G = np.zeros_like(img)
    # When non-differentiable, set to 0.
    grad_norms[grad_norms == 0] = np.inf # not necessary if eps > 0

    G[1:-1, 2:] += col_diff/grad_norms
    G[1:-1, :-2] += - col_diff/grad_norms
    G[2:, 1:-1] += row_diff/grad_norms
    G[:-2, 1:-1] += - row_diff/grad_norms

    return (tv, G)
