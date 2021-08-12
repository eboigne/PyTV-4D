import numpy as np
# import line_profiler
import tifffile

# [Latest] TV 3: Centered TV formulation (invariant to basic symmetry, and isotropic subgradient around edges)
def tv_centered(img, mask = []):
    
#     weights = np.ones_like(img)
#     weights_mask = tifffile.imread('/home/ihme/eboigne/4DCT/data1903phantomRun93-98/maskBorder.tiff') > 1.0
#     weights[weights_mask] = 1e-2

    if mask != []:
        img[~mask] = 0

    # The intensity differences across rows.
    row_diff_downwind = np.zeros_like(img)
    row_diff_downwind[1:,:] = img[:-1, :] - img[1:, :] # The row_diff at the first row is 0
#     row_diff_downwind = row_diff_downwind * weights
    
    # The intensity differences across columns.
    col_diff_downwind = np.zeros_like(img)
    col_diff_downwind[:,1:] = img[:, :-1] - img[:, 1:] # The col_diff at the first col is 0
#     col_diff_downwind = col_diff_downwind * weights

    # Square the row/col differences only once! (More memory storage, but less redundant calculations)
    diff_norm = np.zeros_like(img)
    
    squared_diff = np.square(row_diff_downwind) # row
    diff_norm += squared_diff # add upwind row term
    diff_norm[:-1,:] += squared_diff[1:,:] # add downwind row term
    
    squared_diff = np.square(col_diff_downwind) # col
    diff_norm += squared_diff # add upwind col term
    diff_norm[:,:-1] += squared_diff[:,1:] # add downwind col term
    
    diff_norm = np.sqrt(diff_norm)
    
    #  Compute the total variation.
    tv = np.sum(np.sum(diff_norm))

    # Find a subgradient.
    G = np.zeros_like(img, dtype = 'float')
    # When non-differentiable, set to 0.
    diff_norm[diff_norm == 0] = np.inf

    ## In-place operations (faster)
    
    # Original (readable, kept for clarity, not optimal)
#     G[1:,:] -= row_diff_downwind[1:,:]/diff_norm[1:,:]
#     G[:,1:] -= col_diff_downwind[:,1:]/diff_norm[:,1:]
    
#     G[:-1, :] += row_diff_downwind[1:,:]/diff_norm[:-1,:] 
#     G[:, :-1] += col_diff_downwind[:,1:]/diff_norm[:,:-1]

#     G[:-1, :] += row_diff_downwind[1:,:]/diff_norm[1:,:]
#     G[:, :-1] += col_diff_downwind[:,1:]/diff_norm[:,1:]
    
#     G[1:, :] -= row_diff_downwind[1:,:]/diff_norm[:-1,:]
#     G[:, 1:] -= col_diff_downwind[:,1:]/diff_norm[:,:-1]
    
    # Optimized version (not so clear, refer to above version commented version first)
    diff = row_diff_downwind[1:,:]/diff_norm[1:,:]
    G[1:,:] -= diff
    G[:-1,:] += diff
    diff = row_diff_downwind[1:,:]/diff_norm[:-1,:]
    G[:-1,:] += diff
    G[1:, :] -= diff
    diff = col_diff_downwind[:,1:]/diff_norm[:,1:]
    G[:,1:] -= diff
    G[:,:-1] += diff
    diff = col_diff_downwind[:,1:]/diff_norm[:,:-1]
    G[:,:-1] += diff
    G[:,1:] -= diff

    return (tv, G)

# @profile
def tvUpwind(img, mask = []):
    # Return the total variation of the 2D image img. If mask is specified, only accounts for the value inside the mask

    weights = np.ones_like(img)
#     weights_mask = tifffile.imread('/home/ihme/eboigne/4DCT/data1903phantomRun93-98/maskBorder.tiff') > 1.0
#     weights[weights_mask] = 1e-2

    if mask != []:
        img[~mask] = 0

    # The intensity differences across rows.
    row_diff = img[:-1, 1:] - img[:-1, :-1]
    # The intensity differences across columns.
    col_diff = img[1:, :-1] - img[:-1, :-1]

    row_diff *= weights[:-1, :-1]
    col_diff *= weights[:-1, :-1]

    #  Compute the total variation.
    eps = 0
    grad_norms = np.sqrt(np.square(row_diff)+np.square(col_diff)+eps)
    tv = np.sum(np.sum(grad_norms))

    # Find a subgradient.
    G = np.zeros_like(img)
    # When non-differentiable, set to 0.
    grad_norms[grad_norms == 0] = np.inf # not necessary if eps > 0

    # v3: corners not implemented, but not so fast anyway ..
    # G[1:-1, 1:-1] =  - (row_diff[1:,1:]+col_diff[1:,1:])/grad_norms[1:,1:] + row_diff[1:,:-1]/grad_norms[1:,:-1] + col_diff[:-1,1:]/grad_norms[:-1,1:]

    # v2: slightly faster than OG
    G[:-1, :-1] =  - (row_diff+col_diff)/grad_norms
    G[:-1, 1:] = G[:-1, 1:] + row_diff/grad_norms
    G[1:, :-1] = G[1:, :-1] + col_diff/grad_norms

    return (tv, G)

def tvDownwind(img, mask = []):
    # Return the total variation of the 2D image img. If mask is specified, only accounts for the value inside the mask

#     weights_mask = tifffile.imread('/home/ihme/eboigne/4DCT/data1903phantomRun93-98/maskBorder.tiff') > 1.0
    weights = np.ones_like(img)
#     weights[weights_mask] = 1e-2

    if mask != []:
        img[~mask] = 0

    # The intensity differences across rows.
    row_diff = img[1:, 1:] - img[1:, :-1]
    # The intensity differences across columns.
    col_diff = img[1:, 1:] - img[:-1, 1:]

    row_diff *= weights[1:, 1:]
    col_diff *= weights[1:, 1:]

    #  Compute the total variation.
    eps = 0
    grad_norms = np.sqrt(np.square(row_diff)+np.square(col_diff)+eps)
    tv = np.sum(np.sum(grad_norms))

    # Find a subgradient.
    G = np.zeros_like(img)
    # When non-differentiable, set to 0.
    grad_norms[grad_norms == 0] = np.inf # not necessary if eps > 0

    # v3: corners not implemented, but not so fast anyway ..
    # G[1:-1, 1:-1] =  - (row_diff[1:,1:]+col_diff[1:,1:])/grad_norms[1:,1:] + row_diff[1:,:-1]/grad_norms[1:,:-1] + col_diff[:-1,1:]/grad_norms[:-1,1:]

    # v2: slightly faster than OG
    G[:-1, :-1] =  - (row_diff+col_diff)/grad_norms
    G[:-1, 1:] = G[:-1, 1:] + row_diff/grad_norms
    G[1:, :-1] = G[1:, :-1] + col_diff/grad_norms

    return (tv, G)


def tvHybrid(img, mask = []):
    # Return the total variation of the 2D image img. If mask is specified, only accounts for the value inside the mask

    (tv1, G1) = tvDownwind(img, mask = mask)
    (tv2, G2) = tvUpwind(img, mask = mask)

    tv = (tv1+tv2)/2.0
    G = (G1+G2)/2.0

    return(tv, G)

def tvCentered(img, mask = []):
    # Return the total variation of the 2D image img. If mask is specified, only accounts for the value inside the mask

    if mask != []:
        img[~mask] = 0

    # The intensity differences across rows.
    row_diff = 0.5 * ( img[1:-1, 2:] - img[1:-1, :-2] )
    # The intensity differences across columns.
    col_diff = 0.5 * ( img[2:, 1:-1] - img[:-2, 1:-1] )

    #  Compute the total variation.
    eps = 0
    grad_norms = np.sqrt(np.square(row_diff)+np.square(col_diff)+eps)
    tv = np.sum(np.sum(grad_norms))

    # Find a subgradient.
    G = np.zeros_like(img)
    # When non-differentiable, set to 0.
    grad_norms[grad_norms == 0] = np.inf # not necessary if eps > 0

    # OG
    G[1:-1, 2:] += row_diff/grad_norms
    G[1:-1, :-2] += - row_diff/grad_norms
    G[2:, 1:-1] += col_diff/grad_norms
    G[:-2, 1:-1] += - col_diff/grad_norms

    # # v2: slightly faster than OG
    # G[:-1, :-1] =  - (row_diff+col_diff)/grad_norms
    # G[:-1, 1:] = G[:-1, 1:] + row_diff/grad_norms
    # G[1:, :-1] = G[1:, :-1] + col_diff/grad_norms

    return (tv, G)


def tvCenteredWeighted(img, mask = []):
    # Return the total variation of the 2D image img. If mask is specified, only accounts for the value inside the mask

#     weights_mask = tifffile.imread('/home/ihme/eboigne/4DCT/data1903phantomRun93-98/maskBorder.tiff') > 1.0
    weights = np.ones_like(img)
#     weights[weights_mask] = 1e-2

    if mask != []:
        img[~mask] = 0

    # The intensity differences across rows.
    row_diff = 0.5 * ( img[1:-1, 2:] - img[1:-1, :-2] )
    # The intensity differences across columns.
    col_diff = 0.5 * ( img[2:, 1:-1] - img[:-2, 1:-1] )

    row_diff *= weights[1:-1, 1:-1]
    col_diff *= weights[1:-1, 1:-1]

    #  Compute the total variation.
    eps = 0
    grad_norms = np.sqrt(np.square(row_diff)+np.square(col_diff)+eps)
    tv = np.sum(np.sum(grad_norms))

    # Find a subgradient.
    G = np.zeros_like(img)
    # When non-differentiable, set to 0.
    grad_norms[grad_norms == 0] = np.inf # not necessary if eps > 0

    # OG
    G[1:-1, 2:] += row_diff/grad_norms
    G[1:-1, :-2] += - row_diff/grad_norms
    G[2:, 1:-1] += col_diff/grad_norms
    G[:-2, 1:-1] += - col_diff/grad_norms

    # # v2: slightly faster than OG
    # G[:-1, :-1] =  - (row_diff+col_diff)/grad_norms
    # G[:-1, 1:] = G[:-1, 1:] + row_diff/grad_norms
    # G[1:, :-1] = G[1:, :-1] + col_diff/grad_norms

    return (tv, G)




# @profile
def tvSlowOg(img, mask = []):
    # Return the total variation of the 2D image img. If mask is specified, only accounts for the value inside the mask

    if mask != []:
        img[~mask] = 0

    # The intensity differences across rows.
    row_diff = img[:-1, 1:] - img[:-1, :-1]
    # The intensity differences across columns.
    col_diff = img[1:, :-1] - img[:-1, :-1]

    #  Compute the total variation.
    eps = 0
    grad_norms = np.sqrt(np.square(row_diff)+np.square(col_diff)+eps)
    tv = np.sum(np.sum(grad_norms))

    # Find a subgradient.
    G = np.zeros_like(img)
    # When non-differentiable, set to 0.
    grad_norms[grad_norms == 0] = np.inf # not necessary if eps > 0

    G[:-1, :-1] = G[:-1, :-1] - row_diff/grad_norms
    G[:-1, 1:] = G[:-1, 1:] + row_diff/grad_norms
    G[:-1, :-1] = G[:-1, :-1] - col_diff/grad_norms
    G[1:, :-1] = G[1:, :-1] + col_diff/grad_norms

    return (tv, G)
