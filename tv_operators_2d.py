import numpy as np

def compute_L21_norm(D_img):
    '''
    Compute the L2,1 norm of an image of discrete differences: |x|_2,1 = \sum_i \sqrt(\sum_j x_{i,j}^2),
    with index i summing over image pixels, and index j summing over difference terms.
    Usage: TV(img) = reg * compute_L21_norm(D(img))

    Parameters
    ----------
    D_img : np.ndarray
        The numpy array of differences of dimensions Nz x Nd x M x N x N.

    Returns
    -------
    float
        The L2,1 norm of the given input array.
    '''

    out = np.square(D_img)
    out = np.sum(out, axis = tuple(range(len(D_img.shape)-3)))
    out = np.sqrt(out)
    out = np.sum(out, axis = (-1,-2))
    return(out)

def D_hybrid(img, reg_z_over_reg = 0, reg_time = 0, halve_tv_at_both_end = False, factor_reg_static = 0, mask_static = False):
    '''
    Calculates the image of the operator D (gradient discretized using hybrid scheme) applied to variable img
    Parameters:
        img : img of dimensions Nz x M x N x N
    Returns: 
        out: D(img) of dimensions Nz x 4/6/8 x M x N x N
    '''
    if reg_z_over_reg == np.nan:
        reg_z_over_reg = 0.0
    
#     # 1 - Reshape input img as Nz x M x N x N
#     if len(img.shape) == 2: # img: N x N
#         img = np.reshape(img, [1, img.shape[0], img.shape[1]])
        
#     bool_time =len(img.shape) > 3
    
#     if len(img.shape) > 2 and M > 1: # img: M x N x N
#         img = np.reshape(img, [1, img.shape[0], img.shape[1], img.shape[2]])
#     elif len(img.shape) > 2 and M == 1: # img: Nz x N x N:
#         img = np.reshape(img, [img.shape[0], 1, img.shape[1], img.shape[2]])
        
    Nz = img.shape[0]
    M = img.shape[1]
    N = img.shape[-1]

    N_d = 4
    if Nz > 1 and reg_z_over_reg > 0:
        N_d += 2
    if reg_time > 0 and M > 1:
        N_d += 2
    D_img = np.zeros([Nz, N_d, M, N, N])

    row_diff = np.zeros_like(img)
    row_diff[:,:,:-1, :-1] = img[:,:, 1:, :-1] - img[:,:, :-1, :-1]

    col_diff = np.zeros_like(img)
    col_diff[:,:,:-1, :-1] = img[:,:, :-1, 1:] - img[:,:, :-1, :-1]

    # The intensity differences across rows (Upwind / Forward)
    # D_img[:,0,:,:-1,:] = img[:,:, 1:, :] - img[:,:, :-1, :]  # The row_diff at the last row is 0
    # D_img[:,0,:,:-1,:-1] = img[:,:, 1:, :-1] - img[:,:, :-1, :-1]
    D_img[:,0,:,:-1,:-1] = row_diff[:,:,:-1, :-1]

    # The intensity differences across columns (Upwind / Forward)
    # D_img[:,1,:,:,:-1] = img[:,:, :, 1:] - img[:,:, :, :-1]
    # D_img[:,1,:,:-1,:-1] = img[:,:, :-1, 1:] - img[:,:, :-1, :-1]
    D_img[:,1,:,:-1,:-1] = col_diff[:,:,:-1, :-1]

    # The intensity differences across rows (Downwind / Backward)
    # D_img[:,2,:,1:,:] = img[:, :, :-1, :] - img[:, :, 1:, :]
    # D_img[:,2,:,:-1,:-2] = - (img[:,:, 1:, 1:-1] - img[:,:, :-1, 1:-1])
    D_img[:,2,:,:-1,:-2] = - row_diff[:,:,:-1, 1:-1]

    # # The intensity differences across columns (Downwind / Backward)
    # # D_img[:,3,:,:,1:] = img[:,:, :, :-1] - img[:,:, :, 1:]
    # # D_img[:,3,:,:-2,:-1] = - (img[:,:, 1:-1, 1:] - img[:,:, 1:-1, :-1])
    D_img[:,3,:,:-2,:-1] = - col_diff[:,:,1:-1, :-1]

    i_d = 4
    if Nz > 1 and reg_z_over_reg > 0:
        # The intensity differences across z (Upwind / Forward)
        D_img[:-1,i_d,:,:,:] = np.sqrt(reg_z_over_reg) * (img[1:,:, :, :] - img[:-1,:, :, :])  # The row_diff at the last z is 0
        i_d += 1
        
        # The intensity differences across z (Downwind / Backward)
        D_img[1:, i_d,:,:,:] = np.sqrt(reg_z_over_reg) * (img[:-1, :, :, :] - img[1:, :, :, :]) # The row_diff at the first z is 0
        i_d += 1

    if reg_time > 0 and M > 1:
        # f^k+1 - f^k
        D_img[:,i_d,:-1,:,:] =  np.sqrt(reg_time) * (img[:,1:,:,:] - img[:,:-1,:,:])

        # f^k-1 - f^k
        D_img[:,i_d+1,1:,:,:] =  np.sqrt(reg_time) * (img[:,:-1,:,:] - img[:,1:,:,:])
        
        if isinstance(mask_static, np.ndarray):
            mask_static = np.tile(mask_static, [Nz, 1, 1, 1])
            
            D_img_temp = D_img[:,i_d,:-1,:,:].copy()
            D_img_temp[mask_static] *= np.sqrt(factor_reg_static)
            D_img[:,i_d,:-1,:,:] = D_img_temp
            
            D_img_temp = D_img[:,i_d+1,1:,:,:].copy()
            D_img_temp[mask_static] *= np.sqrt(factor_reg_static)
            D_img[:,i_d+1,1:,:,:] = D_img_temp
        
        i_d += 2
        
    if halve_tv_at_both_end and M > 2:
        D_img[:, 0, :, :] /= 2.0
        D_img[:, -1, :, :] /= 2.0

    return (D_img/np.sqrt(2.0))

def D_T_hybrid(img, reg_z_over_reg = 0, reg_time = 0, halve_tv_at_both_end = False, factor_reg_static = 0, mask_static = False):
    '''
    Calculates the image of the operator D^T (transposed gradient discretized using centered scheme) applied to variable img
    Parameters:
        img : img of dimensions squeeze(Nz x 4/6/8 x M x N x N)
    Returns: 
        out: D_T(img) of dimensions Nz x N x N (or Nz x M x N x N)
    '''

    if reg_z_over_reg == np.nan:
        reg_z_over_reg = 0.0
        
#     if img.shape[0] == 4 and len(img.shape) == 3:
#         img = np.reshape(img, [1, img.shape[0], img.shape[1], img.shape[2]])
#     if img.shape[0] == 6 and len(img.shape) == 4:
#         img = np.reshape(img, [1, img.shape[0], img.shape[1], img.shape[2], img.shape[3]])
    
    Nz = img.shape[0]
    N_d = img.shape[1]
    N = img.shape[-1]
        
    M = img.shape[2]
    D_T_img = np.zeros([Nz, M, N, N])
    
    if halve_tv_at_both_end and M > 2:
        img = img.copy()
        img[:,:,0,:,:] /= 2.0
        img[:,:,-1,:,:] /= 2.0

    # Forward row term
    # D_T_img[:,:,1:-1,:] += img[:,0,:,:-2,:]-img[:,0,:,1:-1,:]
    # D_T_img[:,:,0,:] += -img[:,0,:,0,:]
    # D_T_img[:,:,-1,:] += img[:,0,:,-2,:]
    D_T_img[:,:,1:-1,:-1] += img[:,0,:,:-2,:-1]-img[:,0,:,1:-1,:-1]
    D_T_img[:,:,0,:-1] += -img[:,0,:,0,:-1]
    D_T_img[:,:,-1,:-1] += img[:,0,:,-2,:-1]

    # Forward col term
    # D_T_img[:,:,:,1:-1] += img[:,1,:,:,:-2]-img[:,1,:,:,1:-1]
    # D_T_img[:,:,:,0] += -img[:,1,:,:,0]
    # D_T_img[:,:,:,-1] += img[:,1,:,:,-2]
    D_T_img[:,:,:-1,1:-1] += img[:,1,:,:-1,:-2]-img[:,1,:,:-1,1:-1]
    D_T_img[:,:,:-1,0] += -img[:,1,:,:-1,0]
    D_T_img[:,:,:-1,-1] += img[:,1,:,:-1,-2]

    # Backward row term
    # D_T_img[:,:,1:-1,:] += img[:,2,:,2:,:] - img[:,2,:,1:-1,:]
    # D_T_img[:,:,0,:] += img[:,2,:,1,:]
    # D_T_img[:,:,-1,:] += -img[:,2,:,-1,:]
    D_T_img[:,:,1:-1,1:-1] += img[:,2,:,1:-1,:-2] - img[:,2,:,:-2,:-2]
    D_T_img[:,:,0,1:-1] += img[:,2,:,0,:-2]
    D_T_img[:,:,-1,1:-1] += -img[:,2,:,-2,:-2]

    # Backward col term
    # D_T_img[:,:,:,1:-1] += img[:,3,:,:,2:] - img[:,3,:,:,1:-1]
    # D_T_img[:,:,:,0] += img[:,3,:,:,1]
    # D_T_img[:,:,:,-1] += - img[:,3,:,:,-1]
    D_T_img[:,:,1:-1,1:-1] += img[:,3,:,:-2,1:-1] - img[:,3,:,:-2,:-2]
    D_T_img[:,:,1:-1,0] += img[:,3,:,:-2,0]
    D_T_img[:,:,1:-1,-1] += - img[:,3,:,:-2,-2]

    i_d = 4
    if Nz > 1 and reg_z_over_reg > 0: # z-terms
        # Forward z term
        D_T_img[1:-1,:,:,:] += np.sqrt(reg_z_over_reg) * (img[:-2,i_d,:,:,:]-img[1:-1,i_d,:,:,:])
        D_T_img[0,:,:,:] += -np.sqrt(reg_z_over_reg) * img[0,i_d,:,:,:]
        D_T_img[-1,:,:,:] += np.sqrt(reg_z_over_reg) * img[-2,i_d,:,:,:]
        i_d += 1

        # Backward z term
        D_T_img[1:-1,:,:,:] += np.sqrt(reg_z_over_reg) * (img[2:,i_d,:,:,:] - img[1:-1,i_d,:,:,:])
        D_T_img[0,:,:,:] += np.sqrt(reg_z_over_reg) * img[1,i_d,:,:,:]
        D_T_img[-1,:,:,:] += -np.sqrt(reg_z_over_reg) * img[-1,i_d,:,:,:]
        i_d += 1

    if reg_time > 0 and M > 1:
        D_T_img_time_update = np.zeros_like(D_T_img)
        
        # Forward time term
        D_T_img_time_update[:,1:-1,:,:] += np.sqrt(reg_time) * (img[:,i_d,:-2,:,:]-img[:,i_d,1:-1,:,:])
        D_T_img_time_update[:,0,:,:] += -np.sqrt(reg_time) * img[:,i_d,0,:,:]
        D_T_img_time_update[:,-1,:,:] += np.sqrt(reg_time) * img[:,i_d,-2,:,:]
        i_d += 1

        # Backward time term
        D_T_img_time_update[:,1:-1,:,:] += np.sqrt(reg_time) * (img[:,i_d,2:,:,:] - img[:,i_d,1:-1,:,:])
        D_T_img_time_update[:,0,:,:] += np.sqrt(reg_time) * img[:,i_d,1,:,:]
        D_T_img_time_update[:,-1,:,:] += -np.sqrt(reg_time) * img[:,i_d,-1,:,:]
        i_d += 1
        
        if isinstance(mask_static, np.ndarray):
            mask_static = np.tile(mask_static, [Nz, M, 1, 1])
            D_T_img_time_update[mask_static] *= np.sqrt(factor_reg_static)
        
        D_T_img += D_T_img_time_update

    return(D_T_img/np.sqrt(2.0))

def D_downwind(img, reg_z_over_reg = 0, reg_time = 0):
    '''
    Calculates the image of the operator D (gradient discretized using downwind scheme) applied to variable img.
    Parameters:
        img : img of dimensions Nz x M x N x N
    Returns:
        out: D(img) of dimensions Nz x 2/3/4 x M x N x N
    '''
    if reg_z_over_reg == np.nan:
        reg_z_over_reg = 0.0

    Nz = img.shape[0]
    M = img.shape[1]
    N = img.shape[-1]

    N_d = 2
    if Nz > 1 and reg_z_over_reg > 0:
        N_d += 1
    if reg_time > 0 and M > 1:
        N_d += 1

    D_img = np.zeros([Nz, N_d, M, N, N])

    # The intensity differences across rows (Downwind / Backward)
    # D_img[:,0,:,1:,:] = img[:, :, :-1, :] - img[:, :, 1:, :] # The row_diff at the first row is 0
    D_img[:,0,:,1:,1:-1] = img[:, :, :-1, 1:-1] - img[:, :, 1:, 1:-1]

    # The intensity differences across columns (Downwind / Backward)
    # D_img[:,1,:,:,1:] = img[:,:, :, :-1] - img[:,:, :, 1:] # The col_diff at the first col is 0
    D_img[:,1,:,1:-1,1:] = img[:,:, 1:-1, :-1] - img[:,:, 1:-1, 1:]

    i_d = 2
    if Nz > 1 and reg_z_over_reg > 0:
        # The intensity differences across z (Downwind / Backward)
        D_img[1:, i_d,:,:,:] = np.sqrt(reg_z_over_reg) * (img[:-1, :, :, :] - img[1:, :, :, :]) # The row_diff at the first z is 0
        i_d += 1

    if reg_time > 0 and M > 1:
        # f^k-1 - f^k
        D_img[:,i_d,1:,:,:] =  np.sqrt(reg_time) * (img[:,:-1,:,:] - img[:,1:,:,:])
        i_d += 1

    return D_img

def D_upwind(img, reg_z_over_reg = 0, reg_time = 0, pad = 1):
    '''
    Calculates the image of the operator D (gradient discretized using centered scheme) applied to variable img
    Parameters:
        img : img of dimensions Nz x M x N x N
    Returns:
        out: D(img) of dimensions Nz x 2/3/4 x M x N x N
    '''
    if reg_z_over_reg == np.nan:
        reg_z_over_reg = 0.0

    Nz = img.shape[0]
    M = img.shape[1]
    N = img.shape[-1]

    N_d = 2
    if Nz > 1 and reg_z_over_reg > 0:
        N_d += 1
    if reg_time > 0 and M > 1:
        N_d += 1
    D_img = np.zeros([Nz, N_d, M, N, N])

    # The intensity differences across rows (Upwind / Forward)
    # D_img[:,0,:,:-1,:] = img[:,:, 1:, :] - img[:,:, :-1, :]  # The row_diff at the last row is 0
    D_img[:,0,:,:-1,:-1] = img[:,:, 1:, :-1] - img[:,:, :-1, :-1]

    # The intensity differences across columns (Upwind / Forward)
    # D_img[:,1,:,:,:-1] = img[:,:, :, 1:] - img[:,:, :, :-1]
    D_img[:,1,:,:-1,:-1] = img[:,:, :-1, 1:] - img[:,:, :-1, :-1]

    i_d = 2
    if Nz > 1 and reg_z_over_reg > 0:
        # The intensity differences across z (Upwind / Forward)
        D_img[:-1,i_d,:,:,:] = np.sqrt(reg_z_over_reg) * (img[1:,:, :, :] - img[:-1,:, :, :])  # The row_diff at the last z is 0
        i_d += 1
        
    if reg_time > 0 and M > 1:
        # f^k+1 - f^k
        D_img[:,i_d,:-1,:,:] =  np.sqrt(reg_time) * (img[:,1:,:,:] - img[:,:-1,:,:])
        i_d += 1

    return D_img

def D_centered(img, reg_z_over_reg = 0, reg_time = 0):
    '''
    Calculates the image of the operator D (gradient discretized using centered scheme) applied to variable img
    Parameters:
        img : img of dimensions Nz x M x N x N
    Returns: 
        out: D(img) of dimensions Nz x 2/3/4 x M x N x N
    '''
    if reg_z_over_reg == np.nan:
        reg_z_over_reg = 0.0
    
#     # 1 - Reshape input img as Nz x M x N x N
#     if len(img.shape) == 2: # img: N x N
#         img = np.reshape(img, [1, img.shape[0], img.shape[1]])
        
#     bool_time =len(img.shape) > 3
    
#     if len(img.shape) > 2 and M > 1: # img: M x N x N
#         img = np.reshape(img, [1, img.shape[0], img.shape[1], img.shape[2]])
#     elif len(img.shape) > 2 and M == 1: # img: Nz x N x N:
#         img = np.reshape(img, [img.shape[0], 1, img.shape[1], img.shape[2]])
        
    Nz = img.shape[0]
    M = img.shape[1]
    N = img.shape[-1]

    N_d = 2
    if Nz > 1 and reg_z_over_reg > 0:
        N_d += 1
    if reg_time > 0 and M > 1:
        N_d += 1
    D_img = np.zeros([Nz, N_d, M, N, N])
    
    # D_img[:,0,:,1:-1,:] = 0.5 * (img[:,:, 2:, :] - img[:,:, :-2, :])
    D_img[:,0,:,1:-1,1:-1] = 0.5 * (img[:,:, 2:, 1:-1] - img[:,:, :-2, 1:-1])

    # D_img[:,1,:,:,1:-1] = 0.5 * (img[:,:, :, 2:] - img[:,:, :, :-2])
    D_img[:,1,:,1:-1,1:-1] = 0.5 * (img[:,:, 1:-1, 2:] - img[:,:, 1:-1, :-2])
    
    i_d = 2
    if Nz > 1 and reg_z_over_reg > 0:
        D_img[1:-1,i_d,:,:,:] = np.sqrt(reg_z_over_reg) * 0.5 * (img[2:,:, :, :] - img[:-2,:, :, :])
        i_d += 1

    if reg_time > 0 and M > 1:
        D_img[:,i_d,1:-1,:,:] =  np.sqrt(reg_time) * 0.5 * (img[:,2:,:,:] - img[:,:-2,:,:])
        i_d += 1

    return (D_img)

def D_T_downwind(img, reg_z_over_reg = 0, reg_time = 0):
    '''
    Calculates the image of the operator D^T (transposed gradient discretized using centered scheme) applied to variable img
    Parameters:
        img : img of dimensions squeeze(Nz x 2/3/4 x M x N x N)
    Returns: 
        out: D_T(img) of dimensions Nz x N x N (or Nz x M x N x N)
    '''

    if reg_z_over_reg == np.nan:
        reg_z_over_reg = 0.0
        
#     if img.shape[0] == 4 and len(img.shape) == 3:
#         img = np.reshape(img, [1, img.shape[0], img.shape[1], img.shape[2]])
#     if img.shape[0] == 6 and len(img.shape) == 4:
#         img = np.reshape(img, [1, img.shape[0], img.shape[1], img.shape[2], img.shape[3]])
    
    Nz = img.shape[0]
    N_d = img.shape[1]
    N = img.shape[-1]
        
    M = img.shape[2]
    D_T_img = np.zeros([Nz, M, N, N])

    # Backward row term
    # D_T_img[:,:,1:-1,:] += img[:,0,:,2:,:] - img[:,0,:,1:-1,:]
    # D_T_img[:,:,0,:] += img[:,0,:,1,:]
    # D_T_img[:,:,-1,:] += -img[:,0,:,-1,:]
    D_T_img[:,:,1:-1,1:-1] += img[:,0,:,2:,1:-1] - img[:,0,:,1:-1,1:-1]
    D_T_img[:,:,0,1:-1] += img[:,0,:,1,1:-1]
    D_T_img[:,:,-1,1:-1] += -img[:,0,:,-1,1:-1]

    # Backward col term
    # D_T_img[:,:,:,1:-1] += img[:,1,:,:,2:] - img[:,1,:,:,1:-1]
    # D_T_img[:,:,:,0] += img[:,1,:,:,1]
    # D_T_img[:,:,:,-1] += - img[:,1,:,:,-1]
    D_T_img[:,:,1:-1,1:-1] += img[:,1,:,1:-1,2:] - img[:,1,:,1:-1,1:-1]
    D_T_img[:,:,1:-1,0] += img[:,1,:,1:-1,1]
    D_T_img[:,:,1:-1,-1] += - img[:,1,:,1:-1,-1]

    i_d = 2
    if Nz > 1 and reg_z_over_reg > 0: # z-terms
        # Backward z term
        D_T_img[1:-1,:,:,:] += np.sqrt(reg_z_over_reg) * (img[2:,i_d,:,:,:] - img[1:-1,i_d,:,:,:])
        D_T_img[0,:,:,:] += np.sqrt(reg_z_over_reg) * img[1,i_d,:,:,:]
        D_T_img[-1,:,:,:] += -np.sqrt(reg_z_over_reg) * img[-1,i_d,:,:,:]
        i_d += 1

    if reg_time > 0 and M > 1:
        # Backward time term
        D_T_img[:,1:-1,:,:] += np.sqrt(reg_time) * (img[:,i_d,2:,:,:] - img[:,i_d,1:-1,:,:])
        D_T_img[:,0,:,:] += np.sqrt(reg_time) * img[:,i_d,1,:,:]
        D_T_img[:,-1,:,:] += -np.sqrt(reg_time) * img[:,i_d,-1,:,:]
        i_d += 1

    return(D_T_img)

def D_T_upwind(img, reg_z_over_reg = 0, reg_time = 0):
    '''
    Calculates the image of the operator D^T (transposed gradient discretized using centered scheme) applied to variable img
    Parameters:
        img : img of dimensions squeeze(Nz x 4/6/8 x M x N x N)
    Returns: 
        out: D_T(img) of dimensions Nz x N x N (or Nz x M x N x N)
    '''

    if reg_z_over_reg == np.nan:
        reg_z_over_reg = 0.0
        
#     if img.shape[0] == 4 and len(img.shape) == 3:
#         img = np.reshape(img, [1, img.shape[0], img.shape[1], img.shape[2]])
#     if img.shape[0] == 6 and len(img.shape) == 4:
#         img = np.reshape(img, [1, img.shape[0], img.shape[1], img.shape[2], img.shape[3]])
    
    Nz = img.shape[0]
    N_d = img.shape[1]
    N = img.shape[-1]
        
    M = img.shape[2]
    D_T_img = np.zeros([Nz, M, N, N])

    # Forward row term
    # D_T_img[:,:,1:-1,:] += img[:,0,:,:-2,:]-img[:,0,:,1:-1,:]
    # D_T_img[:,:,0,:] += -img[:,0,:,0,:]
    # D_T_img[:,:,-1,:] += img[:,0,:,-2,:]
    D_T_img[:,:,1:-1,:-1] += img[:,0,:,:-2,:-1]-img[:,0,:,1:-1,:-1]
    D_T_img[:,:,0,:-1] += -img[:,0,:,0,:-1]
    D_T_img[:,:,-1,:-1] += img[:,0,:,-2,:-1]

    # Forward col term
    # D_T_img[:,:,:,1:-1] += img[:,1,:,:,:-2]-img[:,1,:,:,1:-1]
    # D_T_img[:,:,:,0] += -img[:,1,:,:,0]
    # D_T_img[:,:,:,-1] += img[:,1,:,:,-2]
    D_T_img[:,:,:-1,1:-1] += img[:,1,:,:-1,:-2]-img[:,1,:,:-1,1:-1]
    D_T_img[:,:,:-1,0] += -img[:,1,:,:-1,0]
    D_T_img[:,:,:-1,-1] += img[:,1,:,:-1,-2]

    i_d = 2
    if Nz > 1 and reg_z_over_reg > 0: # z-terms
        # Forward z term
        D_T_img[1:-1,:,:,:] += np.sqrt(reg_z_over_reg) * (img[:-2,i_d,:,:,:]-img[1:-1,i_d,:,:,:])
        D_T_img[0,:,:,:] += -np.sqrt(reg_z_over_reg) * img[0,i_d,:,:,:]
        D_T_img[-1,:,:,:] += np.sqrt(reg_z_over_reg) * img[-2,i_d,:,:,:]
        i_d += 1

    if reg_time > 0 and M > 1:
        # Forward time term
        D_T_img[:,1:-1,:,:] += np.sqrt(reg_time) * (img[:,i_d,:-2,:,:]-img[:,i_d,1:-1,:,:])
        D_T_img[:,0,:,:] += -np.sqrt(reg_time) * img[:,i_d,0,:,:]
        D_T_img[:,-1,:,:] += np.sqrt(reg_time) * img[:,i_d,-2,:,:]
        i_d += 1

    return(D_T_img)

def D_T_centered(img, reg_z_over_reg = 0, reg_time = 0):
    '''
    Calculates the image of the operator D^T (transposed gradient discretized using centered scheme) applied to variable img
    Parameters:
        img : img of dimensions squeeze(Nz x 2/3/4 x M x N x N)
    Returns: 
        out: D_T(img) of dimensions Nz x N x N (or Nz x M x N x N)
    '''

    if reg_z_over_reg == np.nan:
        reg_z_over_reg = 0.0
        
#     if img.shape[0] == 4 and len(img.shape) == 3:
#         img = np.reshape(img, [1, img.shape[0], img.shape[1], img.shape[2]])
#     if img.shape[0] == 6 and len(img.shape) == 4:
#         img = np.reshape(img, [1, img.shape[0], img.shape[1], img.shape[2], img.shape[3]])
    
    Nz = img.shape[0]
    N_d = img.shape[1]
    N = img.shape[-1]
        
    M = img.shape[2]
    D_T_img = np.zeros([Nz, M, N, N])
    
    # Row term

    # Old from v1
    # D_T_img[:,:,2:-2,:] += 0.5 * (img[:,0,:,1:-3,:]-img[:,0,:,3:-1,:]) # Inner terms
    # D_T_img[:,:,0:2,:] += - 0.5 * img[:,0,:,1:3,:] # First two terms
    # D_T_img[:,:,-2:,:] += 0.5 * img[:,0,:,-3:-1,:] # Last two terms

    # Compact v2
    # D_T_img[:,:,2:,1:-1] += 0.5 * img[:,0,:,1:-1,1:-1]
    # D_T_img[:,:,:-2,1:-1] += -0.5 * img[:,0,:,1:-1,1:-1]

    # Explicit v2
    D_T_img[:,:,2:-2,1:-1] += 0.5 * (img[:,0,:,1:-3,1:-1] - img[:,0,:,3:-1,1:-1])
    D_T_img[:,:,-2:,1:-1] += 0.5 * img[:,0,:,-3:-1,1:-1]
    D_T_img[:,:,0:2,1:-1] += -0.5 * img[:,0,:,1:3,1:-1]

    # Col term

    # Old from v1
    # D_T_img[:,:,:,2:-2] += 0.5 * (img[:,1,:,:,1:-3]-img[:,0,:,:,3:-1]) # Inner terms
    # D_T_img[:,:,:,0:2] += - 0.5 * img[:,1,:,:,1:3] # First two terms
    # D_T_img[:,:,:,-2:] += 0.5 * img[:,1,:,:,-3:-1] # Last two terms

    # Compact v2
    # D_T_img[:,:,1:-1, 2:] += 0.5 * img[:,1,:,1:-1,1:-1]
    # D_T_img[:,:,1:-1, :-2] += -0.5 * img[:,1,:,1:-1,1:-1]

    # Explicit v2
    D_T_img[:,:,1:-1,2:-2] += 0.5 * (img[:,1,:,1:-1,1:-3] - img[:,1,:,1:-1,3:-1])
    D_T_img[:,:,1:-1,-2:] += 0.5 * img[:,1,:,1:-1,-3:-1]
    D_T_img[:,:,1:-1,0:2] += -0.5 * img[:,1,:,1:-1,1:3]

    i_d = 2
    if Nz > 1 and reg_z_over_reg > 0: # z-terms
        # Forward z term
        D_T_img[2:-2,:,:,:] += np.sqrt(reg_z_over_reg) * 0.5 * (img[1:-3,i_d,:,:,:]-img[3:-1,i_d,:,:,:])
        D_T_img[0:2,:,:,:] += - np.sqrt(reg_z_over_reg) * 0.5 * img[1:3,i_d,:,:,:]
        D_T_img[-2:,:,:,:] += np.sqrt(reg_z_over_reg) * 0.5 * img[-3:-1,i_d,:,:,:]
        i_d += 1

    if reg_time > 0 and M > 1:
        # Forward time term
        D_T_img[:,2:-2,:,:] += np.sqrt(reg_time) * 0.5 * (img[:,i_d,1:-3,:,:]-img[:,i_d,3:-1,:,:])
        D_T_img[:,0:2,:,:] += -np.sqrt(reg_time) * 0.5 * img[:,i_d,1:3,:,:]
        D_T_img[:,-2:,:,:] += np.sqrt(reg_time) * 0.5 * img[:,i_d,-3:-1,:,:]
        i_d += 1

    return(D_T_img)


