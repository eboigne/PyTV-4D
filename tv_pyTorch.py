import numpy as np
import torch
import tv_pyTorch_2d

def tv_centered(img, mask = []):

    if mask != []:
        img[~mask] = 0

    if len(img) == 2:
        return(tv_pyTorch_2d.tv_centered(img))
    elif (len(img.shape) == 3 and img.shape[0] < 3):
        return(tv_pyTorch_2d.tv_centered(img[0]))

    kernel_row = np.array([[[-0.5], [0], [0.5]]]).astype('float32')
    kernel_row = torch.as_tensor(np.reshape(kernel_row, (1,1)+kernel_row.shape)).cuda()

    kernel_col = np.array([[[-0.5, 0, 0.5]]]).astype('float32')
    kernel_col = torch.as_tensor(np.reshape(kernel_col, (1,1)+kernel_col.shape)).cuda()

    kernel_slice = np.array([[[-0.5]], [[0]], [[0.5]]]).astype('float32')
    kernel_slice = torch.as_tensor(np.reshape(kernel_slice, (1,1)+kernel_slice.shape)).cuda()

    img = np.reshape(img, (1,1)+img.shape).astype('float32')
    img_tensor = torch.as_tensor(img).cuda()
    row_diff_tensor = (torch.zeros_like(img_tensor)).squeeze()
    col_diff_tensor = (torch.zeros_like(img_tensor)).squeeze()
    slice_diff_tensor = (torch.zeros_like(img_tensor)).squeeze()

    row_diff_tensor[:, 1:-1, :] = torch.nn.functional.conv3d(img_tensor, kernel_row, bias=None, stride=1, padding = 0).squeeze()
    col_diff_tensor[:, :, 1:-1] = torch.nn.functional.conv3d(img_tensor, kernel_col, bias=None, stride=1, padding = 0).squeeze()
    slice_diff_tensor[1:-1, :, :] = torch.nn.functional.conv3d(img_tensor, kernel_slice, bias=None, stride=1, padding = 0).squeeze()

    # row_diff_tensor[:,[0, -1]] = 0 # To match CPU explicit versions
    # col_diff_tensor[[0,-1],:] = 0 # To match CPU explicit versions

    # To match CPU explicit versions
    row_diff_tensor[:, :, [0, -1]] = 0
    row_diff_tensor[[0, -1], :, :] = 0
    col_diff_tensor[:, [0,-1],:] = 0
    col_diff_tensor[[0,-1], :,:] = 0
    slice_diff_tensor[:, [0,-1], :] = 0
    slice_diff_tensor[:, :, [0,-1]] = 0

    grad_norms = (torch.zeros_like(img_tensor)).squeeze()
    grad_norms[:, :, :] = torch.sqrt(torch.square(row_diff_tensor[:, :, :])+torch.square(col_diff_tensor[:, :, :])+torch.square(slice_diff_tensor[:, :, :]))
    tv = grad_norms.sum().cpu().detach().numpy().squeeze()
    grad_norms[grad_norms == 0] = np.inf

    G = torch.zeros_like(img_tensor).squeeze()
    G[1:-1, 1:-1, 2:] += col_diff_tensor[1:-1, 1:-1, 1:-1]/grad_norms[1:-1, 1:-1, 1:-1]
    G[1:-1, 1:-1, :-2] += - col_diff_tensor[1:-1, 1:-1, 1:-1]/grad_norms[1:-1, 1:-1, 1:-1]
    G[1:-1, 2:, 1:-1] += row_diff_tensor[1:-1, 1:-1, 1:-1]/grad_norms[1:-1, 1:-1, 1:-1]
    G[1:-1, :-2, 1:-1] += - row_diff_tensor[1:-1, 1:-1, 1:-1]/grad_norms[1:-1, 1:-1, 1:-1]
    G[2:, 1:-1, 1:-1] += slice_diff_tensor[1:-1, 1:-1, 1:-1]/grad_norms[1:-1, 1:-1, 1:-1]
    G[:-2, 1:-1, 1:-1] += - slice_diff_tensor[1:-1, 1:-1, 1:-1]/grad_norms[1:-1, 1:-1, 1:-1]

    G_cpu = G.cpu().detach().numpy().squeeze()
    torch.cuda.empty_cache()
    del G, grad_norms, row_diff_tensor, col_diff_tensor, slice_diff_tensor, img_tensor, kernel_row, kernel_col, kernel_slice

    return (tv, G_cpu)

def tv_hybrid(img, mask = []):

    if mask != []:
        img[~mask] = 0

    if len(img) == 2:
        return(tv_pyTorch_2d.tv_hybrid(img))
    elif (len(img.shape) == 3 and img.shape[0] < 3):
        return(tv_pyTorch_2d.tv_hybrid(img[0]))

    kernel_col = np.array([[[-1,1]]]).astype('float32')
    kernel_col = torch.as_tensor(np.reshape(kernel_col, (1,1)+kernel_col.shape)).cuda()

    kernel_row = np.array([[[-1],[1]]]).astype('float32')
    kernel_row = torch.as_tensor(np.reshape(kernel_row, (1,1)+kernel_row.shape)).cuda()

    kernel_slice = np.array([[[-1]], [[1]]]).astype('float32')
    kernel_slice = torch.as_tensor(np.reshape(kernel_slice, (1,1)+kernel_slice.shape)).cuda()

    img = np.reshape(img, (1,1)+img.shape).astype('float32')
    img_tensor = torch.as_tensor(img).cuda()
    row_diff_tensor = (torch.zeros_like(img_tensor)).squeeze()
    col_diff_tensor = (torch.zeros_like(img_tensor)).squeeze()
    slice_diff_tensor = (torch.zeros_like(img_tensor)).squeeze()

    row_diff_tensor[:, :-1, :] = torch.nn.functional.conv3d(img_tensor, kernel_row, bias=None, stride=1, padding = 0).squeeze()
    col_diff_tensor[:, :, :-1] = torch.nn.functional.conv3d(img_tensor, kernel_col, bias=None, stride=1, padding = 0).squeeze()
    slice_diff_tensor[:-1, :, :] = torch.nn.functional.conv3d(img_tensor, kernel_slice, bias=None, stride=1, padding = 0).squeeze()

    # To match CPU explicit versions
    row_diff_tensor[:, :, -1] = 0
    row_diff_tensor[-1, :, :] = 0
    col_diff_tensor[:, -1, :] = 0
    col_diff_tensor[-1, :, :] = 0
    slice_diff_tensor[:, -1, :] = 0
    slice_diff_tensor[:, :, -1] = 0

    grad_norms = (torch.zeros_like(img_tensor)).squeeze()
    grad_norms[:-1, :-1, :-1] = torch.sqrt(torch.square(row_diff_tensor[1:, :-1, 1:]) + torch.square(row_diff_tensor[:-1, :-1, :-1])
                                           + torch.square(col_diff_tensor[1:, 1:, :-1]) + torch.square(col_diff_tensor[:-1, :-1, :-1])
                                           + torch.square(slice_diff_tensor[:-1, 1:, 1:]) + torch.square(slice_diff_tensor[:-1, :-1, :-1])) / np.sqrt(2)
    tv = grad_norms.sum().cpu().detach().numpy().squeeze()
    grad_norms[grad_norms == 0] = np.inf
    
    G = torch.zeros_like(img_tensor).squeeze()

    G[:-1, :-1, :-1] = - (row_diff_tensor+col_diff_tensor+slice_diff_tensor)[:-1, :-1, :-1]/grad_norms[:-1, :-1, :-1]

    G[:-1, :-1, 1:] += col_diff_tensor[:-1, :-1, :-1]/grad_norms[:-1, :-1, :-1]
    G[:-1, 1:, :-1] += row_diff_tensor[:-1, :-1, :-1]/grad_norms[:-1, :-1, :-1]
    G[1:, :-1, :-1] += slice_diff_tensor[:-1, :-1, :-1]/grad_norms[:-1, :-1, :-1]

    G[1:, 1:, :-1] += -col_diff_tensor[1:, 1:, :-1]/grad_norms[:-1, :-1, :-1]
    G[1:, :-1, 1:] += -row_diff_tensor[1:, :-1, 1:]/grad_norms[:-1, :-1, :-1]
    G[:-1, 1:, 1:] += -slice_diff_tensor[:-1, 1:, 1:]/grad_norms[:-1, :-1, :-1]

    G[1:, 1:, 1:] += (row_diff_tensor[1:, :-1, 1:] + col_diff_tensor[1:, 1:, :-1] + slice_diff_tensor[:-1, 1:, 1:])/grad_norms[:-1, :-1, :-1]

    G_cpu = G.cpu().detach().numpy().squeeze()
    torch.cuda.empty_cache()
    del G, grad_norms, row_diff_tensor, col_diff_tensor, slice_diff_tensor, img_tensor, kernel_row, kernel_col, kernel_slice
    
    return(tv, G_cpu)


def tv_downwind(img, mask = []):

    if mask != []:
        img[~mask] = 0

    if len(img) == 2:
        return(tv_pyTorch_2d.tv_downwind(img))
    elif (len(img.shape) == 3 and img.shape[0] < 3):
        return(tv_pyTorch_2d.tv_downwind(img[0]))

    kernel_col = np.array([[[-1,1]]]).astype('float32')
    kernel_col = torch.as_tensor(np.reshape(kernel_col, (1,1)+kernel_col.shape)).cuda()

    kernel_row = np.array([[[-1],[1]]]).astype('float32')
    kernel_row = torch.as_tensor(np.reshape(kernel_row, (1,1)+kernel_row.shape)).cuda()

    kernel_slice = np.array([[[-1]], [[1]]]).astype('float32')
    kernel_slice = torch.as_tensor(np.reshape(kernel_slice, (1,1)+kernel_slice.shape)).cuda()

    img = np.reshape(img, (1,1)+img.shape).astype('float32')
    img_tensor = torch.as_tensor(img).cuda()
    row_diff_tensor = (torch.zeros_like(img_tensor)).squeeze()
    col_diff_tensor = (torch.zeros_like(img_tensor)).squeeze()
    slice_diff_tensor = (torch.zeros_like(img_tensor)).squeeze()

    row_diff_tensor[:, :-1, :] = torch.nn.functional.conv3d(img_tensor, kernel_row, bias=None, stride=1, padding = 0).squeeze()
    col_diff_tensor[:, :, :-1] = torch.nn.functional.conv3d(img_tensor, kernel_col, bias=None, stride=1, padding = 0).squeeze()
    slice_diff_tensor[:-1, :, :] = torch.nn.functional.conv3d(img_tensor, kernel_slice, bias=None, stride=1, padding = 0).squeeze()

    # To match CPU explicit versions
    row_diff_tensor[:, :, -1] = 0
    row_diff_tensor[-1, :, :] = 0
    col_diff_tensor[:, -1, :] = 0
    col_diff_tensor[-1, :, :] = 0
    slice_diff_tensor[:, -1, :] = 0
    slice_diff_tensor[:, :, -1] = 0

    grad_norms = (torch.zeros_like(img_tensor)).squeeze()
    grad_norms[:-1, :-1, :-1] = torch.sqrt(torch.square(row_diff_tensor[1:, :-1, 1:])+torch.square(col_diff_tensor[1:, 1:, :-1])+torch.square(slice_diff_tensor[:-1, 1:, 1:]))
    tv = grad_norms.sum().cpu().detach().numpy().squeeze()
    grad_norms[grad_norms == 0] = np.inf
    
    G = torch.zeros_like(img_tensor).squeeze()
    G[1:, 1:, 1:] = (row_diff_tensor[1:, :-1, 1:] + col_diff_tensor[1:, 1:, :-1] + slice_diff_tensor[:-1, 1:, 1:]) / grad_norms[:-1, :-1, :-1]
    G[1:, :-1, 1:] += - row_diff_tensor[1:, :-1, 1:] / grad_norms[:-1, :-1, :-1]
    G[1:, 1:, :-1] += - col_diff_tensor[1:, 1:, :-1] / grad_norms[:-1, :-1, :-1]
    G[:-1, 1:, 1:] += - slice_diff_tensor[:-1, 1:, 1:] / grad_norms[:-1, :-1, :-1]

    G_cpu = G.cpu().detach().numpy().squeeze()
    torch.cuda.empty_cache()
    del G, grad_norms, row_diff_tensor, col_diff_tensor, slice_diff_tensor, img_tensor, kernel_row, kernel_col, kernel_slice
    
    return(tv, G_cpu)


def tv_upwind(img, mask = []):

    if mask != []:
        img[~mask] = 0

    if len(img) == 2:
        return(tv_pyTorch_2d.tv_upwind(img))
    elif (len(img.shape) == 3 and img.shape[0] < 3):
        return(tv_pyTorch_2d.tv_upwind(img[0]))

    kernel_col = np.array([[[-1,1]]]).astype('float32')
    kernel_col = torch.as_tensor(np.reshape(kernel_col, (1,1)+kernel_col.shape)).cuda()

    kernel_row = np.array([[[-1],[1]]]).astype('float32')
    kernel_row = torch.as_tensor(np.reshape(kernel_row, (1,1)+kernel_row.shape)).cuda()

    kernel_slice = np.array([[[-1]], [[1]]]).astype('float32')
    kernel_slice = torch.as_tensor(np.reshape(kernel_slice, (1,1)+kernel_slice.shape)).cuda()

    img = np.reshape(img, (1,1)+img.shape).astype('float32')
    img_tensor = torch.as_tensor(img).cuda()
    row_diff_tensor = (torch.zeros_like(img_tensor)).squeeze()
    col_diff_tensor = (torch.zeros_like(img_tensor)).squeeze()
    slice_diff_tensor = (torch.zeros_like(img_tensor)).squeeze()

    # img_tensor of size: (N,Cin,H,W), N: batch size, Cin: number of input channels.
    row_diff_tensor[:, :-1, :] = torch.nn.functional.conv3d(img_tensor, kernel_row, bias=None, stride=1, padding = 0).squeeze()
    col_diff_tensor[:, :, :-1] = torch.nn.functional.conv3d(img_tensor, kernel_col, bias=None, stride=1, padding = 0).squeeze()
    slice_diff_tensor[:-1, :, :] = torch.nn.functional.conv3d(img_tensor, kernel_slice, bias=None, stride=1, padding = 0).squeeze()

    # To match CPU explicit versions
    row_diff_tensor[:, :, -1] = 0
    row_diff_tensor[-1, :, :] = 0
    col_diff_tensor[:, -1, :] = 0
    col_diff_tensor[-1, :, :] = 0
    slice_diff_tensor[:, -1, :] = 0
    slice_diff_tensor[:, :, -1] = 0

    grad_norms = (torch.zeros_like(img_tensor)).squeeze()
    grad_norms = torch.sqrt(torch.square(row_diff_tensor)+torch.square(col_diff_tensor)+torch.square(slice_diff_tensor))
    tv = grad_norms.sum().cpu().detach().numpy().squeeze()
    grad_norms[grad_norms == 0] = np.inf
    
    G = torch.zeros_like(img_tensor).squeeze()
    G[:-1, :-1, :-1] =  - (row_diff_tensor+col_diff_tensor+slice_diff_tensor)[:-1, :-1, :-1]/grad_norms[:-1, :-1, :-1]
    G[:-1, 1:, :-1] += row_diff_tensor[:-1, :-1, :-1]/grad_norms[:-1, :-1, :-1]
    G[:-1, :-1, 1:] += col_diff_tensor[:-1, :-1, :-1]/grad_norms[:-1, :-1, :-1]
    G[1:, :-1, :-1] += slice_diff_tensor[:-1, :-1, :-1]/grad_norms[:-1, :-1, :-1]

    G_cpu = G.cpu().detach().numpy().squeeze()
    torch.cuda.empty_cache()
    del G, grad_norms, row_diff_tensor, col_diff_tensor, slice_diff_tensor, img_tensor, kernel_row, kernel_col, kernel_slice
    
    return(tv, G_cpu)
