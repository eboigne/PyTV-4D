import numpy as np
import os

def cameraman():
    '''
    A function that provides the grayscale cameraman standard image, of dimensions 256 x 256.

    Returns
    -------
    np.ndarray
    '''

    return(np.load(os.path.join(os.path.dirname(__file__), 'media','cameraman.npy')))

