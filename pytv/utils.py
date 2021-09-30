import numpy as np
import os

def lenna():
    return(np.load(os.path.join(os.path.dirname(__file__), 'media','Lenna.npy')))
