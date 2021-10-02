import numpy as np
import os

def cameraman():
    return(np.load(os.path.join(os.path.dirname(__file__), 'media','cameraman.npy')))

