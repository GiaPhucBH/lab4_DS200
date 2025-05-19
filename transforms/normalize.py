from typing import Tuple
import numpy as np

class Normalize:

    def __init__(self,mean: Tuple, std: Tuple) -> None:
        self.mean = mean
        self.std = std

    def transform(self,matrix: np.ndarray) -> np.ndarray:
        
        shape = matrix.shape
        matrix = matrix/255.0
        matrix = matrix.transpose(2,0,1)
        r = matrix[0]
        g = matrix[1]
        b = matrix[2]

        r = (r-self.mean[0])/self.std[0]
        g = (g-self.mean[1])/self.std[1]
        b = (b-self.mean[2])/self.std[2]

        matrix[0] = r
        matrix[1] = g
        matrix[2] = b

        matrix = matrix.transpose(1,2,0)
        assert matrix.shape == shape
        return matrix