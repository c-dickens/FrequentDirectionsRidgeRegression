import numpy as np
from scipy import linalg

class FrequentDirections:
    def __init__(self,d,sketch_dim=8):
        '''
        Class wrapper for all FD-type methods

        __rotate_and_reduce__ is not defined for the standard FrequentDirections but is for the 
        subsequent subclasses which inherit from FrequentDirections.
        '''
        self.d = d
        self.delta = 0. # For RFD

        if sketch_dim is not None:
            self.sketch_dim = sketch_dim
        self.sketch = np.zeros((self.sketch_dim,self.d),dtype=float)
        self.Vt = np.zeros((self.sketch_dim,self.d),dtype=float)
        self.sigma_squared = np.zeros(self.sketch_dim,dtype=float) 
        
    def fit(self,X,batch_size=1):
        '''
        Fits the FD transform to dataset X
        '''
        n = X.shape[0]
        for i in range(0,n,batch_size):
            batch = X[i:i+batch_size,:]
            aux = np.concatenate((self.sketch,batch),axis=0)
            try:
                _, s, self.Vt = np.linalg.svd(aux, full_matrices=False)
            except np.linalg.LinAlgError:
                _, s, self.Vt = linalg.svd(aux, full_matrices=False, lapack_driver='gesvd')
            self.sigma_squared = s**2 
            self.__rotate_and_reduce__() 
            self.sketch = self.Vt * np.sqrt(self.sigma_squared).reshape(-1, 1)
        
    def get(self):
        return self.sketch, self.sigma_squared, self.Vt
        
            
class FastFrequentDirections(FrequentDirections):
    '''
    Implements the fast version of FD by doubling space
    '''
    
    def __rotate_and_reduce__(self):
        self.sigma_squared = self.sigma_squared[:self.sketch_dim] - self.sigma_squared[self.sketch_dim]
        self.Vt = self.Vt[:self.sketch_dim]
        
class RobustFrequentDirections(FrequentDirections):
    '''
    Implements the RFD version of FD by maintaining counter self.delta.
    Still operates in the `fast` regimen by doubling space, as in 
    FastFrequentDirections
    '''
    def __rotate_and_reduce__(self):
        self.delta += self.sigma_squared[self.sketch_dim]/2.
        self.sigma_squared = self.sigma_squared[:self.sketch_dim] - self.sigma_squared[self.sketch_dim]
        self.Vt = self.Vt[:self.sketch_dim]
