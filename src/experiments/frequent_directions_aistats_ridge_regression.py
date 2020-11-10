import numpy as np
import sys
import os
from pathlib import Path
path = Path(os.getcwd())
sys.path.append(str(path.parent.parent/ 'src'))
from frequent_directions import FastFrequentDirections, RobustFrequentDirections

class FDRidge:
    
    def __init__(self, fd_dim:int,fd_mode='FD',gamma=1.0):
        """
        Approximate ridge regression using the FD sketch.

        fd_dim (int) - the number of rows retained in the FD sketch.
        fd_mode (str) : mode for frequent directions FD or RFD.
        alpha : float - the regularisation parameter for ridge regression.
        """
        self.fd_dim       = fd_dim
        self.fd_mode      = fd_mode
        if self.fd_mode not in ['FD', 'RFD']:
            raise NotImplementedError('Only F(ast) and R(obust) FD methods are supported.')
        self.gamma        = gamma
        self.is_fitted    = False

    def _sketch(self, X):
        '''
        Private function for calling the sketch methods
        '''
        if self.fd_mode == 'FD':
            sketcher = FastFrequentDirections(X.shape[1],sketch_dim=self.fd_dim)
        elif self.fd_mode == 'RFD':
            sketcher = RobustFrequentDirections(X.shape[1],sketch_dim=self.fd_dim)
        sketcher.fit(X,batch_size=self.fd_dim)
        self.sketch = sketcher.sketch
        self.alpha = sketcher.delta # == 0 if using FastFrequentDirections so can use self.gamma + self.alpha everywhere 
        self.is_fitted = True
    
    def fit(self,X,y):
        '''
        Fits the ridge regression model on data X with targets y
        '''
        d = X.shape[1]
        self._sketch(X)
        H = self.sketch.T@self.sketch + (self.gamma+self.alpha)*np.eye(d)
        self.coef_ = np.linalg.solve(H, X.T@y)
        self.H_inv = np.linalg.pinv(H)
        
    def iterate(self,X,y,iterations=10):
        '''
        Fits the iterated ridge model with FD
        '''
        d = X.shape[1]
        w = np.zeros((d,1),dtype=float)
        all_w = np.zeros((d,iterations))
        XTy = (X.T@y).reshape(-1,1)
        
        # Fit the FD
        if not self.is_fitted:
            self._sketch(X)
        H = self.sketch.T@self.sketch + (self.gamma+self.alpha)*np.eye(d)
        H_inv = np.linalg.pinv(H)
        for it in range(iterations):
            grad = X.T@(X@w) + self.gamma*w - XTy
            w += - H_inv@grad
            all_w[:,it] = np.squeeze(w)
        return np.squeeze(w), all_w
    
    def get_bias(self,X,w0):
        '''
        Returns the bias of the estimate
        '''
        # sketcher = FastFrequentDirections(X.shape[1],m=self.fd_dim)
        # sketcher.fit(X,batch_size=self.fd_dim)
        # B = sketcher.sketch
        if not self.is_fitted:
            self._sketch(X)
        H = self.sketch.T@self.sketch + (self.gamma + self.alpha)*np.eye(X.shape[1])
        return np.linalg.pinv(H)@(X.T@(X@w0)) - w0

    def get_variance(self,X):
        '''
        Returns the variance term: ||A H_gamma^{-1}||_F^2
        '''
        return np.linalg.norm(X@self.H_inv,ord='fro')**2
    
        