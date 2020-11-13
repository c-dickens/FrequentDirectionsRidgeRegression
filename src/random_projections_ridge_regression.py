# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from ridge_regression import RidgeRegression

class RPRR(RidgeRegression):
    def __init__(self, rp_dim:int,gamma,rp_method='Classical',rp_mode='Gaussian',solve_method='Exact'):
        """
        Approximate ridge regression using random projections.

        rp_dim (int)    : the number of rows retained in the random projection.
        rp_method (str) : The method to use (Classical or Hessian) 
        rp_mode (str)   : sketch mode used to decide on the sketch.
        gamma : float   : the regularisation parameter for ridge regression.
        """
        self.rp_dim       = rp_dim
        self.rp_method    = rp_method
        self.rp_mode      = rp_mode
        self.solve_method = solve_method
        if self.rp_method not in ['Classical', 'Hessian']:
            raise NotImplementedError('Only Classical and Hessian methods are supported.')
        if self.rp_mode not in ['Gaussian', 'SJLT']:
            raise NotImplementedError('Only Gaussian and SJLT modes are supported.')
        if self.solve_method not in ['Exact','ShiWoodbury']:
            raise NotImplementedError('Only Exact and ShiWoodbury methods are implemented')
        if not isinstance(gamma,list):
            self.gamma = [gamma]
        else:
            self.gamma = gamma
    
    def fit(self,data,targets):
        '''
        Fits the ridge model to gamma (which can be a single float or list/ndarray).
        '''
        _,d = data.shape
        X, y = self._preprocess_data(data, targets)
        # Perform self._sketch() once to avoid recomputing
        self._sketch(np.c_[X,y]) 
        self.params = {
            a : {
            'coef_'   :  np.zeros(d)
                     } for a in self.gamma}
        # Can potentially parallelise this loop for better time performance
        for a in self.gamma:
            # There should be either 1 or n_targets penalties
            gamma_reg = np.asarray(a, dtype=X.dtype).ravel() # shift regularisation by the alpha parameter
            assert gamma_reg.size == 1
            weights = self._solve(X,y,gamma_reg)
            self.params[a]['coef_'] = weights
        
    
            
    def _solve(self, X, y,reg):
        '''
        Obtains the inverse term explicitly
        '''
        if self.rp_method == 'Classical':
            #B,z = self._fit_classical(X,y) 
            B, z = self.SX, self.Sy
            if self.solve_method == 'ShiWoodbury':
                return self._small_space_solve(X,y,reg)
            else:
                return (np.linalg.pinv(B.T@B + reg*np.eye(X.shape[1])))@(B.T@z) 
        if self.rp_method == 'Hessian':
            B,z = self.SX, y ##self._fit_hessian(X,y)
            if self.solve_method == 'ShiWoodbury':
                return self._small_space_solve(X,y,reg)
            else:
                return (np.linalg.pinv(B.T@B + reg*np.eye(X.shape[1])))@(X.T@z)

    def _sketch(self,Xy,seed=10):
        '''
        Performs the sketch depending on the chosen mode.
        '''
        np.random.seed(seed)
        if self.rp_mode == 'Gaussian':
            sk = self._gaussian_projection(Xy,self.rp_dim)
        elif self.rp_mode == 'SJLT':
            sk = self._sparse_projection(Xy,self.rp_dim)
        else:
            raise NotImplementedError
        self.SX, self.Sy = sk[:,:-1], sk[:,-1]
            
    def _sparse_projection(self,mat,sparsity=1,random_seed=10):
        """
        Performs the sparse johnson lindenstrauss transform of Kane and Nelson
        """
        [n,_] = mat.shape
        sketch = np.zeros((self.rp_dim ,n),dtype=float)
        for i in range(n):
            nnz_loc = np.random.choice(self.rp_dim ,size=sparsity,replace=False)
            nnz_sign = np.random.choice([-1,1],size=sparsity,replace=True)
            sketch[nnz_loc,i] = nnz_sign
        self.sketch_mat = sketch
        return (1./np.sqrt(sparsity))*sketch@mat

    def _gaussian_projection(self,mat,random_seed=10):
        """
        Performs the sparse johnson lindenstrauss transform of Kane and Nelson
        """
        [n,_] = mat.shape
        np.random.seed(random_seed)
        S = np.random.randn(self.rp_dim,n) / np.sqrt(self.rp_dim)
        self.sketch_mat = S
        return S@mat
    
    def _small_space_solve(self,X,y,reg):
        _,S,Vt = np.linalg.svd(self.SX,full_matrices=False)
        V = Vt.T
        inv_diag = np.linalg.pinv(np.diag(S**2 + reg)) #1./(S**2 + self.gamma)
        if self.rp_method == 'Classical':
            z = self.SX.T@self.Sy
        else:
            z = X.T@y
        first_term = (V@(inv_diag))@(Vt@z)
        second_term = (1./reg)*z
        third_term = (1./reg)*V@(Vt@z)
        return first_term + second_term - third_term