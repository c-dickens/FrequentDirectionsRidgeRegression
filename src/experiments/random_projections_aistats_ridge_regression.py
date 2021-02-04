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
from timeit import default_timer as timer

class RPRidge:
    
    def __init__(self,rp_dim:int,rp_mode='Sparse',gamma=1.0):
        self.rp_dim       = rp_dim
        self.rp_mode      = rp_mode
        self.gamma        = gamma

  
    def iterate_single(self,X,y,iterations=10,seed=100):
        '''
        Fits the iterated ridge model with FD
        '''
        d = X.shape[1]
        w = np.zeros((d,1),dtype=float)
        all_w = np.zeros((d,iterations))
        XTy = (X.T@y).reshape(-1,1)
        
        # Fit the FD
        SA = self._get_sketch(X,seed)
        H = SA.T@SA + (self.gamma)*np.eye(d)
        H_inv = np.linalg.pinv(H)
        for it in range(iterations):

            grad = X.T@(X@w) + self.gamma*w - XTy
            w += - H_inv@grad
            all_w[:,it] = np.squeeze(w)
        return np.squeeze(w), all_w

    def iterate_multiple(self,X,y,iterations=10):
        '''
        Fits the iterated ridge model with FD
        '''
        d = X.shape[1]
        w = np.zeros((d,1),dtype=float)
        all_w = np.zeros((d,iterations))
        XTy = (X.T@y).reshape(-1,1)
        
        for it in range(iterations):
            SA = self._get_sketch(X,seed=it)
            H = SA.T@SA + (self.gamma)*np.eye(d)
            H_inv = np.linalg.pinv(H)
            grad = X.T@(X@w) + self.gamma*w - XTy
            w += - H_inv@grad
            all_w[:,it] = np.squeeze(w)
        return np.squeeze(w), all_w

    def _naive_hessian_update(self, sketched_data, vec):
        """
        Naively performs the hessian update by evaluating the approximate Hessian 
        and inverting.
        """
        H = sketched_data.T@sketched_data + (self.gamma)*np.eye(sketched_data.shape[1])
        H_inv = np.linalg.pinv(H)
        return H_inv@vec
    
    def _linear_solve_hessian_update(self, sketched_data, vec):
        """
        Implements the hessian_inv@gradient efficiently by using Woodbury.
        """
        K = sketched_data@sketched_data.T / self.gamma + np.eye(sketched_data.shape[0])
        u = (1/self.gamma)*vec - (sketched_data.T / self.gamma**2)@(np.linalg.solve(K,sketched_data@vec))
        return u

    def iterate_multiple_timing(self, X, y, iterations=10):
        """
        Fits the iterated ridge model with a new sketch every iteration
        """
        # * Initialisation not timed
        d = X.shape[1]
        w = np.zeros((d,1),dtype=float)
        all_w = np.zeros((d,iterations))
        if self.rp_dim < d:
            update_method = self._linear_solve_hessian_update
        else:
            update_method = self._naive_hessian_update

        measurables = {
        'sketch time' : None,
        'all_times'   : np.zeros(iterations+1,dtype=float),
        'gradients'   : np.zeros((d,iterations),dtype=float),
        'updates'     : np.zeros((d,iterations),dtype=float),
        'sketch'      : None
        }

        TIMER_START = timer()
        XTy = (X.T@y).reshape(-1,1)
        for it in range(iterations):
            SA = self._get_sketch(X,seed=it)
            grad = X.T@(X@w) + self.gamma*w - XTy
            update = update_method(SA, grad)
            w += - update 
            all_w[:,it] = np.squeeze(w)
            measurables['all_times'][it+1] = timer() - TIMER_START
        return np.squeeze(w), all_w, measurables


    def iterate_single_timing(self, X, y, iterations=10, seed=100):
        """
        Fits the iterated ridge model with a new sketch every iteration
        """
        # * Initialisation not timed
        d = X.shape[1]
        w = np.zeros((d,1),dtype=float)
        all_w = np.zeros((d,iterations))
        if self.rp_dim < d:
            update_method = self._linear_solve_hessian_update
        else:
            update_method = self._naive_hessian_update

        measurables = {
        'sketch time' : None,
        'all_times'   : np.zeros(iterations+1,dtype=float),
        'gradients'   : np.zeros((d,iterations),dtype=float),
        'updates'     : np.zeros((d,iterations),dtype=float),
        'sketch'      : None
        }

        TIMER_START = timer()
        XTy = (X.T@y).reshape(-1,1)
        SA = self._get_sketch(X,seed)
        for it in range(iterations):
            grad = X.T@(X@w) + self.gamma*w - XTy
            update = update_method(SA, grad)
            w += - update 
            all_w[:,it] = np.squeeze(w)
            measurables['all_times'][it+1] = timer() - TIMER_START
        return np.squeeze(w), all_w, measurables

    def _get_sketch(self,data,seed=10):
        '''
        Performs the sketch depending on the chosen mode.
        '''
        np.random.seed(seed)
        if self.rp_mode == 'Gaussian':
            return self._gaussian_projection(data,self.rp_dim)
        elif self.rp_mode == 'SJLT':
            return self._sparse_projection(data,self.rp_dim)
        else:
            raise NotImplementedError

        


    def fit_classical(self,X,y):
        '''
        Fits the ridge regression model on data X with targets y
        '''
        d = X.shape[1]
        data = np.c_[X,y]
        #S_data = self._sparse_projection(data,self.rp_dim)
        #S_data = self._gaussian_projection(data,self.rp_dim)
        S_data = self._get_sketch(data)
        SX = S_data[:,:-1]
        Sy = S_data[:,-1]
        H_est = SX.T@SX + self.gamma*np.eye(d)
        self.H = H_est
        self.classical_coef_ = np.linalg.solve(H_est,SX.T@Sy)
        
    def fit_hessian_sketch(self,X,y):
        '''
        Fits the ridge regression model on data X with targets y
        '''
        d = X.shape[1]
        #SX  = self._gaussian_projection(X,self.rp_dim)
        #SX = self._sparse_projection(X,self.rp_dim)
        SX = self._get_sketch(X)
        H_est = SX.T@SX + self.gamma*np.eye(d)
        self.hessian_coef_ = np.linalg.solve(H_est,X.T@y)
    
    def get_classical_bias(self,X,w0):
        '''
        Returns the bias of the estimate
        '''
        return (self.gamma)*np.linalg.norm(np.linalg.pinv(self.H)@w0)
        #return (np.linalg.pinv(self.H)@(self.H - self.gamma*np.eye(X.shape[1])))@w0 - w0

    def get_classical_variance(self,X):
        '''
        Returns the variance term: ||S.T@S@A H_gamma^{-1}||_F^2
        '''
        S = self.sketch_mat
        #return np.linalg.norm(np.linalg.pinv(self.H)@(X.T@(S.T@S)),ord='fro')**2
        return np.linalg.norm( S.T@(S@(X@np.linalg.pinv(self.H)))  ,ord='fro')**2

    def get_hessian_sketch_bias(self,X,w0):
        '''
        Returns the bias of the Hessian sketch method for regression
        '''
        return np.linalg.pinv(self.H)@(X.T@(X@w0)) - w0

    def get_hessian_sketch_variance(self,X):
        '''
        Returns the variance term: ||A H_gamma^{-1}||_F^2
        '''
        return np.linalg.norm(X@np.linalg.pinv(self.H),ord='fro')**2

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
        Performs the dense gaussian random projection.
        """
        [n,_] = mat.shape
        np.random.seed(random_seed)
        S = np.random.randn(self.rp_dim,n) / np.sqrt(self.rp_dim)
        self.sketch_mat = S
        return S@mat
    
    
        