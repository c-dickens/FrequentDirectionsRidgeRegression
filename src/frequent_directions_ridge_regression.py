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
from frequent_directions import FastFrequentDirections,RobustFrequentDirections


class FDRR(RidgeRegression):
    """
    Implements the Frequent Directions Reidge Regression
    """
    def __init__(self, fd_dim:int,gamma,fd_mode='RFD',solve_method='Exact'):
        """
        Approximate ridge regression using the FD sketch.

        fd_dim (int) - the number of rows retained in the FD sketch.
        fd_mode (str) : mode for frequent directions FD or RFD.
        gamma : float - the regularisation parameter for ridge regression.
        """
        self.fd_dim       = fd_dim
        self.fd_mode      = fd_mode
        self.solve_method = solve_method
        if self.fd_mode not in ['FD', 'RFD']:
            raise NotImplementedError('Only F(ast) and R(obust) FD methods are supported.')
        if self.solve_method not in ['Exact','ShiWoodbury']:
            raise NotImplementedError('Only Exact and ShiWoodbury methods are implemented')
        self.gamma        = gamma
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
        self._sketch(X) 
        self.params = {
            a : {
            'coef_'   :  np.zeros(d)
                     } for a in self.gamma}
        # Can potentially parallelise this loop for better time performance
        for a in self.gamma:
            # There should be either 1 or n_targets penalties
            gamma_reg = np.asarray(a+self.alpha, dtype=X.dtype).ravel() # shift regularisation by the alpha parameter
            assert gamma_reg.size == 1
            weights = self._solve(X,y,gamma_reg)
            #intercept = self._set_intercept(weights,X_offset, y_offset, X_scale)
            self.params[a]['coef_'] = weights
            #self.params[a]['intercept'] = intercept
            
    def _sketch(self,X):
        if self.fd_mode == 'FD':
            sketcher = FastFrequentDirections(X.shape[1],sketch_dim=self.fd_dim)
        elif self.fd_mode == 'RFD':
            sketcher = RobustFrequentDirections(X.shape[1],sketch_dim=self.fd_dim)
        sketcher.fit(X,batch_size=self.fd_dim)
        self.V = sketcher.Vt.T
        self.SigmaSquared = sketcher.sigma_squared
        self.sketch_mat = sketcher.sketch
        self.alpha = sketcher.delta # == 0 if using FastFrequentDirections so can use self.gamma + self.alpha everywhere 
        
    def _solve(self,X,y,reg):
        if self.solve_method == 'ShiWoodbury':
            return self._small_space_solve(X,y,reg)
        else:
            # The exact / naive method
            return (np.linalg.pinv(self.sketch_mat.T@self.sketch_mat + reg*np.eye(X.shape[1])))@(X.T@y)
        
    def _small_space_solve(self,X,y,reg):
        '''
        Solves in small space using the algorithm of shi and phillips.
        This is just Woodbury identity but over the basis and singular values rather than 
        the raw sketch.
        Using the basis seems more numerically than using the sketch for some reason...?
        '''
        ATy = X.T@y
        inv_diag = np.linalg.pinv(np.diag(self.SigmaSquared + reg)) #1./(S**2 + self.gamma)
        first_term = (self.V@(inv_diag))@(self.V.T@ATy)
        second_term = (1./reg)*ATy
        third_term = (1./reg)*self.V@(self.V.T@ATy)
        return first_term + second_term - third_term