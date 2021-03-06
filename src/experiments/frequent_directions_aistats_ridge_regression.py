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
import sys
import os
from timeit import default_timer as timer
from pathlib import Path
path = Path(os.getcwd())
sys.path.append(str(path.parent.parent/ 'src'))
from frequent_directions import FastFrequentDirections, RobustFrequentDirections

class FDRidge:
    
    def __init__(self, fd_dim:int,fd_mode='FD',gamma=1.0,batch_size=None):
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
        if batch_size == None:
            self.batch_size = self.fd_dim
        else:
            self.batch_size = batch_size

    def _sketch(self, X):
        '''
        Private function for calling the sketch methods
        '''
        if self.fd_mode == 'FD':
            sketcher = FastFrequentDirections(X.shape[1],sketch_dim=self.fd_dim)
        elif self.fd_mode == 'RFD':
            sketcher = RobustFrequentDirections(X.shape[1],sketch_dim=self.fd_dim)
        sketcher.fit(X,batch_size=self.batch_size)
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

    def fast_iterate(self,X,y,iterations):
        """
        Performs the iterations of ifdrr efficiently in small space and time.
        """

        # * Initialisation not timed
        d = X.shape[1]
        w = np.zeros((d,1),dtype=float)
        all_w = np.zeros((d,iterations))
        if self.fd_mode == 'FD':
            sketcher = FastFrequentDirections(X.shape[1],sketch_dim=self.fd_dim)
        elif self.fd_mode == 'RFD':
            sketcher = RobustFrequentDirections(X.shape[1],sketch_dim=self.fd_dim)
        measurables = {
        'sketch time' : None,
        'all_times'   : np.zeros(iterations+1,dtype=float),
        'gradients'   : np.zeros((d,iterations),dtype=float),
        'updates'     : np.zeros((d,iterations),dtype=float),
        'sketch'      : None
        }

        # ! Sketching
        TIMER_START = timer()
        sketcher.fit(X,batch_size=self.batch_size)
        SKETCH_TIME = timer() - TIMER_START
        _, SigSq, Vt, implicit_reg = sketcher.get()
        V = Vt.T
        invTerm = (1./(SigSq + implicit_reg + self.gamma )).reshape(-1,1)

        # Extra parameters we may need 
        XTy = (X.T@y).reshape(-1,1)

        # * This lambda function evaluates H^{-1}g efficiently for gradient vector g
        H_inv_grad = lambda g, vtg : (1/self.gamma )*(g - V@vtg) + V@(invTerm*vtg)

        for it in range(iterations):   
            grad = X.T@(X@w) + self.gamma *w - XTy
            VTg = Vt@grad
            update = H_inv_grad(grad, VTg)
            w += - update
            all_w[:,it] = np.squeeze(w)
            measurables['all_times'][it+1] = timer() - TIMER_START
            measurables['gradients'][:,it] = np.squeeze(grad)
            measurables['updates'][:,it] = np.squeeze(update)
        measurables['sketch time'] = SKETCH_TIME
        measurables['sketch'] = sketcher.sketch
        return np.squeeze(w), all_w, measurables
    
    
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
    
        