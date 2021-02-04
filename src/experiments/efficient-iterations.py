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

from data_factory import DataFactory
from frequent_directions_aistats_ridge_regression import FDRidge
from random_projections_aistats_ridge_regression import RPRidge
from sklearn.linear_model import Ridge,LinearRegression
from plot_config import fd_params, rfd_params, gauss_single_params, sjlt_single_params, gauss_ihs_params, sjlt_ihs_params
import numpy as np
from math import floor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.kernel_approximation import RBFSampler
import matplotlib.pyplot as plt

def get_errors(arr,x):
    e = [np.linalg.norm(arr[:,i] - x)/np.linalg.norm(x) for i in range(arr.shape[1])]
    e.insert(0,1)
    return e


def synthetic_real_experiment(data_name,gamma_reg,ax=None):
    gamma = gamma_reg
    n = 10000
    ds = DataFactory(n=n)
    if data_name == 'CoverType':
        X,y = ds.fetch_forest_cover()
        rff_features = True
    elif data_name == 'w8a':
        X,y = ds.fetch_w8a()
        rff_features = True
    elif data_name == 'CaliforniaHousing':
        feature_size = 4000
        _X, _y = fetch_california_housing(return_X_y=True)
        X_train, X_test, y, y_test = train_test_split(_X, _y, test_size=0.4, random_state=42)
        rbf_feature = RBFSampler(gamma=0.0001, random_state=100,n_components=feature_size)
        X = rbf_feature.fit_transform(X_train)
        rff_features = False
    else:
        X,y = ds.fetch_year_predictions()
        rff_features = True


    # Whether to fit fourier features
    if rff_features:
        X, y = X[:n], y[:n]
        X = ds.feature_expansion(X,n_extra_features=1024)
    d = X.shape[1]



    # Optimal solution
    print('#'*60)
    print('Solving exactly')
    H = X.T@X + gamma*np.eye(d)
    x_opt = np.linalg.solve(H,X.T@y)

    # Iterate the FD regression
    iterations = 10
    m = int(2**8)

    print('#'*40)
    print('#'*10, '\t FREQUENT DIRECTIONS \t', '#'*10)
    fdr = FDRidge(fd_dim=m,gamma=gamma)
    _, all_x,fd_measured = fdr.fast_iterate(X,y,iterations)
    
    print('#'*10, '\t ROBUST FREQUENT DIRECTIONS \t', '#'*10)
    rfdr = FDRidge(fd_dim=m,fd_mode='RFD',gamma=gamma)
    _, rfd_all_x, rfd_measured = rfdr.fast_iterate(X,y,iterations)

    print('#'*10, '\t GAUSS SINGLE \t', '#'*10)
    gauss_single = RPRidge(rp_dim=m,rp_mode='Gaussian',gamma=gamma)
    _, gauss_single_all_x, gauss_single_measured = gauss_single.iterate_single_timing(X,y)

    print('#'*10, '\t SJLT SINGLE \t', '#'*10)
    sjlt_single = RPRidge(rp_dim=m,rp_mode='SJLT',gamma=gamma)
    _, sjlt_single_all_x, sjlt_single_measured = sjlt_single.iterate_single_timing(X,y)

    print('#'*10, '\t GAUSS IHS \t', '#'*10)
    ihs_gauss = RPRidge(rp_dim=5*m,rp_mode='Gaussian',gamma=gamma)
    _, ihs_gauss_all_x, ihs_gauss_measured= ihs_gauss.iterate_multiple_timing(X,y)

    print('#'*10, '\t SJLT IHS \t', '#'*10)
    ihs_sjlt = RPRidge(rp_dim=m,rp_mode='SJLT',gamma=gamma)
    _, ihs_sjlt_all_x, ihs_sjlt_measured = ihs_sjlt.iterate_multiple_timing(X,y)

    # Measurement arrays
    fd_errors = np.zeros(iterations)
    rfd_errors = np.zeros_like(fd_errors,dtype=float)
    gauss_single_errors = np.zeros_like(fd_errors)
    sjlt_single_errors = np.zeros_like(fd_errors)
    gauss_ihs_errors = np.zeros_like(fd_errors)
    sjlt_ihs_errors = np.zeros_like(fd_errors)

    for it in range(iterations):
        err = np.linalg.norm(all_x[:,it] - x_opt)/np.linalg.norm(x_opt)
        rfd_err = np.linalg.norm(rfd_all_x[:,it] - x_opt)/np.linalg.norm(x_opt)
        gauss_err = np.linalg.norm(gauss_single_all_x[:,it] - x_opt)/np.linalg.norm(x_opt)
        sjlt_err = np.linalg.norm(sjlt_single_all_x[:,it] - x_opt)/np.linalg.norm(x_opt)
        ihs_gauss_err = np.linalg.norm(ihs_gauss_all_x[:,it] - x_opt)/np.linalg.norm(x_opt)
        ihs_sjlt_err = np.linalg.norm(ihs_sjlt_all_x[:,it] - x_opt)/np.linalg.norm(x_opt)
        print(f'Iteration {it}\tFD:{err:.5E}\tRFD:{rfd_err:.5E},\tRP:{gauss_err:.5E},\tIHS:{ihs_gauss_err:.5E}')
        fd_errors[it] = err
        rfd_errors[it] = rfd_err
        gauss_single_errors[it] = gauss_err
        sjlt_single_errors[it] = sjlt_err
        gauss_ihs_errors[it] = ihs_gauss_err
        sjlt_ihs_errors[it] = ihs_sjlt_err

    # ! THIS IS THE AISTATS PLOT SIZE fig, axes = plt.subplots(nrows=2,figsize=(5,2.5),dpi=100)
    fig, axes = plt.subplots(nrows=2,dpi=100)
    ax, ax_time = axes[0], axes[1]

    # ! Error vs Iterations plot
    ax.plot(1+np.arange(iterations), fd_errors,label='FD', **fd_params)
    ax.plot(1+np.arange(iterations), rfd_errors,label='RFD', **rfd_params)
    ax.plot(1+np.arange(iterations), gauss_single_errors,label='Gaussian', **gauss_single_params)
    ax.plot(1+np.arange(iterations), sjlt_single_errors,label='SJLT',**sjlt_single_params)
    ax.plot(1+np.arange(iterations), gauss_ihs_errors,label='ihs:Gauss',**gauss_ihs_params)
    ax.plot(1+np.arange(iterations), sjlt_ihs_errors,label='ihs:SJLT',**sjlt_ihs_params)

    
    if gamma == 100.:
        ax.legend(ncol=2,loc='best')
    ax.set_yscale('log')
    #ax.set_xlabel('Iterations')
    #ax.set_ylabel(r'$\|\mathbf{x}^t - \mathbf{x}^*\|_2 / \| \mathbf{x}^*\|_2$')
    #ax.set_ylabel('Error') # Use this if latex is not present

    # ! Error vs time plot
    ax_time.plot(fd_measured['all_times'], get_errors(all_x,x_opt),label='FD', **fd_params)
    ax_time.plot(rfd_measured['all_times'], get_errors(rfd_all_x,x_opt),label='RFD', **rfd_params)
    ax_time.plot(gauss_single_measured['all_times'], get_errors(gauss_single_all_x,x_opt),label='Gaussian', **gauss_single_params)
    ax_time.plot(sjlt_single_measured['all_times'], get_errors(sjlt_single_all_x,x_opt),label='SJLT', **sjlt_single_params)
    ax_time.plot(ihs_gauss_measured['all_times'], get_errors(ihs_gauss_all_x,x_opt),label='ihs:Gauss', **gauss_ihs_params)
    ax_time.plot(ihs_sjlt_measured['all_times'], get_errors(ihs_sjlt_all_x,x_opt),label='ihs:SJLT', **sjlt_ihs_params)
    ax_time.set_yscale('log',base=10)
    ax_time.set_xscale('log',base=10)
    #ax.set_xscale('log',base=10)
    #ax_time.set_ylim(1E-16, 1E1)

    # ! Saving the plots
    fname = '/home/dickens/code/FrequentDirectionsRidgeRegression/sandbox/figures/efficient-iterative-'+data_name+str(int(gamma))+'.png'
    # ! commenting this line as it is the save format for the paper
    # fig.savefig(fname,dpi=150,bbox_inches='tight',pad_inches=None) 
    fig.savefig(fname,dpi=200,bbox_inches='tight',pad_inches=None)
    #plt.show()
    print(get_errors(all_x,x_opt))
    print(get_errors(rfd_all_x,x_opt))
def main():
    datasets = ['CaliforniaHousing']  #['w8a','CoverType', 'YearPredictions']
    gammas = [1.]#, 100., 1000.]
    for d in datasets:
        for g in gammas:
            synthetic_real_experiment(d,g)

if __name__ == '__main__':
    main()

