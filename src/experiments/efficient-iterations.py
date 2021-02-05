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
import pandas as pd
from utils import get_errors
from timeit import default_timer as timer

# def get_errors(arr,x):
#     e = [np.linalg.norm(arr[:,i] - x)/np.linalg.norm(x) for i in range(arr.shape[1])]
#     e.insert(0,1)
#     return e



def real_experiment(data_name,gamma_reg,):
    gamma = gamma_reg
    n = 20000
    ds = DataFactory(n=n)
    if data_name == 'CoverType':
        X,y = ds.fetch_forest_cover()
        rff_features = True
    elif data_name == 'w8a':
        X,y = ds.fetch_w8a()
        rff_features = True
    elif data_name == 'CaliforniaHousing':
        feature_size = 18000
        _X, _y = fetch_california_housing(return_X_y=True)
        X_train, y = _X, _y
        #X_train, X_test, y, y_test = train_test_split(_X, _y, test_size=0.1, random_state=42)
        rbf_feature = RBFSampler(gamma=0.005, random_state=100,n_components=feature_size)
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
    print('Solving exactly: Data shape: ', X.shape)
    solve_start = timer()
    H = X.T@X + gamma*np.eye(d)
    x_opt = np.linalg.solve(H,X.T@y)
    solve_time = timer() - solve_start
    print('Solving exactly: ', solve_time)
    # Iterate the FD regression
    iterations = 10
    m = int(2**9)
    alpha = 1.0
    fd_sk_dim = int(alpha*m)
    fd_buffer =int((2-alpha)*m)

    print('#'*40)



    print('#'*10, '\t FREQUENT DIRECTIONS \t', '#'*10)
    # fdr = FDRidge(fd_dim=fd_sk_dim,gamma=gamma,batch_size=fd_buffer) # nb this was 2m so expect to incresae slightly
    fdr = FDRidge(fd_dim=fd_sk_dim,gamma=gamma,batch_size=fd_buffer) # nb this was 2m so expect to incresae slightly
    _, all_x,fd_measured = fdr.fast_iterate(X,y,iterations)
    print('#'*10, '\t FREQUENT DIRECTIONS \t ', fd_measured['sketch time'], '#'*10)


    print('#'*10, '\t ROBUST FREQUENT DIRECTIONS \t', '#'*10)
    rfdr = FDRidge(fd_dim=fd_sk_dim,fd_mode='RFD',gamma=gamma,batch_size=fd_buffer)
    _, rfd_all_x, rfd_measured = rfdr.fast_iterate(X,y,iterations)
    print('#'*10, '\t ROBUST FREQUENT DIRECTIONS \t ', rfd_measured['sketch time'], '#'*10)
    

    

    print('#'*10, '\t GAUSS SINGLE \t', '#'*10)
    gauss_single = RPRidge(rp_dim=m,rp_mode='Gaussian',gamma=gamma)
    _, gauss_single_all_x, gauss_single_measured = gauss_single.iterate_single_timing(X,y)

    print('#'*10, '\t SJLT SINGLE \t', '#'*10)
    sjlt_single = RPRidge(rp_dim=m,rp_mode='SJLT',gamma=gamma)
    _, sjlt_single_all_x, sjlt_single_measured = sjlt_single.iterate_single_timing(X,y)

    print('#'*10, '\t GAUSS IHS \t', '#'*10)
    ihs_gauss = RPRidge(rp_dim=m,rp_mode='Gaussian',gamma=gamma)
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

    print('#'*10, '\t PLOTTING \t', '#'*10)
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
    ax_time.plot(fd_measured['all_times'], get_errors(all_x,x_opt), **fd_params)
    ax_time.plot(rfd_measured['all_times'], get_errors(rfd_all_x,x_opt), **rfd_params)
    ax_time.plot(gauss_single_measured['all_times'], get_errors(gauss_single_all_x,x_opt), **gauss_single_params)
    ax_time.plot(sjlt_single_measured['all_times'], get_errors(sjlt_single_all_x,x_opt), **sjlt_single_params)
    ax_time.plot(ihs_gauss_measured['all_times'], get_errors(ihs_gauss_all_x,x_opt), **gauss_ihs_params)
    ax_time.plot(ihs_sjlt_measured['all_times'], get_errors(ihs_sjlt_all_x,x_opt), **sjlt_ihs_params)
    ax_time.set_yscale('log',base=10)
    ax_time.set_xscale('log',base=10)
    ax_time.legend(title=f'Exact:{solve_time:.3f}s')
    #ax.set_xscale('log',base=10)
    #ax_time.set_ylim(1E-16, 1E1)

    # ! Saving the plots
    fname = '/home/dickens/code/FrequentDirectionsRidgeRegression/sandbox/figures/efficient-iterative-'+data_name+str(int(gamma))+'.png'
    # ! commenting this line as it is the save format for the paper
    # fig.savefig(fname,dpi=150,bbox_inches='tight',pad_inches=None) 
    fig.savefig(fname,dpi=200,bbox_inches='tight',pad_inches=None)
    #plt.show()

    # ! Separate the sketch time plots

    # * First build the dataframes
    build_dict = {
        'FD'        : fd_measured['sketch time'],
        'RFD'       : rfd_measured['sketch time'],
        'Gauss'     : gauss_single_measured['sketch time'],
        'SJLT'      : sjlt_single_measured['sketch time'],
        'ihs:Gauss' : ihs_gauss_measured['sketch time'], 
        'ihs:SJLT'  : ihs_sjlt_measured['sketch time']
    }
    mean_iter_time_single = lambda a : np.mean(a['all_times'][1:] - a['sketch time'])
    mean_iter_time_multi = lambda a : np.mean(a['all_times'][1:] - a['sketch time']/iterations)
    iteration_dict = {
        'FD'        :  mean_iter_time_single(fd_measured),
        'RFD'       :  mean_iter_time_single(rfd_measured), 
        'Gauss'     :  mean_iter_time_single(gauss_single_measured), 
        'SJLT'      :  mean_iter_time_single(sjlt_single_measured),
        'ihs:Gauss' :  mean_iter_time_multi(ihs_gauss_measured), 
        'ihs:SJLT'  :  mean_iter_time_multi(ihs_sjlt_measured) 
    }
    print(build_dict)
    print(iteration_dict)
    # * Do the plotting
    bar_cols = [x['color'] for x in [fd_params, rfd_params, gauss_single_params, sjlt_single_params, gauss_ihs_params, sjlt_ihs_params]] 
    timing_fig, timing_axes = plt.subplots(ncols=2,dpi=150)
    timing_fname  = '/home/dickens/code/FrequentDirectionsRidgeRegression/sandbox/figures/efficient-iterative-'+data_name+str(int(gamma))+'separate-time-cost.png'
    build_ax, iter_ax = timing_axes

    # 
    build_ax.barh(list(build_dict.keys()), list(build_dict.values()),color=bar_cols)
    iter_ax.barh(list(iteration_dict.keys()), list(iteration_dict.values()),color=bar_cols)
    build_ax.set_xlabel('Build Time (seconds)')
    iter_ax.set_xlabel('Iteration Time (seconds)')
    iter_ax.set_xlim(0,iteration_dict['ihs:Gauss']+1)
    #iter_ax.set_xscale('symlog')
    _ihs_sjlt_time = iteration_dict['ihs:SJLT']
    _title = f'ihs:sjlt:{_ihs_sjlt_time:.3f}'
    iter_ax.legend(title=_title)
    timing_fig.savefig(timing_fname,dpi=200,pad_inches=None)




def main():
    datasets = ['CaliforniaHousing']#,'CoverType', 'YearPredictions']['w8a']: #
    gammas = [100.]
    for d in datasets:
        for g in gammas:
            real_experiment(d,g)

if __name__ == '__main__':
    main()

