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
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.kernel_approximation import RBFSampler
from plot_config import fd_params, rfd_params, sjlt_rp_params, gauss_rp_params, gauss_hs_params, sjlt_hs_params 
import numpy as np
from math import floor
import json
import matplotlib.pyplot as plt
from utils import get_errors, get_x_opt

def error_vs_sketch_size(gamma_reg=10.):
    # Experimental parameters
    gamma = gamma_reg
    iterations = 5
    n = 2**15 # 10000 #
    d = 2**10 # 4000 #
    eff_rank = int(floor(0.25*d + 0.5))
    ds = DataFactory(n=n,d=d,effective_rank=eff_rank,tail_strength=0.125,random_seed=100)
    X,y,w0 = ds.shi_phillips_synthetic()
    # _X, _y = fetch_california_housing(return_X_y=True)
    # X_train, X_test, y, y_test = train_test_split(_X[:n], _y[:n], test_size=0.4, random_state=42)
    # rbf_feature = RBFSampler(gamma=0.001, random_state=100,n_components=d)
    # X = rbf_feature.fit_transform(X_train)

    X = StandardScaler().fit_transform(X)
    X /= np.linalg.norm(X,ord='fro')**2 
    x_opt = get_x_opt(X,y,gamma)

    sketch_sizes = [400] 
    fd_errors = {m : np.zeros(iterations+1) for m in sketch_sizes}
    fd_times = {m : np.zeros(iterations+1) for m in sketch_sizes}
    fd_sketch_times = np.zeros(len(sketch_sizes), dtype=float)
    
    rfd_errors = {m : np.zeros(iterations+1) for m in sketch_sizes}
    rfd_times = {m : np.zeros(iterations+1) for m in sketch_sizes}
    rfd_sketch_times = np.zeros_like(fd_sketch_times)



    for i,m in enumerate(sketch_sizes):
        print('#'*40)
        print('#'*10, f'\t FREQUENT DIRECTIONS: m={m}\t', '#'*10)
        fdr = FDRidge(fd_dim=m,gamma=gamma)
        _, all_x,fd_measured = fdr.fast_iterate(X,y,iterations)
        fd_errors[m] = get_errors(all_x,x_opt)
        fd_sketch_times[i] = fd_measured['sketch time']
        fd_times[m] = fd_measured['all_times']

        print('#'*10, f'\t robust FREQUENT DIRECTIONS: m={m}\t', '#'*10)
        rfdr = FDRidge(fd_dim=m,fd_mode='RFD',gamma=gamma)
        _, rfd_all_x,rfd_measured = rfdr.fast_iterate(X,y,iterations)
        rfd_errors[m] = get_errors(rfd_all_x,x_opt)
        rfd_sketch_times[i] = fd_measured['sketch time']
        rfd_times[m] = rfd_measured['all_times']
    
        # print(np.c_[fd_measured['all_times'], rfd_measured['all_times']])
        # print(np.c_[fd_sketch_times, rfd_sketch_times])
        
    # ! Plotting the error profile
    fig, ax = plt.subplots()
    tfig, tax = plt.subplots()
    fd_color = 'black'
    rfd_color = 'gray'
    markers = ['.', '*', '^', '+']
    lines = ['-', ':', '-.']
    cycle_count = 0
    my_err = np.zeros((iterations+1,2))
    # Plot the FD curves
    for sk_size, err in fd_errors.items():
        fd_err = err
        rfd_err = rfd_errors[sk_size]
        _m = markers[cycle_count]
        # _l = lines[cycle_count]
        cycle_count += 1
        all_fd_time = fd_times[sk_size]
        all_rfd_time = rfd_times[sk_size]
        ax.plot(range(iterations+1), fd_err, marker=_m, linestyle=':', color=fd_color, label=f'FD:{sk_size}')
        ax.plot(range(iterations+1), rfd_err, marker=_m, linestyle='-', color=rfd_color, label=f'RFD:{sk_size}')
        tax.plot(all_fd_time, fd_err, marker=_m, linestyle=':', color=fd_color, label=sk_size)
        tax.plot(all_rfd_time, rfd_err, marker=_m, linestyle='-', color=rfd_color, label=sk_size)
    # tax.plot(sketch_sizes, fd_sketch_times, marker=_m, linestyle=':', color=fd_color, label='FD')
    # tax.plot(sketch_sizes, rfd_sketch_times, marker=_m, linestyle='-', color=rfd_color, label='RFD')

        

    # * format the axes for ITERATIONS
    ax.set_yscale('log',base=10)
    ax.legend()
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Error')
    path = '/home/dickens/code/FrequentDirectionsRidgeRegression/sandbox/figures/'
    fname = path + '/error-profile-sketch-size-iterations-' + str(int(gamma)) + '.png'
    # ! commenting this line as it is the save format for the paper
    fig.savefig(fname,dpi=200,bbox_inches='tight',pad_inches=None)

    # * format the axes for TIME
    tax.set_xlim(left=4)
    tax.set_ylim(bottom=1E-12,top=1.0)
    tax.set_yscale('log',base=10)
    #tax.set_xscale('log',base=2)
    tax.legend()
    #tax.grid()
    tax.set_xlabel('Time')
    tax.set_ylabel('Error')
    tfname = path + '/error-profile-sketch-size-time-' + str(int(gamma)) + '.png'
    # ! commenting this line as it is the save format for the paper
    tfig.savefig(tfname,dpi=200,bbox_inches='tight',pad_inches=None)


    # ! Prepare and save the results in json format 
    res_name = 'results/error-profile-sketch-size-' + str(int(gamma)) + '.json'

    for d in [fd_errors, rfd_errors, fd_times, rfd_times]:
        for k,v in d.items():
            if type(v) == np.ndarray:
                d[k] = v.tolist()

    results = {
        'iterations' : iterations,
        'FD Error'   : fd_errors,
        'RFD Error'  : rfd_errors,
        'FD Times'   : fd_times,
        'RFD Times'  : rfd_times,
        'sketch sizes' : sketch_sizes
    }
    with open(res_name, 'w') as fp:
        json.dump(results, fp,sort_keys=True, indent=4)
   
    




def main():
    for g in [10., 100., 1000.]:
        error_vs_sketch_size(g) 

if __name__ =='__main__':
    main()