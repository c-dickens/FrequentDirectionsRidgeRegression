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
import matplotlib.pyplot as plt
from utils import get_errors, get_x_opt

def error_vs_sketch_size():
    # Experimental parameters
    gamma = 100.
    iterations = 5
    n = 2**12
    d = 2**9
    eff_rank = int(floor(0.25*d + 0.5))
    # ds = DataFactory(n=n,d=d,effective_rank=eff_rank,tail_strength=0.125,random_seed=100)
    # X,y,w0 = ds.shi_phillips_synthetic()
    _X, _y = fetch_california_housing(return_X_y=True)
    X_train, X_test, y, y_test = train_test_split(_X[:n], _y[:n], test_size=0.4, random_state=42)
    rbf_feature = RBFSampler(gamma=0.001, random_state=100,n_components=d)
    X = rbf_feature.fit_transform(X_train)

    X = StandardScaler().fit_transform(X)
    x_opt = get_x_opt(X,y,gamma)

    sketch_sizes = [128 + _*64 for _ in range(3)]
    print(sketch_sizes)
    fd_errors = {m : np.zeros(iterations+1) for m in sketch_sizes}
    rfd_errors = {m : np.zeros(iterations+1) for m in sketch_sizes}

    for m in sketch_sizes:
        print('#'*40)
        print('#'*10, f'\t FREQUENT DIRECTIONS: m={m}\t', '#'*10)
        fdr = FDRidge(fd_dim=m,gamma=gamma)
        _, all_x,fd_measured = fdr.fast_iterate(X,y,iterations)
        fd_errors[m] = get_errors(all_x,x_opt)
        print(fd_errors[m])


        print('#'*10, f'\t robust FREQUENT DIRECTIONS: m={m}\t', '#'*10)
        rfdr = FDRidge(fd_dim=m,fd_mode='RFD',gamma=gamma)
        _, rfd_all_x,rfd_measured = rfdr.fast_iterate(X,y,iterations)
        rfd_errors[m] = get_errors(rfd_all_x,x_opt)
        print(rfd_errors[m])

        # print(all_x)
        # print(rfd_all_x)
    

    # ! Plotting the error profile
    fig, ax = plt.subplots()
    fd_color = 'black'
    rfd_color = 'gray'
    markers = ['.', '*', '^']
    lines = ['-', ':', '-.']
    cycle_count = 0
    my_err = np.zeros((iterations+1,2))
    # Plot the FD curves
    for sk_size, err in fd_errors.items():
        #fd_err = err
        rfd_err = rfd_errors[sk_size]
        # print(np.c_[fd_err, rfd_err])

        _m = markers[cycle_count]
        _l = lines[cycle_count]
        cycle_count += 1
        #ax.plot(range(iterations+1), err, marker=_m, linestyle=_l, color=fd_color, label=sk_size)
        ax.plot(range(iterations+1), rfd_err, marker=_m, linestyle=_l, color=rfd_color, label=sk_size)

    # Plot the RFD curves
    # cycle_count = 0
    # for sk_size, rerr in rfd_errors.items():
    #     my_err[:,1] = rerr
    #     _m = markers[cycle_count]
    #     _l = lines[cycle_count]
    #     cycle_count += 1
    #     ax.plot(range(iterations+1), rerr, marker=_m, linestyle=_l, color=rfd_color, label=sk_size)
        
    #     print(my_err)
    ax.set_yscale('log',base=10)
    ax.legend()
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Error')
    fname = '/home/dickens/code/FrequentDirectionsRidgeRegression/sandbox/figures/error-profile-sketch-size.png'
    # ! commenting this line as it is the save format for the paper
    fig.savefig(fname,dpi=200,bbox_inches='tight',pad_inches=None)


def main():
    error_vs_sketch_size()
    pass 

if __name__ =='__main__':
    main()