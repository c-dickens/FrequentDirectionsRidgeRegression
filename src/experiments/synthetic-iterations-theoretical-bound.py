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

def error_vs_sketch_size(n,d,sketch_size,r):
    # * Experimental parameters
    trials = 10
    iterations = 5
    eff_rank = int(floor(r*d + 0.5))
    g_range = [-1, 0, 1, 2]
    
    # * Results data structures
    fd_errors = {g : np.zeros((iterations+1, trials),dtype=float) for g in g_range}
    rfd_errors = {g : np.zeros((iterations+1, trials),dtype=float) for g in g_range}
    for t in range(trials):
        ds = DataFactory(n=n,d=d,effective_rank=eff_rank,tail_strength=0.125,random_seed=t)
        X,y,w0 = ds.shi_phillips_synthetic()
        X = StandardScaler().fit_transform(X)
        
        gammas = [(2**a)*(np.linalg.norm(X,ord='fro')**2)/sketch_size for a in g_range]
        for i, g in enumerate(gammas):
            print('#'*10, f'\t GAMMA: G={g}, i={i} a={g_range[i]}\t', '#'*10)
            a = g_range[i]
            #X /= np.linalg.norm(X,ord='fro')**2 

            # # ! Optimal solution
            x_opt = get_x_opt(X,y,g)

            # # ! FD Sketching 
            fdr = FDRidge(fd_dim=sketch_size,gamma=g)
            _, all_x,fd_measured = fdr.fast_iterate(X,y,iterations)
            fd_errors[a][:,t] =  get_errors(all_x,x_opt)


            # ! RFD Sketching 
            rfdr = FDRidge(fd_dim=sketch_size,fd_mode='RFD',gamma=g)
            _, rfd_all_x,rfd_measured = rfdr.fast_iterate(X,y,iterations)
            rfd_errors[a][:,t] = get_errors(rfd_all_x,x_opt)

    # ! Results setup  and plotting
    fig, ax = plt.subplots()
    for i,g in enumerate(gammas):
        a = g_range[i]

        # * FD
        fd_errs = fd_errors[a]
        fd_mean_errs = np.mean(fd_errs,axis=1)
        fd_std_errs = np.std(fd_errs,axis=1)
        print(fd_std_errs)
        ax.errorbar(range(iterations+1),fd_mean_errs,yerr=fd_std_errs, label=f'g=2^{a}*(1/m)')

        # * RFD
        rfd_errs = rfd_errors[a]
        rfd_mean_errs = np.mean(rfd_errs,axis=1)
        rfd_std_errs = np.std(rfd_errs,axis=1)
        print(rfd_std_errs)
        ax.errorbar(range(iterations+1),rfd_mean_errs,yerr=rfd_std_errs, label=f'Rg=2^{a}*(1/m)')
    ax.legend()
    ax.set_yscale('log',base=10)
    ax.legend()
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Error')
    path = '/home/dickens/code/FrequentDirectionsRidgeRegression/sandbox/figures/'
    fname = path + '/bound-test-' + str(r) + '.png'
    fig.savefig(fname,dpi=200,bbox_inches='tight',pad_inches=None)

    # ! Prepare and save the results in json format 
    res_name = 'results/bound-test' + str(r) + '.json'
    for d in [fd_errors,rfd_errors]:# fd_means, fd_stds]:#, rfd_errors, fd_times, rfd_times]:
        for k,v in d.items():
            if type(v) == np.ndarray:
                d[k] = v.tolist()
    results = {
        'FD Error'   : fd_errors,
        'RFD Error'   : rfd_errors,
        # 'FD Means'   : fd_means, 
        # 'FD Stds'    : fd_stds
        }
    with open(res_name, 'w') as fp:
        json.dump(results, fp,sort_keys=True, indent=4)


   


def main():
    n = 2**15
    d = 2**12
    for (m,r) in [(300,0.25),(300,0.3),(300,0.4)]: 
        error_vs_sketch_size(n,d,m,r) 

if __name__ =='__main__':
    main()