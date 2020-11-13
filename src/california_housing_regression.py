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
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from ridge_regression import RidgeRegression
from frequent_directions_ridge_regression import FDRR
from random_projections_ridge_regression import RPRR
from sklearn.model_selection import train_test_split

def main():
    fpath = '../notebooks/CaliforniaHousing/datasets/'
    train = np.load(fpath+'train.npy')
    valid = np.load(fpath+'validate.npy')
    test  = np.load(fpath+'test.npy') 

    #X_tr   , y_tr    = transformed[:,:-1],transformed[:,-1]
    X_train, y_train = train[:,:-1], train[:,-1]
    X_valid, y_valid = valid[:,:-1], valid[:,-1]
    X_test,  y_test   = test[:,:-1], test[:,-1]

    y_mean = np.mean(np.concatenate((y_train, y_valid, y_test),axis=0))
    for yy in [y_train, y_valid, y_test]:
        yy -= y_mean
        
    X_train_poly    = PolynomialFeatures(degree=3).fit_transform(X_train)
    X_valid_poly   = PolynomialFeatures(degree=3).fit_transform(X_valid)
    X_test_poly   = PolynomialFeatures(degree=3).fit_transform(X_test)

    print(f"Training sizes: {X_train_poly.shape, y_train.shape}")
    print(f'Validation size: {X_valid_poly.shape, y_valid.shape}')
    print(f'Testing size: {X_test_poly.shape, y_test.shape}')


    # Model hyperparameters
    gammas = [10**_ for _ in np.arange(-5,8,step=0.25)]
    sketch_dimension = 256
    

    # Experimental setup
    num_trials = 5
    all_train_data = np.concatenate((X_train_poly, X_valid_poly),axis=0)
    all_train_labels = np.concatenate((y_train, y_valid),axis=0)

    # Output results
    exact_results = {
        'train_error' : np.zeros((len(gammas),num_trials)),
        'valid_error' : np.zeros((len(gammas),num_trials)),
        'test_error'  : np.zeros((len(gammas),num_trials))
    }

    rfd_results = {
    'train_error' : np.zeros((len(gammas),num_trials)),
    'valid_error' : np.zeros((len(gammas),num_trials)),
    'test_error'  : np.zeros((len(gammas),num_trials))
    }

    cl_results = {
    'train_error' : np.zeros((len(gammas),num_trials)),
    'valid_error' : np.zeros((len(gammas),num_trials)),
    'test_error'  : np.zeros((len(gammas),num_trials))
    }

    hs_results = {
        'train_error' : np.zeros((len(gammas),num_trials)),
        'valid_error' : np.zeros((len(gammas),num_trials)),
        'test_error'  : np.zeros((len(gammas),num_trials))
    }


    for exp in range(num_trials):
        print('Experiment: ', exp)
        # Generate new train-validation split
        _X_train, _X_valid, _y_train, _y_valid = train_test_split(all_train_data, all_train_labels, test_size=0.2,random_state=10*exp)
        
        ############ EXACT MODEL ############
        print('...exact model')
        exact_ridge = RidgeRegression(gammas)
        exact_ridge.fit(_X_train, _y_train)
        exact_results['train_error'][:,exp]  = exact_ridge.get_errors(_X_train, _y_train)
        exact_results['valid_error'][:,exp]  = exact_ridge.get_errors(_X_valid, _y_valid)
        exact_results['test_error'][:,exp]   = exact_ridge.get_errors(X_test_poly, y_test)

        ############ RFD MODEL ############
        print('...RFD model')
        rfd_ridge = FDRR(fd_dim=sketch_dimension,gamma=gammas,fd_mode='RFD',solve_method='ShiWoodbury')
        rfd_ridge.fit(_X_train, _y_train)
        rfd_results['train_error'][:,exp]  = rfd_ridge.get_errors(_X_train, _y_train)
        rfd_results['valid_error'][:,exp]  = rfd_ridge.get_errors(_X_valid, _y_valid)
        rfd_results['test_error'][:,exp]   = rfd_ridge.get_errors(X_test_poly, y_test)

        ############ Classical Sketch MODEL ############
        print('...(CL)assical model')
        cl_ridge = RPRR(rp_dim=sketch_dimension,gamma=gammas,rp_method='Classical',rp_mode='SJLT',solve_method='ShiWoodbury')
        cl_ridge.fit(_X_train, _y_train)
        cl_results['train_error'][:,exp]  = cl_ridge.get_errors(_X_train, _y_train)
        cl_results['valid_error'][:,exp]  = cl_ridge.get_errors(_X_valid, _y_valid)
        cl_results['test_error'][:,exp]   = cl_ridge.get_errors(X_test_poly, y_test)
        
        ############ Hessian Sketch MODEL ############
        print('...(H)essian (S)ketch model')
        hs_ridge = RPRR(rp_dim=sketch_dimension,gamma=gammas,rp_method='Hessian',rp_mode='SJLT',solve_method='ShiWoodbury')
        hs_ridge.fit(_X_train, _y_train)
        hs_results['train_error'][:,exp]  = hs_ridge.get_errors(_X_train, _y_train)
        hs_results['valid_error'][:,exp]  = hs_ridge.get_errors(_X_valid, _y_valid)
        hs_results['test_error'][:,exp]   = hs_ridge.get_errors(X_test_poly, y_test)


    for dictionary in [exact_results,rfd_results, cl_results, hs_results]:
        dictionary = post_process_experiment_dict(dictionary)

        # Get the optimal gamma and associated test error at that gamma
        dictionary['best_gamma'] = gammas[dictionary['mean_valid_error'].argmin()]
        dictionary['best_test_error'] = dictionary['mean_test_error'][dictionary['mean_valid_error'].argmin()]


    ############ PLOTTING THE RESULTS ############ 
    make_plots(gammas, exact_results, rfd_results, cl_results, hs_results)

def make_plots(gammas, exact_results, rfd_results, cl_results, hs_results):
    plt.rcParams.update({
    #"text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": ["Times"],
    "font.size" : 10,
    #"text.latex.preamble" : r"\usepackage{amsmath}"
    })
    fig, axes = plt.subplots(nrows=3,dpi=100,gridspec_kw = {'hspace':0},constrained_layout=True)#,figsize=[6.4, 7])
    ax_tr, ax_va, ax_te = axes
    ALPHA = 0.25 # FOR SHADING

    labels = ['Exact', 'RFD', 'RP:C:S', 'RP:H:S']
    dicts = [exact_results, rfd_results, cl_results,hs_results]
    #plotting_dicts = [exact_params, rfd_params, sjlt_rp_params, sjlt_hs_params]
    ############ TRAINING PLOT ############
    for l,m in zip(labels,dicts):
        _mean = m['mean_train_error']
        _med  = m['median_train_error']
        _std  = m['std_train_error']
        if l == 'Exact':
            _marker = '*'
        else:
            _marker = None
        #ax_tr.plot(gammas, _mean, label=l)
        ax_tr.plot(gammas, _med, label=l, marker=_marker)
        
        # TRAIN Fill error region
        #ax_tr.fill_between(gammas,_mean - _std, _mean+_std,alpha=ALPHA)
    ax_tr.set_ylim(0,0.025)


    # # ############ VALIDATION PLOT ############
    for l,m in zip(labels,dicts):
        _mean = m['mean_valid_error']
        _med  = m['median_valid_error']
        _std   = m['std_valid_error']
        #ax_va.plot(gammas, _mean, label=l)
        ax_va.plot(gammas, _med, label=l)
        
        # VALID Fill error region
        #ax_va.fill_between(gammas,_mean - _std, _mean+_std,alpha=ALPHA)
    ax_va.set_ylim(0,0.1)



    # # ############ TESTING PLOT ############
    for l,m in zip(labels,dicts):
        _mean = m['mean_test_error']
        _med  = m['median_test_error']
        _std   = m['std_test_error']
        #ax_te.plot(gammas, _mean, label=l)
        ax_te.plot(gammas, _med, label=l)
        
        # TEST Fill error region
        #ax_te.fill_between(gammas,_mean - _std, _mean+_std,alpha=ALPHA)
    ax_te.set_ylim(0,0.1)
    ax_te.set_xlabel(r'$\gamma$')

    ax_tr.set_ylabel('Train Error')
    ax_va.set_ylabel('Valid Error')
    ax_te.set_ylabel('Test Error')

    for ax in axes:
        ax.set_xscale('log',basex=10)
        ax.axvline(exact_results['best_gamma'],label=r'$\gamma_{{exact}}$:{:.2f}'.format(exact_results['best_gamma']), linestyle=':',marker='*',color='C0')
        ax.axvline(rfd_results['best_gamma'],label=r'$\gamma_{{RFD}}$:{:.2f}'.format(rfd_results['best_gamma']), linestyle=':',color='C1')
        ax.axvline(cl_results['best_gamma'],label=r'$\gamma_{{CL}}$:{:.2f}'.format(cl_results['best_gamma']), linestyle=':',color='C2')
        ax.axvline(hs_results['best_gamma'],label=r'$\gamma_{{HS}}$:{:.2f}'.format(hs_results['best_gamma']), linestyle=':',color='C3')
        ax.grid()
        ax.set_xlim(1E-3,1E6)

    # Remove x ticks from top two plots
    for ax in [ax_tr, ax_va]:
        ax.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False) # labels along the bottom edge are off
        
    # Get the test errors:
    ax_tr.plot([],[],color='white',label='ExactTest:{:.5f}'.format(exact_results['best_test_error']))
    ax_tr.plot([],[],color='white',label='FDTest:{:.5f}'.format(rfd_results['best_test_error']))
    ax_tr.plot([],[],color='white',label='CLTest:{:.5f}'.format(cl_results['best_test_error']))
    ax_tr.plot([],[],color='white',label='HSTest:{:.5f}'.format(hs_results['best_test_error']))

    # Adjust spacing for legend
    ax_tr.legend(loc='lower center', frameon=False, bbox_to_anchor=(0.5, 1.05), ncol=4,)
    #lgd = fig.legend(ncol=4, loc='lower center', prop={'size': 10}, bbox_to_anchor=(0.5, 1.05))
    #ax_te.legend(loc='center left', bbox_to_anchor=(1, 1.5))
    #plt.tight_layout()
    plt.show()  
    fig.savefig('experiments/figures/california_housing.eps')
    fig.savefig('experiments/figures/california_housing.pdf')

def post_process_experiment_dict(dictionary):
    """
    Input: dictionary with experimental results from the ridge regression.
    Output: Mean and std of the experimental results as (key, value pairs in my_dict.
    """
    temp_mean = {
        'median_train_error' : None,
        'mean_train_error' : None,
        'mean_valid_error' : None,
        'mean_test_error'  : None,
    }
    
    temp_std = {
        'std_train_error' : None,
        'std_valid_error' : None,
        'std_test_error'  : None,
    }
    for k,v in dictionary.items():
        v_med  = np.median(v,axis=1)
        v_mean = np.mean(v,axis=1)
        v_std  = np.std(v,axis=1)
        
        if k == 'train_error':
            temp_mean['mean_train_error']   = v_mean
            temp_mean['median_train_error'] = v_med
            temp_std['std_train_error']     = v_std
        elif k == 'valid_error':
            temp_mean['mean_valid_error'] = v_mean
            temp_mean['median_valid_error'] = v_med
            temp_std['std_valid_error']   = v_std
        elif k == 'test_error':
            temp_mean['mean_test_error'] = v_mean
            temp_mean['median_test_error'] = v_med
            temp_std['std_test_error']   = v_std
        else:
            raise Exception('Key not found.')

    for k,v in temp_mean.items():
        dictionary[k] = v
    for k,v in temp_std.items():
        dictionary[k] = v
    return dictionary


if __name__ == '__main__':
    main()