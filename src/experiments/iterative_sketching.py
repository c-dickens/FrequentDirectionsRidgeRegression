from data_factory import DataFactory
from frequent_directions_aistats_ridge_regression import FDRidge
from random_projections_aistats_ridge_regression import RPRidge
from sklearn.linear_model import Ridge,LinearRegression
from plot_config import fd_params, rfd_params, gauss_single_params, sjlt_single_params, gauss_ihs_params, sjlt_ihs_params
import numpy as np
from math import floor
import matplotlib.pyplot as plt


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
    else:
        X,y = ds.fetch_year_predictions()
        rff_features = True
    X, y = X[:n], y[:n]

    # Whether to fit fourier features
    if rff_features:
        X = ds.feature_expansion(X,n_extra_features=1024)
    d = X.shape[1]


    # Optimal solution
    H = X.T@X + gamma*np.eye(d)
    x_opt = np.linalg.solve(H,X.T@y)

    # Iterate the FD regression
    iterations = 10
    m = int(2**8)
    fdr = FDRidge(fd_dim=m,gamma=gamma)
    fdr.fit(X,y)
    _, all_x = fdr.iterate(X,y,iterations)

    rfdr = FDRidge(fd_dim=m,fd_mode='RFD',gamma=gamma)
    rfdr.fit(X,y)
    _, rfd_all_x = rfdr.iterate(X,y,iterations)

    gauss_single = RPRidge(rp_dim=m,rp_mode='Gaussian',gamma=gamma)
    _, gauss_single_all_x = gauss_single.iterate_single(X,y)

    sjlt_single = RPRidge(rp_dim=m,rp_mode='SJLT',gamma=gamma)
    _, sjlt_single_all_x = sjlt_single.iterate_single(X,y)



    ihs_gauss = RPRidge(rp_dim=m,rp_mode='Gaussian',gamma=gamma)
    _, ihs_gauss_all_x = ihs_gauss.iterate_multiple(X,y)

    ihs_sjlt = RPRidge(rp_dim=m,rp_mode='SJLT',gamma=gamma)
    _, ihs_sjlt_all_x = ihs_sjlt.iterate_multiple(X,y)

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

        print(f'Iteration {it}\tFD:{err:.5E}\tRFD:{rfd_err:.5E}\tRP:{gauss_err:.5E}')#\tIHS:{ihs_err:.5E}')
        fd_errors[it] = err
        rfd_errors[it] = rfd_err
        gauss_single_errors[it] = gauss_err
        sjlt_single_errors[it] = sjlt_err
        gauss_ihs_errors[it] = ihs_gauss_err
        sjlt_ihs_errors[it] = ihs_sjlt_err

    fig, ax = plt.subplots(figsize=(5,2.5))
    ax.plot(1+np.arange(iterations), fd_errors,label='FD', **fd_params)
    ax.plot(1+np.arange(iterations), rfd_errors,label='RFD', **rfd_params)
    ax.plot(1+np.arange(iterations), gauss_single_errors,label='Gaussian', **gauss_single_params)
    ax.plot(1+np.arange(iterations), sjlt_single_errors,label='SJLT',**sjlt_single_params)
    ax.plot(1+np.arange(iterations), gauss_ihs_errors,label='ihs:Gauss',**gauss_ihs_params)
    ax.plot(1+np.arange(iterations), sjlt_ihs_errors,label='ihs:SJLT',**sjlt_ihs_params)
    
    if gamma == 100.:
        ax.legend(ncol=2,loc='best')
    ax.set_yscale('log')
    ax.set_xlabel('Iterations')
    ax.set_ylabel(r'$\|\mathbf{x}^t - \mathbf{x}^*\|_2 / \| \mathbf{x}^*\|_2$')
    fname = 'figures/iterative-'+data_name+str(gamma)+'.eps'
    fig.savefig(fname,dpi=150,bbox_inches='tight',pad_inches=None)
    #plt.show()
def main():
    datasets = ['w8a','CoverType', 'YearPredictions']
    gammas = [10., 100., 1000.]
    for d in datasets:
        for g in gammas:
            synthetic_real_experiment(d,g)

if __name__ == '__main__':
    main()

