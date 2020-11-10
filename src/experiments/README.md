# README: Experiments

This subdirectory contains the experimental scripts and classes that are necessary to perform the 
experiments from (this paper)[https://arxiv.org/abs/2011.03607].
The experiments are:
1. `bias_variance_tradeoff.py` produces Figure 1 from the paper.  This demonstrates how the bias and variance of the returned weights from sketched ridge regression varies depending on the deployed sketch.
2. `iterative_sketching.py` produces Figure 2 from the paper. This demonstrates how various sketching techniques compare when used for obtaining high-accuracy solutions to the ridge regression problem.