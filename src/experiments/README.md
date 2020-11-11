<!--
    Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.
-->

# README: Experiments

This subdirectory contains the experimental scripts and classes that are necessary to perform the 
experiments from (this paper)[https://arxiv.org/abs/2011.03607].
The experiments are:
1. `bias_variance_tradeoff.py` produces Figure 1 from the paper.  This demonstrates how the bias and variance of the returned weights from sketched ridge regression varies depending on the deployed sketch.
2. `iterative_sketching.py` produces Figure 2 from the paper. This demonstrates how various sketching techniques compare when used for obtaining high-accuracy solutions to the ridge regression problem.