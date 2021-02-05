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

# README: Datasets 

We have used various open datasets for experimentation.  These are generally too large to be stored in 
the repo but they can easily be obtained by downloading the data and the running `convert_to_numpy.py`.
For every dataset `d` in the dictionary `datasets` there is a `url` key which is the url from which the data is pulled.
Then execute `wget url - O path` where `path` is again a key in the dictionary.
When the three datasets are downloaded one should execute `convert_to_numpy.py` which will put them in 
a consistent`[X,y]` format for later usage.
Additionally, the arrays will be saved in the `.npy` format.
- 