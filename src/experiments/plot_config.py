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

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

# Uncomment when ready to include latex as it takes longer to plot
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": ["Times"],
    "font.size" : 12,
    "text.latex.preamble" : r"\usepackage{amsmath}"
})


fd_params = {
    'color'     : 'black',
    'linewidth' : 1.5,
    'linestyle' : '-',
    'marker'    : '*'
}

rfd_params = {
    'color'     : 'gray',
    'linewidth' : 1.5,
    'linestyle' : '-.',
    'marker'    : '1',
    'markersize': 17.5
}

sjlt_rp_params = {
    'color'     : 'red',
    'linewidth' : 1.5,
    'linestyle' : '--',
    'marker'    : 'x',
    'markersize': 5.0
}

gauss_rp_params = {
    'color'     : 'red',
    'linewidth' : 1.5,
    'linestyle' : '--',
    'marker'    : 's',
    'markersize': 5.0
}

gauss_hs_params = {
    'color'     : 'blue',
    'linewidth' : 1.5,
    'linestyle' : ':',
    'marker'    : '^',
    'markersize': 5.0
}

sjlt_hs_params = {
    'color'     : 'blue',
    'linewidth' : 1.5,
    'linestyle' : ':',
    'marker'    : '+',
    'markersize': 5.0
}

bound_params = {
    'color' : 'red',
    'linewidth' : 1.5,
    'linestyle' : ':'
}

# Extra config for iterative exps

gauss_single_params = gauss_rp_params
sjlt_single_params = sjlt_rp_params

gauss_ihs_params = {
    'color'     : 'magenta',
    'linewidth' : 1.5,
    'linestyle' : ':',
    'marker'    : 'D',
    'markersize': 5.0
}

sjlt_ihs_params = {
    'color'     : 'cyan',
    'linewidth' : 1.5,
    'linestyle' : ':',
    'marker'    : 'o',
    'markersize': 5.0
}
