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
