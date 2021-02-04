"""
Example code by Jason demonstrating some pygama convenience functions.
"""
import numpy as np
import matplotlib.pyplot as plt
import pygama.analysis.histograms as pgh
import pygama.analysis.peak_fitting as pga

np.random.seed(0) # fix the seed s/t we can reproduce the plot

n = 10
data = np.random.normal(0, 1, n)

hist, bins, var = pgh.get_hist(data, range=(-5,5), dx=1)
pgh.plot_hist(hist, bins, var, label="data")

pars, cov = pga.fit_hist(pga.gauss, hist, bins, var=var, guess=[0,1,n])
pgh.print_fit_results(pars, cov, ['mu', 'sig', 'A'])
pgh.plot_func(pga.gauss, pars, label="chi2 fit")

nbnd = (-np.inf, np.inf)
pos = (0, np.inf)
pars, cov = pga.fit_hist(pga.gauss, hist, bins, var=var,
                         guess=[0,1,n], bounds=[nbnd,pos,pos], poissonLL=True)
pgh.print_fit_results(pars, cov, ['mu', 'sig', 'A'])
pgh.plot_func(pga.gauss, pars, label="poissonLL fit")

plt.legend(loc=1)

plt.show()