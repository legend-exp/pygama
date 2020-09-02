## kinda preliminary version which summarises the results in a pdf in ./output/low_cut_results.pdf
## doesn't overwrite if the folder already exists, instead makes a new folder

import pygama
import os, time, json
import numpy as np
import pandas as pd
from pprint import pprint
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
from scipy.integrate import quad
import tinydb as db
import argparse
from statsmodels.stats import proportion
import json
from scipy.optimize import curve_fit
import math
from scipy.special import erf
from scipy.signal import savgol_filter
import os

from pygama import DataSet
from pygama.analysis.calibration import *
from pygama.analysis.histograms import *
import pygama.utils as pgu
from matplotlib.lines import Line2D
from pygama.utils import set_plot_style
import h5py
from matplotlib.backends.backend_pdf import PdfPages


def main():

	## took that from original script because why not
	par = argparse.ArgumentParser(description="A/E cut for HADES")
	arg, st, sf = par.add_argument, "store_true", "store_false"
	#arg("-ds", nargs='*', action="store", help="load runs for a DS")
	#arg("-r", default=1, help="select a run")
	#arg("-db", "--writeDB", action=st, default=False, help="store results in DB")
	arg("-v",  default=0, help="verbosity level, 0: Just do it, 1: Tell me when you start a new step, 2: Tell me everything")
	arg("-o", action=st, default=False,  help="stores every plot generated in ./output")
	arg("-md",  help="select runDB to process entire dataset")
	#arg("-c", help="calibration json file")



	args = vars(par.parse_args())
	print(args)

    # -- declare the DataSet --
	if args["md"]:
		#find_cut_using_md(args["md"], int(args["r"]), int(args["v"]), args["writeDB"], args["o"])
		#find_cut_using_md(args["md"],  int(args["v"]), args["writeDB"], args["o"], args["c"])
		find_cut_using_md(args["md"],  int(args["v"]), args["o"])
	else:
		print("No metadata given, script will terminate")
	exit()

'''
    if args["ds"]:
        ds_lo = int(args["ds"][0])
        try:
            ds_hi = int(args["ds"][1])
        except:
            ds_hi = None
        ds = DataSet(ds_lo, ds_hi,
                     md=run_db, cal = cal_db) #,tier_dir=tier_dir)
        find_cut(ds, ds_lo, args["writeDB"])

    if args["run"]:
        ds = DataSet(run=int(args["run"][0]),
                     md=run_db, cal=cal_db)
        find_cut(ds, ds_lo, args["writeDB"])
'''


#Code to find and record the optimal A/E cut
#def find_cut_using_md(md, r=1, v=0, write_db=False, o=False):
#def find_cut_using_md(md,  v=0, write_db=False, o=False, calDB = None):
def find_cut_using_md(md,  v=0, o=False, calDB = None):

	if o:
		
		if os.path.isdir("output"):
			output_dir = input("Directory output already exists, enter different name:\n")
		else:
			output_dir = "output"
		os.mkdir(output_dir)
		output_name = output_dir + "/low_cut_results.pdf"
		pdf = PdfPages(output_name)
		print("Output will be saved to ./{}".format(output_name))

	if v > 0:
		print("Starting to read subfiles")


	currents = np.array([])
	aoe = np.array([])
	cal_energies = np.array([])
	a_over_e = np.array([])
	a_over_e_sigma = np.array([])

	file_count = 0

	with open(md, "r") as md_file:
		md_dict = json.load(md_file)
	t2_path = md_dict["tier2_dir"]
	for key in md_dict["ds"].keys():
		try:
			r = int(key)
		except:
			continue
		new_uncal_energies = np.array([])
		new_currents = np.array([])
		new_aoe = np.array([])
		for f in os.listdir(t2_path):
						## make run number 4 digits
			if ("run" + f'{r:04}' in f) and (".lh5" in f):
				data =  h5py.File(t2_path + "/" + f, "r")
				new_uncal_energies = np.append(new_uncal_energies, data["data"]["trapE"])
				new_currents = np.append(new_currents, data["data"]["A_10"])
				#aoe = np.append(new_aoe, data["data"]["AoE"])

				file_count += 1


		## calibration
		if calDB:
			new_cal_energies = apply_calibration(new_uncal_energies, calDB, r)
		else:
			new_cal_energies = do_calibration(new_uncal_energies, md, v=v, output=pdf, run=r)

		uncal_a_over_e = new_currents  / new_cal_energies

		new_a_over_e, new_a_over_e_sigma = aoe_correction(new_cal_energies, uncal_a_over_e, output=pdf, run=r)

		cal_energies = np.append(cal_energies, new_cal_energies)
		a_over_e = np.append(a_over_e, new_a_over_e)
		a_over_e_sigma = np.append(a_over_e_sigma, new_a_over_e_sigma)
		
	print("Found {} output files".format(file_count))

	#e_over_unc = cal_energies / uncal_energies #Needed to normalize or something, idk

	plt.figure(figsize=(14,7))
	_h = plt.hist(cal_energies, range=(0,3000), bins=3000)
	plt.xlabel("Energy in keV")
	plt.ylabel("Counts")
	plt.title("Calibrated energy spectrum", fontsize=25)
	pdf.savefig()
	plt.close()

	#uncal_a_over_e = currents  / cal_energies

	#a_over_e, a_over_e_sigma = aoe_correction(cal_energies, uncal_a_over_e, output_dir=output_dir)

	plt.figure(figsize=(14,7))
	_h = plt.hist2d(cal_energies, a_over_e, range=((0,3000), (0.9,1.05)), bins=(150,150), cmap="jet",  norm=mcolors.PowerNorm(0.3))
	plt.colorbar()
	plt.xlabel("Energy in keV")
	plt.ylabel("A/E in a.u.")
	plt.title("Corrected A/E", fontsize=25)
	pdf.savefig()
	plt.close()

	classifier = calc_classifier(cal_energies, a_over_e, a_over_e_sigma, v=0,)

	find_low_cut_value(cal_energies, classifier, n=4.5, output=pdf)


	pdf.close()

	#find_low_cut(cal_energies, a_over_e, v=v, output_dir=output_dir)


	exit()


def apply_calibration(uncal_energies, calDB, r):
	with open(calDB, "r") as cal_file:
		cal_dict = json.load(cal_file)
	parameters = cal_dict["cal_pass3"]["{}".format(r)]

	cal_energies = uncal_energies * parameters["slope"] + parameters["offset"]
	return cal_energies



def aoe_correction(cal_energies, uncal_a_over_e, v=0, output=None, run=1):

	if v > 0:
		print("Starting A/E correction")
	pdf = output

	## take intervalls along compton edge
	interval_centers = np.linspace(1000,2000,25)
	## add some gamma lines
	interval_centers = np.append(np.array([238.6, 583.2, ]), interval_centers)
	aoe_means = []
	aoe_sigmas = []
	
	for center in interval_centers:
		mean, sigma = fit_aoe_histogram(cal_energies, uncal_a_over_e, center, v=v, output_pdf=pdf)
		aoe_means += [mean]
		aoe_sigmas += [sigma]

	aoe_corr_par, aoe_corr_cov = curve_fit(aoe_root, interval_centers, aoe_means, maxfev=1000000)
	plot_xs = np.linspace(200, 3000, 2800)
	plt.figure(figsize=(14,7))
	plt.plot(interval_centers, aoe_means, "bx")
	plt.plot(plot_xs, aoe_root(plot_xs, *aoe_corr_par), "red")
	plt.title("Fit for A/E correction", fontsize=25)
	plt.xlabel("Energy in keV", fontsize=15)
	plt.ylabel("Mean of A/E distribution in a.u.", fontsize=15)
	pdf.savefig()
	plt.close()

	aoe_sigma_par, aoe_sigma_cov = curve_fit(aoe_sigma_root, interval_centers, aoe_sigmas, maxfev=1000000, method="lm")

	a_over_e = uncal_a_over_e / aoe_root(cal_energies, *aoe_corr_par)
	a_over_e_sigma = aoe_sigma_root(cal_energies, *aoe_sigma_par)/aoe_root(cal_energies, *aoe_corr_par)

	return a_over_e, a_over_e_sigma

def calc_classifier(cal_energies, a_over_e, a_over_e_sigma, v=0):

	if v > 0:
		print("Starting classifier calculation")

	classifier = (a_over_e - 1)/a_over_e_sigma

	#plt.hist2d(cal_energies, classifier, range=((0,3000),(-20,20)), bins=(150,40))
	#plt.show()

	return classifier

def find_low_cut_value(cal_energies, classifier, n=4.5, v=0, output=None):

	if v > 0:
		print("Starting calibration")
	pdf = output

	dep_range = 1592.5 - 20, 1592.5 + 20
	fep_range = 2614.5 - 20, 2614.5 + 20
	bins=400

	
	dep_class_range = -50, 50
	dep_class_bins = 1000

	fep_class_range = -200, 50
	fep_class_bins = 2500

	dep_range = 1592.5 - 20, 1592.5 + 20
	bins=400
	dep_energies = cal_energies[np.where((cal_energies > dep_range[0]) & (cal_energies < dep_range[1]))]
	dep_energy_counts, dep_energy_edges = np.histogram(dep_energies, range=dep_range, bins=bins)
	dep_energy_centers = (dep_energy_edges[1:] + dep_energy_edges[:-1])/2
	p0 = guess_parameters(dep_energy_centers, dep_energy_counts)
	dep_energy_mean, dep_energy_sigma = curve_fit(gauss_tail_step, dep_energy_centers, dep_energy_counts, p0=p0,  maxfev=1000000, method="trf")[0][1:3] 
	dep_class_centers, dep_class_counts = select_classifier_values(cal_energies, classifier, dep_energy_mean, dep_energy_sigma, bins=dep_class_bins, range=dep_class_range)

	init_dep_counts = np.sum(dep_class_counts)
	init_dep_class_counts = dep_class_counts

	cut_value = -5
	while np.sum(dep_class_counts) / init_dep_counts > 0.9:

		cut_value += 0.1
		dep_class_centers, dep_class_counts = select_classifier_values(cal_energies[np.where(classifier > cut_value)], classifier[np.where(classifier > cut_value)], dep_energy_mean, dep_energy_sigma, bins=dep_class_bins, range=dep_class_range)
	if pdf :
		plt.figure(figsize=(14,7))
		plt.plot(dep_class_centers, init_dep_class_counts)
		plt.plot(dep_class_centers, dep_class_counts)
		#plt.vlines(cut_value, 0, 1.2*np.amax(init_dep_class_counts), "red")
		plt.xlabel("Classifier in a.u.")
		plt.ylabel("Counts")
		plt.title("Low cut for DEP")
		plt.text(-45, 0.95*np.amax(init_dep_class_counts), "Low cut value: {} \n Survival fraction: {}".format(cut_value, np.sum(dep_class_counts) / init_dep_counts))
		pdf.savefig()
		plt.close()

	fep_energies = cal_energies[np.where((cal_energies > fep_range[0]) & (cal_energies < fep_range[1]))]
	fep_energy_counts, fep_energy_edges = np.histogram(fep_energies, range=fep_range, bins=bins)
	fep_energy_centers = (fep_energy_edges[1:] + fep_energy_edges[:-1])/2
	p0 = guess_parameters(fep_energy_centers, fep_energy_counts	)
	fep_energy_mean, fep_energy_sigma = curve_fit(gauss_tail_step, fep_energy_centers, fep_energy_counts, p0=p0,  maxfev=1000000, method="trf")[0][1:3]

	fep_class_centers, init_fep_class_counts = select_classifier_values(cal_energies, classifier, fep_energy_mean, fep_energy_sigma, bins=fep_class_bins, range=fep_class_range)
	init_fep_counts = np.sum(init_fep_class_counts)

	fep_class_centers, fep_class_counts = select_classifier_values(cal_energies[np.where(classifier > cut_value)], classifier[np.where(classifier > cut_value)], fep_energy_mean, fep_energy_sigma, bins=fep_class_bins, range=fep_class_range)

	fep_counts = np.sum(fep_class_counts)

	print(fep_counts/init_fep_counts)
	print(cut_value)

	if pdf :
		plt.figure(figsize=(14,7))
		plt.plot(fep_class_centers, init_fep_class_counts)
		plt.plot(fep_class_centers, fep_class_counts)
		#plt.vlines(cut_value, 0, 1.2*np.amax(init_fep_class_counts), "red")
		plt.title("Low cut for FEP")
		plt.xlabel("Classifier in a.u.")
		plt.ylabel("Counts")
		#plt.text(-175, 0.95*np.amax(init_fep_class_counts), "Low cut value: {} \n Survival fraction: {}".format(cut_value, fep_counts/init_fep_counts))
		pdf.savefig()
		plt.close()

def select_classifier_values(cal_energies, classifier, mean, sigma, n=4.5, bins=500, range=(-25,25),):
	## selecting classifier values satisfying given cut around given mean using given sigma (* n) and subtracting background
	line_classifier = classifier[np.where((cal_energies > mean - n*sigma)&(cal_energies < mean + n*sigma))]
	classifier_bkg_1 = classifier[np.where((cal_energies > mean - 2*n*sigma)&(cal_energies < mean - n*sigma))]
	classifier_bkg_2 = classifier[np.where((cal_energies > mean + n*sigma)&(cal_energies < mean + 2*n*sigma))]
	bin_centers = (np.histogram(classifier, range=range, bins=bins)[1][1:] + np.histogram(classifier, range=range, bins=bins)[1][:-1])/2

	if mean < 2000 : ## 3 bkg regions if DEP
		classifier_bkg_3 = classifier[np.where((cal_energies > mean + 2*n*sigma)&(cal_energies < mean + 3*n*sigma))]
		subtracted_classifier = np.histogram(line_classifier, range=range, bins=bins)[0] - np.histogram(classifier_bkg_1, range=range, bins=bins)[0] - 2* np.histogram(classifier_bkg_2, range=range, bins=bins)[0] + np.histogram(classifier_bkg_3, range=range, bins=bins)[0]
		return bin_centers, subtracted_classifier
	else: ## only 2 regions if fep
		subtracted_classifier = np.histogram(line_classifier, range=range, bins=bins)[0] - np.histogram(classifier_bkg_1, range=range, bins=bins)[0] - np.histogram(classifier_bkg_2, range=range, bins=bins)[0]
		return bin_centers, subtracted_classifier
		
	

def aoe_dep_mean_correction(cal_energies, a_over_e, n=4.5, v=0, output_dir=None):

	bins = 1500
	range = 0, 1.5


	if v > 0:
		print("Starting normalization on DEP mean")

	if output_dir:
		aoe_dep_norm_output_dir = output_dir + "/aoe_dep_normalization"
		os.mkdir(aoe_dep_norm_output_dir)
	else :
		aoe_dep_norm_output_dir = None

	dep_energies = cal_energies[np.where((dep_energies > 1592.5-20) & (dep_energies < 1592.5+20))]

	dep_energy_mean, dep_energy_sigma = fit_energy_histogram(dep_energies, 1592.5, v=v)

	## DEP region
	dep_aoe = a_over_e[np.where( (energies > dep_energy_mean - n * dep_energy_sigma) & (energies < dep_energy_mean + n * dep_energy_sigma)  )]
	### background region I
	dep_aoe_bkg_1 = a_over_e[np.where( (energies > dep_energy_mean - 2*n*dep_energy_sigma) & (energies < dep_energy_mean - n*dep_energy_sigma)  )]
	## background region II
	dep_aoe_bkg_2 = a_over_e[np.where( (energies > dep_energy_mean + n * dep_energy_sigma) & (energies < dep_energy_mean + 2 * n * dep_energy_sigma)  )]	
	### background region III
	dep_aoe_bkg_3 = a_over_e[np.where( (energies > dep_energy_mean + 2 * n * dep_energy_sigma) & (energies < dep_energy_mean + 3 * n * dep_energy_sigma)  )]

	dep_sub_aoe = np.histogram(init_dep_aoe, bins=bins, range=range)[0] - np.histogram(init_dep_aoe_bkg_1, bins=bins, range=range)[0] - 2*np.histogram(init_dep_aoe_bkg_2, bins=bins, range=range)[0] + np.histogram(init_dep_aoe_bkg_3, bins=bins, range=range)[0]
	edges = np.histogram(init_dep_aoe, bins=bins, range=range)[1]
	centers = (edges[1:] + edges[:-1])/2

	dep_aoe_mean = curve_fit(aoe_fit_line, centers, dep_sub_aoe,  p0=p0, maxfev=10000, method="lm")[0][1]

	norm_a_over_e = a_over_e / dep_aoe_mean
	return norm_a_over_e



def fit_aoe_histogram(cal_energies, uncal_a_over_e, interval_center, v=0, output_pdf=None):

	## select only A/E values for given peak
	aoe = uncal_a_over_e[np.where((cal_energies > (interval_center-20)) & (cal_energies < (interval_center + 20)))]

    ## trying to figure out sensible binning
	counts, edges = np.histogram(aoe, bins=3000)
	centers = (edges[:-1] + edges[1:]) / 2
	mean = centers[np.where(np.amax(counts)==counts)][0]
	## make hist +/- 1% of mean
	counts, edges = np.histogram(aoe, bins=3000, range=(0.99*mean, 1.01*mean))
	centers = (edges[:-1] + edges[1:]) / 2
	## try to get FWHM from that
	fwhm = centers[np.where(counts>np.amax(counts/2))][-1] - centers[np.where(counts>np.amax(counts/2))][0]
	## set range to +/-5 FWHM
	range = mean-5*fwhm, mean+5*fwhm
	## fill into histogram
	counts, edges = np.histogram(aoe, bins=250, range=range)
	centers = (edges[:-1] + edges[1:]) / 2
	#plt.plot(centers, hist, drawstyle="steps-mid")

	p0 = guess_aoe_parameters(centers, counts)
	aoe_fit_par, aoe_fit_cov = curve_fit(aoe_fit_line, centers, counts, p0=p0, maxfev=100000000, method="lm")


	if output_pdf:
		plt.figure(figsize=(14,7))
		plt.plot(centers, counts, drawstyle="steps-mid")
		plt.plot(centers, aoe_fit_line(centers, *aoe_fit_par), "red")
		plt.title("A/E line fit for +/- 20 keV intervall around {} keV".format(interval_center), fontsize=25)
		plt.xlabel("A/E in a.u.", fontsize=15)
		plt.ylabel("Counts", fontsize=15)
		output_pdf.savefig()
		plt.close()


	#results = {"line" : interval_center,
	#			"values" : {"mean" : aoe_fit_par[1], "sigma" : aoe_fit_par[2]},
	#			"errors" : {"mean" : math.sqrt(aoe_fit_cov[1,1]),"sigma" : math.sqrt(aoe_fit_cov[2,2])}}

	results =  aoe_fit_par[1], aoe_fit_par[2]

	if v == 2:
		print("Results of fitting:")
		pprint(results)


	return results




def do_calibration(energies, md, v=0, output=None, run=1):
	
	if v > 0:
		print("Starting calibration")
	pdf = output


	## see if peaks for calibration set, otherwise use some standard peaks
	#try:
	#	cal_peaks = md["cal_peaks"]
	#except:
	cal_peaks = np.array([238.6, 583.2, 1592.5, 2614.5])
	#cal_peaks = np.array([583.2, 2614.5])

	## coarse method to estimate uncal peak positions
	uncal_counts, uncal_edges = np.histogram(energies, bins=500)
													## max in last third of histogram probably 208Tl
	estimated_uncal_peaks = uncal_edges[np.where(uncal_counts == np.amax(uncal_counts[int(1/3 * len(uncal_counts)):]))]*cal_peaks/2614.5

	uncal_peaks = []
	for idx, uncal_peak in enumerate(estimated_uncal_peaks):
		#uncal_peaks += [fit_energy_histogram(energies, uncal_peak, cal_peaks[idx], v=0, output_dir=cal_output_dir)["values"]["mean"]]
		uncal_peaks += [fit_energy_histogram(energies, uncal_peak, cal_peaks[idx], v=0, output_pdf=pdf)[0]]

	calibration_par, calibration_cov = curve_fit(square, uncal_peaks, cal_peaks)
	if v == 2:
		print("Calibration function is {} x**2 + {} x + {}".format(*calibration_par))
	cal_energies = square(energies, *calibration_par)


	return cal_energies



def fit_energy_histogram(energies, est_uncal_line, cal_line=None, v=0, output_pdf=None):

	if not cal_line:
		cal_line = est_uncal_line

	## trying to figure out sensible binning
	counts, edges = np.histogram(energies, bins=3000, range=(est_uncal_line-100, est_uncal_line+100))
	centers = (edges[:-1] + edges[1:]) / 2
	mean = centers[np.where(np.amax(counts)==counts)][0]
    
	## make hist +/- 1% of mean
	counts, edges = np.histogram(energies, bins=3000, range=(0.99*mean, 1.01*mean))
	centers = (edges[:-1] + edges[1:]) / 2
	## try to get FWHM from that
	fwhm = centers[np.where(counts>np.amax(counts/2))][-1] - centers[np.where(counts>np.amax(counts/2))][0]
	## set range to +/-2 FWHM
	range=mean-2*fwhm, mean+2*fwhm
	bins=int((range[1] - range[0]))
	counts, edges = np.histogram(energies, bins=bins, range=range)
	centers = (edges[1:] + edges[:-1])/2

	## guess starting values
	p0 = guess_parameters(centers,counts)

	## finally do some fitting
	uncal_line_par, uncal_line_cov = curve_fit(gauss_tail_step, centers, counts, p0=p0, maxfev=500000, method="lm")

	if output_pdf:
		plt.figure(figsize=(14,7))
		plt.plot(centers, counts, drawstyle="steps-mid")
		plt.plot(centers, gauss_tail_step(centers, *uncal_line_par), "red")
		plt.title("Energy line fit for {} keV line".format(cal_line), fontsize=25)
		plt.xlabel("Energy in a.u.", fontsize=15)
		plt.ylabel("Counts", fontsize=15)
		output_pdf.savefig()
		plt.close()
    
	#results = {"line" : cal_line,
	#			"values" : {"integral" : uncal_line_par[0], "mean" : uncal_line_par[1], "sigma" : uncal_line_par[2]},
	#			"errors" : {"integral" : math.sqrt(uncal_line_cov[1,1]), "mean" : math.sqrt(uncal_line_cov[1,1]),"sigma" : math.sqrt(uncal_line_cov[2,2])}}

	results = uncal_line_par[1], uncal_line_par[2]

	if v == 2:
		print("Results of fitting:")
		pprint(results)

	return results

    #plt.bar(centers, b, width=0.04*line/bins)
    #plt.plot(centers,LineFit(centers,*uncal_line_popt), "red")
    #plt.xlabel("Uncalibrated energy values / a.u.")
    #plt.ylabel("Counts")
    #plt.show()
    #plt.clf()

# gamma line fit as in HADES GSTR

def gaussian(xs, amp, mean, sigma):
    return amp/((2*math.pi)**0.5*sigma) * np.exp(-(xs-mean)**2/(2*sigma**2))

def tail(xs, mean, S_t, beta, gamma):
    return S_t/(2*beta) * np.exp((xs-mean)/beta + gamma**2/(2*beta**2))*erf((xs-mean)/(2**0.5*gamma) + gamma/(2**0.5*beta))

def step(xs, mean, sigma, A, B):
    return A/2 * erf((xs-mean)/(2**0.5*sigma))

def gauss_tail_step(xs, amp, mean, sigma, S_t, beta, gamma, A, B):
    return gaussian(xs, amp, mean, sigma) + tail(xs, mean, S_t, beta, gamma) + step(xs, mean, sigma, A, B)

# other functions for fitting

def linear(xs, a, b):
    return np.array([a*x + b for x in xs])

def square(xs, a, b, c):
    return a*xs*xs + b*xs + c

def aoe_tail(xs, m, l, f, d, t):
    return m*(np.exp(f*(xs-l))+d)/(np.exp((xs-l)*t) + l)

def aoe_fit_line(xs, amp, mean, sigma, m, l, f, d, t):
    return gaussian(xs, amp, mean, sigma) + aoe_tail(xs, m, l, f, d, t)


def guess_parameters(xs, ys):
    ys = savgol_filter(ys, 11, 2)
    ## find height of peak
    raw_height = np.amax(ys)
    ## determin position of max
    mean = xs[np.where(ys == raw_height)][0]
    ## get FWHM
    sigma = (xs[np.where(ys > raw_height/2**0.5)][-1] - xs[np.where(ys > raw_height/2**0.5)][0])/2
    ## take into account normalization
    height = raw_height * (2**0.5 * sigma)
    ## subtract gaussian from spectrum to get handle on the rest of the peak
    rest_ys = ys - height/(2**0.5*sigma) * np.exp(-(xs-mean)**2/(2*sigma**2))
    
    
    ## value of constant so that step goes to 0 on other end of spectrum, in the recent guess just 0
    const_height = ys[-1]
    ## try to find step far away from mean, so tail doesn't interfere, subtract constant at other end of spectrum
    step_height = ys[0] - const_height
    ## subtract infinitely sharp step and constant, only a coarse guess
    rest_ys = rest_ys - step_height*np.heaviside(mean-xs, 0.5) - const_height
    
    ##only tail leftover
    raw_tail_height = np.amax(rest_ys)
    if raw_tail_height < 20:
    	return [height, mean, sigma, 0, 1, 1, step_height, const_height]
    ## beta something like a decay length to the lower end, so try something like this
    beta = mean - xs[np.where(rest_ys > raw_tail_height/math.e)][0]
    ## correct tail_height with estimated beta
    tail_height = raw_tail_height * 2 * beta
    ## gamma more or less some kind of sigma, but only to the higher end
    gamma = xs[np.where(rest_ys > raw_tail_height/math.e)][-1] - mean
    
    return [height, mean, sigma, tail_height, beta, gamma, step_height, const_height]


def guess_aoe_parameters(centers, counts):
    counts = savgol_filter(counts, 31, 2)

    ## height without normalization, only for mean and sigma guessing
    raw_height = np.amax(counts)
    ## mean as center of bin with highest count
    mean = centers[np.where(counts == raw_height)][0]
    ## sigma as half of FWHM
    sigma = (centers[np.where(counts>raw_height/2)][-1] - centers[np.where(counts>raw_height/2)][0]) / 2.3548#(2*(2*math.log(2))**0.5)
    ## normalize height the way it occurs in function
    height = (2*math.pi)**0.5 * sigma * raw_height

    ## subtract gauss, really low level
    remaining_counts = counts - gaussian(centers, height, mean, sigma)
    #plt.plot(centers, remaining_counts)
    #plt.plot(centers, counts)
    #plt.show()
    
    ## just like for the gauss
    tail_center = centers[np.where(np.amax(remaining_counts) == remaining_counts)][0]
    tail_height = np.amax(remaining_counts)
    
    ## interpret other values as kind of decay lengths
    tail_right =  .05/(centers[np.where(remaining_counts > tail_height/math.e)][-1] - tail_center)
    tail_left = 10/(tail_center - centers[np.where(remaining_counts > tail_height/math.e)][0])


    #return [height, mean, sigma, tail_height, tail_center, tail_right, 0, tail_left]
    return [height, mean, sigma, tail_height, tail_center+5*sigma, tail_right, 0, tail_left]

def aoe_root(xs, a, b, c):
    return a + b / np.sqrt(xs - c)

def aoe_sigma_root(xs, a, b, c):
	return np.sqrt(a + b/(xs-c))


def find_cut(ds, ds_lo, write_db=False):

    #Make tier2 dataframe
    t2 = ds.get_t2df()
    t2 = t2.reset_index(drop=True)

    #Get e_ftp and pass1 calibration constant TODO: need pass2 constants at some point
    calDB = ds.calDB
    query = db.Query()
    table = calDB.table("cal_pass1")
    vals = table.all()
    df_cal = pd.DataFrame(vals) # <<---- omg awesome
    df_cal = df_cal.loc[df_cal.ds==ds_lo]
    p1cal = df_cal.iloc[0]["p1cal"]
    cal = p1cal * np.asarray(t2["e_ftp"])

    #Make A/E array
    current = "current_max"
    e_over_unc = cal / np.asarray(t2["e_ftp"]) #Needed to normalize or something, idk
    y0 = np.asarray(t2[current])
    a_over_e = y0 * e_over_unc / cal

    y = linear_correction(cal, a_over_e) # Linear correct slight downward trend

    test_code(y, cal, ds)
    exit()

    # Two separate functions, one for Ac contaminated peak(Th232), one for Th228
    ans = input('Are you running A/E on Th232? \n y/n -->')
    if ans == 'y':
        line = th_232(cal, y, ds)
    else:
        line = regular_cut(cal, y, ds)

    # Write cut to the calDB.json file
    if write_db:
        table = calDB.table("A/E_cut")
        for dset in ds.ds_list:
            row = {"ds":dset, "line":line}
            table.upsert(row, query.ds == dset)






if __name__=="__main__":
    main()