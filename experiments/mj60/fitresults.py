import ROOT
import numpy as np
import json, os, sys
import matplotlib.pyplot as plt
import scipy.optimize as opt

# first we input the fit results we store in our metadata directory
if(len(sys.argv) != 3):
    print('Usage: fitresults.py [input root histogram] [fit results root file]')
    sys.exit()

with open("runDB.json") as f:
    runDB = json.load(f)
meta_dir = os.path.expandvars(runDB["meta_dir"])

fitresultsfile = "{}/{}".format(meta_dir,sys.argv[2])
fitresultsname = "results"
infile = ROOT.TFile.Open(fitresultsfile, "READ")
results = infile.Get(fitresultsname)
infile.Close()

histfile = "{}/{}".format(meta_dir,sys.argv[1])
histname = "root_hist"


def main():

    #get_peak_energies()
    #get_calibration_params_using_peaks()
    #get_calibration_params_using_centroids()
    #get_resolution_params()
    #read_hmc_results()
    draw_results()


def get_peak_energies():

    pks_start = results.GetParsIndexForPar(ROOT.GATMultiPeakFitter.kAmp)
    number_of_peaks = results.NParsForPar(ROOT.GATMultiPeakFitter.kAmp)
    print("\nThe peaks in the MultiPeakFitter are:")
    for i in range(pks_start,pks_start+number_of_peaks):
        pk_energy = results.GetPeakEnergy(i)
        print("{} keV".format(pk_energy))


def get_calibration_params_using_peaks():

    # this is for linear calibrations
 
    # get the parameters and their errors
    pars_number = results.NParsForPar(ROOT.GATMultiPeakFitter.kMu)
    if pars_number==2:
        mu0 = results.GetParsForPar(ROOT.GATMultiPeakFitter.kMu)[0]
        mu1 = results.GetParsForPar(ROOT.GATMultiPeakFitter.kMu)[1]
        sig_mu0 = np.sqrt(results.GetCovariance(ROOT.GATMultiPeakFitter.kMu, 0, ROOT.GATMultiPeakFitter.kMu, 0))
        sig_mu1 = np.sqrt(results.GetCovariance(ROOT.GATMultiPeakFitter.kMu, 1, ROOT.GATMultiPeakFitter.kMu, 1))
        sig_mu0_mu1 = results.GetCovariance(ROOT.GATMultiPeakFitter.kMu, 0, ROOT.GATMultiPeakFitter.kMu, 1)
        p0 = -mu0/mu1
        p1 = 1/mu1
        p0_uncertainty = np.sqrt((mu0/mu1**2)**2*sig_mu1**2+(1/mu1)**2*sig_mu0**2-2*mu0*sig_mu0_mu1/(mu1**3))
        p1_uncertainty = np.sqrt((1/mu1**2)**2*sig_mu1**2)
        print("\nThe energy calibration parameters are:")
        print("p0 = {}".format(p0))
        print("p1 = {}".format(p1))
        print("p0_uncertainty = {}".format(p0_uncertainty))
        print("p1_uncertainty = {}".format(p1_uncertainty))
    else:
        print("The energy calibration is not linear.")


def get_calibration_params_using_centroids():

    print("centroid = {}".format(results.GetCentroid(2614.511)))

    # get parameters and errors necessary for centroid
    mu0 = results.GetParsForPar(ROOT.GATMultiPeakFitter.kMu)[0]
    mu1 = results.GetParsForPar(ROOT.GATMultiPeakFitter.kMu)[1]
    f_le = results.GetParsForPar(ROOT.GATMultiPeakFitter.kFt)[0]
    tau_le0 = results.GetParsForPar(ROOT.GATMultiPeakFitter.kTau)[0]
    tau_le1 = results.GetParsForPar(ROOT.GATMultiPeakFitter.kTau)[1]
    f_he = results.GetParsForPar(ROOT.GATMultiPeakFitter.kFht)[0]
    tau_he = results.GetParsForPar(ROOT.GATMultiPeakFitter.kTauHT)[0]

    sig_mu0 = np.sqrt(results.GetCovariance(ROOT.GATMultiPeakFitter.kMu, 0, ROOT.GATMultiPeakFitter.kMu, 0))
    sig_mu1 = np.sqrt(results.GetCovariance(ROOT.GATMultiPeakFitter.kMu, 1, ROOT.GATMultiPeakFitter.kMu, 1))
    sig_f_le = np.sqrt(results.GetCovariance(ROOT.GATMultiPeakFitter.kFt, 0, ROOT.GATMultiPeakFitter.kFt, 0))
    sig_tau_le0 = np.sqrt(results.GetCovariance(ROOT.GATMultiPeakFitter.kTau, 0, ROOT.GATMultiPeakFitter.kTau, 0))
    sig_tau_le1 = np.sqrt(results.GetCovariance(ROOT.GATMultiPeakFitter.kTau, 1, ROOT.GATMultiPeakFitter.kTau, 1))
    sig_f_he = np.sqrt(results.GetCovariance(ROOT.GATMultiPeakFitter.kFht, 0, ROOT.GATMultiPeakFitter.kFht, 0))
    sig_tau_he = np.sqrt(results.GetCovariance(ROOT.GATMultiPeakFitter.kTauHT, 0, ROOT.GATMultiPeakFitter.kTauHT, 0))

    sig_mu0_mu1 = results.GetCovariance(ROOT.GATMultiPeakFitter.kMu, 0, ROOT.GATMultiPeakFitter.kMu, 1)
    sig_mu0_f_le = results.GetCovariance(ROOT.GATMultiPeakFitter.kMu, 0, ROOT.GATMultiPeakFitter.kFt, 0)
    sig_mu0_tau_le0 = results.GetCovariance(ROOT.GATMultiPeakFitter.kMu, 0, ROOT.GATMultiPeakFitter.kTau, 0)
    sig_mu0_tau_le1 = results.GetCovariance(ROOT.GATMultiPeakFitter.kMu, 0, ROOT.GATMultiPeakFitter.kTau, 1)
    sig_mu0_f_he = results.GetCovariance(ROOT.GATMultiPeakFitter.kMu, 0, ROOT.GATMultiPeakFitter.kFht, 0)
    sig_mu0_tau_he = results.GetCovariance(ROOT.GATMultiPeakFitter.kMu, 0, ROOT.GATMultiPeakFitter.kTauHT, 0)

    sig_mu1_f_le = results.GetCovariance(ROOT.GATMultiPeakFitter.kMu, 1, ROOT.GATMultiPeakFitter.kFt, 0)
    sig_mu1_tau_le0 = results.GetCovariance(ROOT.GATMultiPeakFitter.kMu, 1, ROOT.GATMultiPeakFitter.kTau, 0)
    sig_mu1_tau_le1 = results.GetCovariance(ROOT.GATMultiPeakFitter.kMu, 1, ROOT.GATMultiPeakFitter.kTau, 1)
    sig_mu1_f_he = results.GetCovariance(ROOT.GATMultiPeakFitter.kMu, 1, ROOT.GATMultiPeakFitter.kFht, 0)
    sig_mu1_tau_he = results.GetCovariance(ROOT.GATMultiPeakFitter.kMu, 1, ROOT.GATMultiPeakFitter.kTauHT, 0)

    sig_f_le_tau_le0 = results.GetCovariance(ROOT.GATMultiPeakFitter.kFt, 0, ROOT.GATMultiPeakFitter.kTau, 0)
    sig_f_le_tau_le1 = results.GetCovariance(ROOT.GATMultiPeakFitter.kFt, 0, ROOT.GATMultiPeakFitter.kTau, 1)
    sig_f_le_f_he = results.GetCovariance(ROOT.GATMultiPeakFitter.kFt, 0, ROOT.GATMultiPeakFitter.kFht, 0)
    sig_f_le_tau_he = results.GetCovariance(ROOT.GATMultiPeakFitter.kFt, 0, ROOT.GATMultiPeakFitter.kTauHT, 0)

    sig_tau_le0_tau_le1 = results.GetCovariance(ROOT.GATMultiPeakFitter.kTau, 0, ROOT.GATMultiPeakFitter.kTau, 1)
    sig_tau_le0_f_he = results.GetCovariance(ROOT.GATMultiPeakFitter.kTau, 0, ROOT.GATMultiPeakFitter.kFht, 0)
    sig_tau_le0_tau_he = results.GetCovariance(ROOT.GATMultiPeakFitter.kTau, 0, ROOT.GATMultiPeakFitter.kTauHT, 0)

    sig_tau_le1_f_he = results.GetCovariance(ROOT.GATMultiPeakFitter.kTau, 1, ROOT.GATMultiPeakFitter.kFht, 0)
    sig_tau_le1_tau_he = results.GetCovariance(ROOT.GATMultiPeakFitter.kTau, 1, ROOT.GATMultiPeakFitter.kTauHT, 0)

    sig_f_he_tau_he = results.GetCovariance(ROOT.GATMultiPeakFitter.kFht, 0, ROOT.GATMultiPeakFitter.kTauHT, 0)

    # uncomment these lines to print out the raw parameters used for our centroid calculations
    print("\nmu0 = {}".format(mu0))
    print("mu1 = {}".format(mu1))
    print("f_le = {}".format(f_le))
    print("tau_le0 = {}".format(tau_le0))
    print("tau_le1 = {}".format(tau_le1))
    print("f_he = {}".format(f_he))
    print("tau_he = {}".format(tau_he))

    print("\nsig_mu0 = {}".format(sig_mu0))
    print("sig_mu1 = {}".format(sig_mu1))
    print("sig_f_le = {}".format(sig_f_le))
    print("sig_tau_le0 = {}".format(sig_tau_le0))
    print("sig_tau_le1 = {}".format(sig_tau_le1))
    print("sig_f_he = {}".format(sig_f_he))
    print("sig_tau_he = {}".format(tau_he))

    print("\nsig_mu0_mu1 = {}".format(sig_mu0_mu1))
    print("sig_mu0_f_le = {}".format(sig_mu0_f_le))
    print("sig_mu0_tau_le0 = {}".format(sig_mu0_tau_le0))
    print("sig_mu0_tau_le1 = {}".format(sig_mu0_tau_le1))
    print("sig_mu0_f_he = {}".format(sig_mu0_f_he))
    print("sig_mu0_tau_he = {}".format(sig_mu0_tau_he))

    print("\nsig_mu1_f_le = {}".format(sig_mu1_f_le))
    print("sig_mu1_tau_le0 = {}".format(sig_mu1_tau_le0))
    print("sig_mu1_tau_le1 = {}".format(sig_mu1_tau_le1))
    print("sig_mu1_f_he = {}".format(sig_mu1_f_he))
    print("sig_mu1_tau_he = {}".format(sig_mu1_tau_he))

    print("\nsig_f_le_tau_le0 = {}".format(sig_f_le_tau_le0))
    print("sig_f_le_tau_le1 = {}".format(sig_f_le_tau_le1))
    print("sig_f_le_f_he = {}".format(sig_f_le_f_he))
    print("sig_f_le_tau_he = {}".format(sig_f_le_tau_he))

    print("\nsig_tau_le0_tau_le1 = {}".format(sig_tau_le0_tau_le1))
    print("sig_tau_le0_f_he = {}".format(sig_tau_le0_f_he))
    print("sig_tau_le0_tau_he = {}".format(sig_tau_le0_tau_he))

    print("\nsig_tau_le1_f_he = {}".format(sig_tau_le1_f_he))
    print("sig_tau_le1_tau_he = {}".format(sig_tau_le1_tau_he))

    print("\nsig_f_he_tau_he = {}".format(sig_f_he_tau_he))

    def centroid(E):
        centroid = mu0 + mu1*E  - f_le*(tau_le0 + tau_le1*E) + f_he*tau_he
        return centroid

    def linear_calibration(x, a, b):
        return a + b*x 

    print("\ncentroid = {}".format(centroid(2614.511)))

    popt, pcov = opt.curve_fit(linear_calibration, centroid(np.arange(0,3000,1)), np.arange(0,3000,1), sigma = None)

    p0 = popt[0]
    p1 = popt[1]

    print("p0 = {}".format(p0))
    print("p1 = {}".format(p1))


def get_resolution_params():

    # the resolution parameters are currently is units of ADC
    # the parameters need to be converted to the correct units, and errors need to be propagated

    # first get the necessary values from the multipeak fitter
    sig0 = results.GetParsForPar(ROOT.GATMultiPeakFitter.kSig)[0]
    sig1 = results.GetParsForPar(ROOT.GATMultiPeakFitter.kSig)[1]
    sig2 = results.GetParsForPar(ROOT.GATMultiPeakFitter.kSig)[2]
    mu0 = results.GetParsForPar(ROOT.GATMultiPeakFitter.kMu)[0]
    mu1 = results.GetParsForPar(ROOT.GATMultiPeakFitter.kMu)[1]
    
    sig_sig0 = np.sqrt(results.GetCovariance(ROOT.GATMultiPeakFitter.kSig, 0, ROOT.GATMultiPeakFitter.kSig, 0))
    sig_sig1 = np.sqrt(results.GetCovariance(ROOT.GATMultiPeakFitter.kSig, 1, ROOT.GATMultiPeakFitter.kSig, 1))
    sig_sig2 = np.sqrt(results.GetCovariance(ROOT.GATMultiPeakFitter.kSig, 2, ROOT.GATMultiPeakFitter.kSig, 2))
    sig_mu0 = np.sqrt(results.GetCovariance(ROOT.GATMultiPeakFitter.kMu, 0, ROOT.GATMultiPeakFitter.kMu, 0))
    sig_mu1 = np.sqrt(results.GetCovariance(ROOT.GATMultiPeakFitter.kMu, 1, ROOT.GATMultiPeakFitter.kMu, 1))
    
    sig_sig0_sig1 = results.GetCovariance(ROOT.GATMultiPeakFitter.kSig, 0, ROOT.GATMultiPeakFitter.kSig, 1)
    sig_sig0_sig2 = results.GetCovariance(ROOT.GATMultiPeakFitter.kSig, 0, ROOT.GATMultiPeakFitter.kSig, 2)
    sig_sig0_mu0 = results.GetCovariance(ROOT.GATMultiPeakFitter.kSig, 0, ROOT.GATMultiPeakFitter.kMu, 0)
    sig_sig0_mu1 = results.GetCovariance(ROOT.GATMultiPeakFitter.kSig, 0, ROOT.GATMultiPeakFitter.kMu, 1)
    
    sig_sig1_sig2 = results.GetCovariance(ROOT.GATMultiPeakFitter.kSig, 1, ROOT.GATMultiPeakFitter.kSig, 2)
    sig_sig1_mu0 = results.GetCovariance(ROOT.GATMultiPeakFitter.kSig, 1, ROOT.GATMultiPeakFitter.kMu, 0)
    sig_sig1_mu1 = results.GetCovariance(ROOT.GATMultiPeakFitter.kSig, 1, ROOT.GATMultiPeakFitter.kMu, 1)
    
    sig_sig2_mu0 = results.GetCovariance(ROOT.GATMultiPeakFitter.kSig, 2, ROOT.GATMultiPeakFitter.kMu, 0)
    sig_sig2_mu1 = results.GetCovariance(ROOT.GATMultiPeakFitter.kSig, 2, ROOT.GATMultiPeakFitter.kMu, 1)
    
    sig_mu0_mu1 = results.GetCovariance(ROOT.GATMultiPeakFitter.kMu, 0, ROOT.GATMultiPeakFitter.kMu, 1)

    # uncomment these lines to print out the raw parameters used for our calculation
    #print("sig0 = {}".format(sig0))
    #print("sig1 = {}".format(sig1))
    #print("sig2 = {}".format(sig2))
    #print("mu0 = {}".format(mu0))
    #print("mu1 = {}".format(mu1))
    
    #print("sig_sig0 = {}".format(sig_sig0))
    #print("sig_sig1 = {}".format(sig_sig1))
    #print("sig_sig2 = {}".format(sig_sig2))
    #print("sig_mu0 = {}".format(sig_mu0))
    #print("sig_mu1 = {}".format(sig_mu1))
    
    #print("sig_sig0_sig1 = {}".format(sig_sig0_sig1))
    #print("sig_sig0_sig2 = {}".format(sig_sig0_sig2))
    #print("sig_sig0_mu0 = {}".format(sig_sig0_mu0))
    #print("sig_sig0_mu1 = {}".format(sig_sig0_mu1))
    
    #print("sig_sig1_sig2 = {}".format(sig_sig1_sig2))
    #print("sig_sig1_mu0 = {}".format(sig_sig1_mu0))
    #print("sig_sig1_mu1 = {}".format(sig_sig1_mu1))
    
    #print("sig_sig2_mu0 = {}".format(sig_sig2_mu0))
    #print("sig_sig2_mu1 = {}".format(sig_sig2_mu1))
    
    #print("sig_mu0_mu1 = {}".format(sig_mu0_mu1))

    # we compute the parameters for the resolution in energy
    sigA = np.sqrt((sig0/mu1)**2 + mu0*(sig1/mu1)**2 + (sig2*mu0/mu1)**2)
    sigB = np.sqrt(2*mu0*(sig2**2)/mu1 + (sig1**2)/mu1)
    sigC = np.sqrt(sig2**2)

    # the errors on the parameters is computed using propagation of errors
    # the derivatives were all checked in mathematica for agreement
    # the following equations come from mathematica
    # they are very dirty, and unfortunately cannot be very cleaned up
    sigma_a_uncertainty = np.sqrt((4*sig_sig0**2*mu1**2*sig0**2 + 4*sig_mu1**2*(sig0**2 + mu0*(sig1**2 + mu0*sig2**2))**2 + mu1*(4*sig_sig1**2*mu0**2*mu1*sig1**2 + sig_mu0**2*mu1*(sig1**2 + 2*mu0*sig2**2)**2 + 4*(sig_sig2**2*mu0**2*mu1*sig2**2 - 2*sig0**3*sig_sig0_mu1 - sig0**2*(sig1**2*sig_mu0_mu1 + 2*mu0*sig1*sig_sig1_mu1 + 2*mu0*sig2*(sig2*sig_mu0_mu1 + sig0*sig_sig2_mu1)) + sig0*(-2*mu0*(sig1**2 + mu0*sig2**2)*sig0*sig_sig0_mu1 + mu1*(sig1 ** 2*sig_sig0_mu0 + 2*mu0*sig1*sig_sig0_sig1 + 2*mu0*sig2*(sig2*sig_sig0_mu0 + mu0*sig_sig0_sig2))) + mu0*(-sig1 **4*sig_mu0_mu1 + sig1**3*(mu1*sig_sig1_mu0 - 2*mu0*sig_sig1_mu1) + mu0*sig1**2*sig2*(-3*sig2*sig_mu0_mu1 + mu1*sig_sig2_mu0 - 2*mu0*sig_sig2_mu1) - 2*mu0**2*sig2**3*(sig2*sig_mu0_mu1 - mu1*sig_sig2_mu0 + mu0*sig_sig2_mu1) + 2*mu0*sig1*sig2*(-mu0*sig2*sig_sig1_mu1 + mu1*(sig2*sig_sig1_mu0 + mu0*sig_sig1_sig2))))))/(4*mu1**4*(sig0**2 + mu0*(sig1**2 + mu0*sig2**2))))

    sigma_b_uncertainty = np.sqrt(((sig1**2 + 2*mu0*sig2**2)**2*sig_mu1**2 - 4*mu1*(sig1**2 + 2*mu0*sig2**2)*(sig2**2*sig_mu0_mu1 + sig1*sig_sig1_mu1 + 2*mu0*sig2*sig_sig2_mu1) + 4*mu1**2*(sig2**4*sig_mu0**2 + sig1**2*sig_sig1**2 + 2*sig2**2*(sig1*sig_sig1_mu0 + 2*mu0**2*sig_sig2**2) + 4*mu0*sig2**3*sig_sig2_mu0 + 4*mu0*sig1*sig2*sig_sig1_sig2))/(4*mu1**3*(sig1**2 + 2*mu0*sig2**2)))

    sigma_c_uncertainty = sig_sig2


    print("\nThe resolution function parameters are:")
    print("sigma_a = {}".format(sigA))
    print("sigma_b = {}".format(sigB))
    print("sigma_c = {}".format(sigC))
    print("sigma_a_uncertainty = {}".format(sigma_a_uncertainty))
    print("sigma_b_uncertainty = {}".format(sigma_b_uncertainty))
    print("sigma_c_uncertainty = {}".format(sigma_c_uncertainty))


    def reso(E):
        return np.sqrt(sigA**2 + (sigB**2)*E + sigC**2*E**2)

    resolutions = reso(np.arange(0,3000,1))

    plt.plot(np.arange(0,3000,1), resolutions, color='black')
    plt.xlabel('Energy (keV)', ha='right', x=1.0)
    plt.ylabel(r'$\sigma$ (keV)', ha='right', y=1.0)
    plt.tight_layout()
    plt.show()


def read_hmc_results():

    print("centroid = {}".format(results.GetCentroid(2614.511)))


def draw_results():

    # open histogram
    infile = ROOT.TFile.Open(histfile, "READ")
    hist = infile.Get(histname)
    hist.SetDirectory(0)
    infile.Close()
    hist.Draw()
    hist.GetXaxis().SetTitle("ADC")
    hist.GetYaxis().SetTitle("Counts")
    hist.SetTitle("MultiPeakFitter Results")

    # open fit results / draw all componets of fit
    infile = ROOT.TFile.Open(fitresultsfile, "READ")
    results = infile.Get(fitresultsname)
    infile.Close()
    results.SetHists(hist)
    #results.DrawRegion(4, "name", "a")

    # draw pulls
    #results.DrawRegion(3, "name", "p")

    # draw entire spectrum w/ fits
    results.DrawRegions()

if __name__ == '__main__':
        main()
