"""
This module is for extracting a single pole zero constant from the decay tail
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import pickle as pkl
from collections import OrderedDict

import matplotlib as mpl

mpl.use("agg")
import matplotlib.pyplot as plt
import numpy as np

import pygama
import pygama.lgdo as lgdo
import pygama.lgdo.lh5_store as lh5
import pygama.math.histogram as pgh
import pygama.math.peak_fitting as pgf
import pygama.pargen.cuts as cts
import pygama.pargen.dsp_optimize as opt
import pygama.pargen.energy_optimisation as om

log = logging.getLogger(__name__)


def load_data(
    raw_file: list[str],
    lh5_path: str,
    n_events: int = 10000,
    threshold: int = 5000,
    wf_field: str = "waveform",
) -> lgdo.Table:
    sto = lh5.LH5Store()
    df = lh5.load_dfs(raw_file, ["daqenergy", "timestamp"], lh5_path)

    pulser_props = cts.find_pulser_properties(df, energy="daqenergy")
    if len(pulser_props) > 0:
        out_df = cts.tag_pulsers(df, pulser_props, window=0.001)
        ids = ~(out_df.isPulser == 1)
        log.debug(f"pulser found: {pulser_props}")
    else:
        log.debug("no_pulser")
        ids = np.ones(len(df.daqenergy.values), dtype=bool)

    cuts = np.where((df.daqenergy.values > threshold) & (ids))[0]

    waveforms = sto.read_object(
        f"{lh5_path}/{wf_field}", raw_file, idx=cuts, n_rows=n_events
    )[0]
    baseline = sto.read_object(
        f"{lh5_path}/baseline", raw_file, idx=cuts, n_rows=n_events
    )[0]
    tb_data = lh5.Table(col_dict={f"{wf_field}": waveforms, "baseline": baseline})
    return tb_data


def get_decay_constant(
    slopes: np.array, wfs: lgdo.WaveformTable, display: int = 0
) -> dict:

    """
    Finds the decay constant from the modal value of the tail slope after cuts
    and saves it to the specified json.

    Parameters
    ----------
    slopes : array
        tail slope array

    dict_file : str
        path to json file to save decay constant value to.
        It will be saved as a dictionary of form {'pz': {'tau': decay_constant}}

    Returns
    -------
    tau_dict : dict
    """
    tau_dict = {}

    pz = tau_dict.get("pz")

    counts, bins, var = pgh.get_hist(slopes, bins=500000, range=(-0.01, 0))
    bin_centres = pgh.get_bin_centers(bins)
    high_bin = bin_centres[np.argmax(counts)]
    try:
        pars, cov = pgf.gauss_mode_width_max(
            counts,
            bins,
            mode_guess=high_bin,
            n_bins=10,
            cost_func="Least Squares",
            inflate_errors=False,
            gof_method="var",
        )
        high_bin = pars[0]
    except:
        pass
    tau = round(-1 / (high_bin), 1)

    sampling_rate = wfs["dt"].nda[0]
    units = wfs["dt"].attrs["units"]
    tau = f"{tau*sampling_rate}*{units}"

    tau_dict["pz"] = {"tau": tau}
    if display > 0:
        out_plot_dict = {}
        plt.rcParams["figure.figsize"] = (10, 6)
        plt.rcParams["font.size"] = 8
        fig, ax = plt.subplots()
        bins = 10000  # change if needed
        counts, bins, bars = ax.hist(slopes, bins=bins, histtype="step")
        plot_max = np.argmax(counts)
        in_min = plot_max - 10
        if in_min < 0:
            in_min = 0
        in_max = plot_max + 11
        if in_max >= len(bins):
            in_min = len(bins) - 1
        plt.xlabel("Slope")
        plt.ylabel("Counts")
        plt.yscale("log")
        axins = ax.inset_axes([0.5, 0.45, 0.47, 0.47])
        axins.hist(
            slopes[(slopes > bins[in_min]) & (slopes < bins[in_max])],
            bins=200,
            histtype="step",
        )
        axins.set_xlim(bins[in_min], bins[in_max])
        labels = ax.get_xticklabels()
        ax.set_xticklabels(labels=labels, rotation=45)
        out_plot_dict["slope"] = fig
        if display > 1:
            plt.show()
        else:
            plt.close()
        return tau_dict, out_plot_dict
    else:
        return tau_dict


def fom_dpz(tb_data, verbosity=0, rand_arg=None):

    std = tb_data["pz_std"].nda
    counts, start_bins, var = pgh.get_hist(std, dx=0.1, range=(0, 400))
    max_idx = np.argmax(counts)
    mu = start_bins[max_idx]
    try:
        pars, cov = pgf.gauss_mode_width_max(
            counts,
            start_bins,
            mode_guess=mu,
            n_bins=10,
            cost_func="Least Squares",
            inflate_errors=False,
            gof_method="var",
        )

        mu = pars[0]

    except:
        mu = start_bins[max_idx]

    return {"y_val": np.abs(mu)}


def get_dpz_consts(grid_out, opt_dict):
    std_grid = np.ndarray(shape=grid_out.shape)
    for i in range(grid_out.shape[0]):
        for j in range(grid_out.shape[1]):
            std_grid[i, j] = grid_out[i, j]["y_val"]
    min_val = np.amin(std_grid)
    min_point = np.where(std_grid == np.amin(std_grid))

    opt_name = list(opt_dict.keys())[0]
    keys = list(opt_dict[opt_name].keys())
    param_list = []
    shape = []
    db_dict = {}
    for key in keys:
        param_dict = opt_dict[opt_name][key]
        grid_axis = np.arange(
            param_dict["start"], param_dict["end"], param_dict["spacing"]
        )
        unit = param_dict.get("unit")
        param_list.append(grid_axis)
        shape.append(len(grid_axis))
    for i, key in enumerate(keys):
        unit = opt_dict[opt_name][key].get("unit")

        if unit is not None:
            try:
                db_dict[opt_name].update(
                    {key: f"{param_list[i][min_point[i]][0]}*{unit}"}
                )
            except:
                db_dict[opt_name] = {key: f"{param_list[i][min_point[i]][0]}*{unit}"}
        else:
            try:
                db_dict[opt_name].update({key: f"{param_list[i][min_point[i]][0]}"})
            except:
                db_dict[opt_name] = {key: f"{param_list[i][min_point[i]][0]}"}
    return db_dict


def dsp_preprocess_decay_const(
    raw_files: list[str],
    dsp_config: dict,
    lh5_path: str,
    double_pz: bool = False,
    display: int = 0,
    opt_dict: dict = None,
    threshold: int = 5000,
    wf_field: str = "waveform",
    wf_plot: str = "wf_pz",
    norm_param: str = "pz_mean",
    cut_parameters: dict = {"bl_mean": 4, "bl_std": 4, "bl_slope": 4},
) -> dict:
    """
    This function calculates the pole zero constant for the input data

    Parameters
    ----------
    f_raw : str
        The raw file to run the macro on
    dsp_config: str
        Path to the dsp config file, this is a stripped down version which just includes cuts and slope of decay tail
    channel:  str
        Name of channel to process, should be name of lh5 group in raw files

    Returns
    -------
    tau_dict : dict
    """

    tb_data = load_data(raw_files, lh5_path, wf_field=wf_field, threshold=threshold)
    tb_out = opt.run_one_dsp(tb_data, dsp_config)
    log.debug("Processed Data")
    cut_dict = cts.generate_cuts(tb_out, parameters=cut_parameters)
    log.debug("Generated Cuts:", cut_dict)
    idxs = cts.get_cut_indexes(tb_out, cut_dict)
    log.debug("Applied cuts")
    log.debug(f"{len(idxs)} events passed cuts")
    slopes = tb_out["tail_slope"].nda
    log.debug("Calculating pz constant")
    if display > 0:
        tau_dict, plot_dict = get_decay_constant(
            slopes[idxs], tb_data[wf_field], display=display
        )
    else:
        tau_dict = get_decay_constant(slopes[idxs], tb_data[wf_field])
    if double_pz == True:
        log.debug("Calculating double pz constants")
        pspace = om.set_par_space(opt_dict)
        grid_out = opt.run_grid(
            tb_data, dsp_config, pspace, fom_dpz, tau_dict, fom_kwargs=None
        )
        out_dict = get_dpz_consts(grid_out, opt_dict)
        tau_dict["pz"].update(out_dict["pz"])
    if display > 0:
        tb_out = opt.run_one_dsp(tb_data, dsp_config, db_dict=tau_dict)
        wfs = tb_out[wf_plot]["values"].nda[idxs]
        wf_idxs = np.random.choice(len(wfs), 100)
        if norm_param is not None:
            means = tb_out[norm_param].nda[idxs][wf_idxs]
            wfs = np.divide(wfs[wf_idxs], np.reshape(means, (len(wf_idxs), 1)))
        else:
            wfs = wfs[wf_idxs]
        fig2 = plt.figure()
        for wf in wfs:
            plt.plot(np.arange(0, len(wf), 1), wf)
        plt.axhline(1, color="black")
        plt.axhline(0, color="black")
        plt.xlabel("Samples")
        plt.ylabel("ADU")
        plot_dict["waveforms"] = fig2
        if display > 1:
            plt.show()
        else:
            plt.close()
        return tau_dict, plot_dict
    else:
        return tau_dict
