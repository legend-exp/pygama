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
import matplotlib.pyplot as plt
import numpy as np

import pygama
import pygama.lgdo as lgdo
import pygama.lgdo.lh5_store as lh5
import pygama.math.histogram as pgh
import pygama.pargen.cuts as cts
from pygama.pargen.dsp_optimize import run_one_dsp

log = logging.getLogger(__name__)


def run_tau(
    raw_file: list[str],
    config: dict,
    lh5_path: str,
    n_events: int = 30000,
    threshold: int = 3000,
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
        ids = np.zeros(len(df.daqenergy.values), dtype=bool)

    cuts = np.where((df.daqenergy.values > threshold) & (ids))[0]

    waveforms = sto.read_object(
        f"{lh5_path}/waveform", raw_file, idx=cuts, n_rows=n_events
    )[0]
    baseline = sto.read_object(
        f"{lh5_path}/baseline", raw_file, idx=cuts, n_rows=n_events
    )[0]
    tb_data = lh5.Table(col_dict={"waveform": waveforms, "baseline": baseline})
    return run_one_dsp(tb_data, config)


def get_decay_constant(slopes: np.array, wfs: np.array, plot_path: str = None) -> dict:

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

    counts, bins, var = pgh.get_hist(slopes, bins=50000, range=(-0.01, 0))
    bin_centres = pgh.get_bin_centers(bins)
    tau = round(-1 / (bin_centres[np.argmax(counts)]), 1)

    tau_dict["pz"] = {"tau": tau}
    if plot_path is None:
        return tau_dict
    else:
        out_plot_dict = {}
        pathlib.Path(os.path.dirname(plot_path)).mkdir(parents=True, exist_ok=True)
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
        plt.close()

        wf_idxs = np.random.choice(len(wfs), 100)
        fig2 = plt.figure()
        for wf_idx in wf_idxs:
            plt.plot(np.arange(0, len(wfs[wf_idx]), 1), wfs[wf_idx])
        plt.xlabel("Samples")
        plt.ylabel("ADU")
        out_plot_dict["waveforms"] = fig2
        with open(plot_path, "wb") as f:
            pkl.dump(out_plot_dict, f)
        plt.close()
        return tau_dict


def dsp_preprocess_decay_const(
    raw_files: list[str], dsp_config: dict, lh5_path: str, plot_path: str = None
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

    tb_out = run_tau(raw_files, dsp_config, lh5_path)
    log.debug("Processed Data")
    cut_dict = cts.generate_cuts(
        tb_out, parameters={"bl_mean": 4, "bl_std": 4, "bl_slope": 4}
    )
    log.debug("Generated Cuts:", cut_dict)
    idxs = cts.get_cut_indexes(tb_out, cut_dict)
    log.debug("Applied cuts")
    slopes = tb_out["tail_slope"].nda
    wfs = tb_out["wf_blsub"]["values"].nda
    log.debug("Calculating pz constant")
    tau_dict = get_decay_constant(slopes[idxs], wfs[idxs], plot_path=plot_path)
    return tau_dict
