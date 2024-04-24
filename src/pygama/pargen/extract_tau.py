"""
This module is for extracting a single pole zero constant from the decay tail
"""

from __future__ import annotations

import logging

import lgdo
import lgdo.lh5 as lh5
import matplotlib.pyplot as plt
import numpy as np

import pygama.math.binned_fitting as pgf
import pygama.math.histogram as pgh
import pygama.pargen.dsp_optimize as opt
import pygama.pargen.energy_optimisation as om
from pygama.pargen.data_cleaning import get_mode_stdev

log = logging.getLogger(__name__)
sto = lh5.LH5Store()


class ExtractTau:
    def __init__(self, dsp_config, wf_field, debug_mode=False):
        self.dsp_config = dsp_config
        self.wf_field = wf_field
        self.output_dict = {}
        self.results_dict = {}
        self.debug_mode = debug_mode

    def get_decay_constant(
        self, slopes: np.array, wfs: lgdo.WaveformTable, display: int = 0
    ) -> dict:
        """
        Finds the decay constant from the modal value of the tail slope after cuts
        and saves it to the specified json. Updates self.output_dict with tau value

        Parameters
        ----------
        - slopes: numpy array of tail slopes
        - wfs: WaveformTable object containing waveform data
        - display: integer indicating the level of display (0: no display, 1: plot histogram, 2: show histogram)

        Returns
        -------
        - out_plot_dict: dictionary containing the plot figure (only returned if display > 0)
        """

        mode, stdev = get_mode_stdev(slopes)
        tau = round(-1 / (mode), 1)
        err = round((-1 / (mode + (stdev / np.sqrt(len(slopes))))) - tau, 1)

        sampling_rate = wfs["dt"].nda[0]
        units = wfs["dt"].attrs["units"]
        tau = f"{tau*sampling_rate}*{units}"

        if "pz" in self.output_dict:
            self.output_dict["pz"].update({"tau": tau, "tau_err": err})
        else:
            self.output_dict["pz"] = {"tau": tau, "tau_err": err}

        self.results_dict.update(
            {"single_decay_constant": {"slope_pars": {"mode": mode, "stdev": stdev}}}
        )
        if display <= 0:
            return
        else:
            out_plot_dict = {}

            return out_plot_dict

    def get_dpz_consts(self, grid_out, opt_dict):
        std_grid = np.ndarray(shape=grid_out.shape)
        for i in range(grid_out.shape[0]):
            for j in range(grid_out.shape[1]):
                std_grid[i, j] = grid_out[i, j]["y_val"]
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
                except BaseException as e:
                    if e == KeyboardInterrupt:
                        raise (e)
                    elif self.debug_mode:
                        raise (e)
                    db_dict[opt_name] = {
                        key: f"{param_list[i][min_point[i]][0]}*{unit}"
                    }
            else:
                try:
                    db_dict[opt_name].update({key: f"{param_list[i][min_point[i]][0]}"})
                except BaseException as e:
                    if e == KeyboardInterrupt:
                        raise (e)
                    elif self.debug_mode:
                        raise (e)
                    db_dict[opt_name] = {key: f"{param_list[i][min_point[i]][0]}"}
        return db_dict

    def calculate_dpz(self, tb_data, opt_dict):
        log.debug("Calculating double pz constants")
        pspace = om.set_par_space(opt_dict)
        grid_out = opt.run_grid(
            tb_data, self.dsp_config, pspace, fom_dpz, self.output_dict, fom_kwargs=None
        )
        out_dict = self.get_dpz_consts(grid_out, opt_dict)
        if "pz" in self.output_dict:
            self.output_dict["pz"].update(out_dict["pz"])
        else:
            self.output_dict["pz"] = out_dict["pz"]

    def plot_waveforms_after_correction(
        self, tb_data, wf_field, norm_param=None, display=0
    ):
        tb_out = opt.run_one_dsp(tb_data, self.dsp_config, db_dict=self.output_dict)
        wfs = tb_out[wf_field]["values"].nda
        wf_idxs = np.random.choice(len(wfs), 100)
        if norm_param is not None:
            means = tb_out[norm_param].nda[wf_idxs]
            wfs = np.divide(wfs[wf_idxs], np.reshape(means, (len(wf_idxs), 1)))
        else:
            wfs = wfs[wf_idxs]
        fig = plt.figure()
        for wf in wfs:
            plt.plot(np.arange(0, len(wf), 1), wf)
        plt.axhline(1, color="black")
        plt.axhline(0, color="black")
        plt.xlabel("Samples")
        plt.ylabel("ADU")
        plot_dict = {"waveforms": fig}
        if display > 1:
            plt.show()
        else:
            plt.close()
        return plot_dict

    def plot_slopes(self, slopes, display=0):
        high_bin = self.results_dict["single_decay_constant"]["slope_pars"]["mode"]
        sigma = self.results_dict["single_decay_constant"]["slope_pars"]["stdev"]
        plt.rcParams["figure.figsize"] = (10, 6)
        plt.rcParams["font.size"] = 8
        fig, ax = plt.subplots()
        bins = np.arange(
            np.nanpercentile(slopes, 1),
            np.nanpercentile(slopes, 99),
            np.nanpercentile(slopes, 51) - np.nanpercentile(slopes, 50),
        )
        counts, bins, bars = ax.hist(slopes, bins=bins, histtype="step")
        ax.axvline(high_bin, color="red")
        in_min = high_bin - 4 * sigma
        in_max = high_bin + 4 * sigma
        plt.xlabel("Slope")
        plt.ylabel("Counts")
        axins = ax.inset_axes([0.6, 0.6, 0.4, 0.4])
        axins.hist(
            slopes[(slopes > in_min) & (slopes < in_max)],
            bins=50,
            histtype="step",
        )
        axins.axvline(high_bin, color="red")
        axins.set_xlim(in_min, in_max)
        ax.set_xlim(np.nanpercentile(slopes, 1), np.nanpercentile(slopes, 99))
        out_plot_dict = {"slope": fig}
        if display > 1:
            plt.show()
        else:
            plt.close()
        return out_plot_dict


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

    except Exception:
        mu = start_bins[max_idx]

    return {"y_val": np.abs(mu)}
