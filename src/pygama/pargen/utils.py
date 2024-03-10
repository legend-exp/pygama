from __future__ import annotations

import logging
from types import FunctionType

import numpy as np
import pandas as pd
from iminuit import Minuit, cost
from lgdo import lh5

log = logging.getLogger(__name__)
sto = lh5.LH5Store()


def convert_to_minuit(pars, func):
    c = cost.UnbinnedNLL(np.array([0]), func.get_pdf)
    if isinstance(pars, dict):
        m = Minuit(c, **pars)
    else:
        m = Minuit(c, *pars)
    return m


def return_nans(input):
    if isinstance(input, FunctionType):
        args = input.__code__.co_varnames[: input.__code__.co_argcount][1:]
        m = convert_to_minuit(np.full(len(args), np.nan), input)
        return m.values, m.errors, np.full((len(m.values), len(m.values)), np.nan)
    else:
        args = input.required_args()
        m = convert_to_minuit(np.full(len(args), np.nan), input)
        return m.values, m.errors, np.full((len(m.values), len(m.values)), np.nan)


def get_params(file_params, param_list):
    out_params = []
    if isinstance(file_params, dict):
        possible_keys = file_params.keys()
    elif isinstance(file_params, list):
        possible_keys = file_params
    for param in param_list:
        for key in possible_keys:
            if key in param:
                out_params.append(key)
    return np.unique(out_params).tolist()


# do these all belong in dataflow?

def load_data(
    files: list,
    lh5_path: str,
    cal_dict: dict,
    params: list,
    cal_energy_param: str = "cuspEmax_ctc_cal",
    threshold=None,
    return_selection_mask=False,
) -> tuple(np.array, np.array, np.array, np.array):
    """
    Loads in the A/E parameters needed and applies calibration constants to energy
    """

    if isinstance(files, dict):
        keys = lh5.ls(
            files[list(files)[0]][0],
            lh5_path if lh5_path[-1] == "/" else lh5_path + "/",
        )
        keys = [key.split("/")[-1] for key in keys]
        if list(files)[0] in cal_dict:
            params = get_params(keys + list(cal_dict[list(files)[0]].keys()), params)
        else:
            params = get_params(keys + list(cal_dict.keys()), params)

        df = []
        all_files = []
        masks = np.array([], dtype=bool)
        for tstamp, tfiles in files.items():
            table = sto.read(lh5_path, tfiles)[0]

            file_df = pd.DataFrame(columns=params)
            if tstamp in cal_dict:
                cal_dict_ts = cal_dict[tstamp]
            else:
                cal_dict_ts = cal_dict

            for outname, info in cal_dict_ts.items():
                outcol = table.eval(info["expression"], info.get("parameters", None))
                table.add_column(outname, outcol)

            for param in params:
                file_df[param] = table[param]

            file_df["run_timestamp"] = np.full(len(file_df), tstamp, dtype=object)

            if threshold is not None:
                mask = file_df[cal_energy_param] > threshold
                file_df.drop(np.where(~mask)[0], inplace=True)
            else:
                mask = np.ones(len(file_df), dtype=bool)
            masks = np.append(masks, mask)
            df.append(file_df)
            all_files += tfiles

        params.append("run_timestamp")
        df = pd.concat(df)

    elif isinstance(files, list):
        keys = lh5.ls(files[0], lh5_path if lh5_path[-1] == "/" else lh5_path + "/")
        keys = [key.split("/")[-1] for key in keys]
        params = get_params(keys + list(cal_dict.keys()), params)

        table = sto.read(lh5_path, files)[0]
        df = pd.DataFrame(columns=params)
        for outname, info in cal_dict.items():
            outcol = table.eval(info["expression"], info.get("parameters", None))
            table.add_column(outname, outcol)
        for param in params:
            df[param] = table[param]
        if threshold is not None:
            masks = df[cal_energy_param] > threshold
            df.drop(np.where(~masks)[0], inplace=True)
        else:
            masks = np.ones(len(df), dtype=bool)
        all_files = files

    for col in list(df.keys()):
        if col not in params:
            df.drop(col, inplace=True, axis=1)

    log.debug("data loaded")
    if return_selection_mask:
        return df, masks
    else:
        return df

def get_wf_indexes(sorted_indexs, n_events):
    out_list = []
    if isinstance(n_events, list):
        for i in range(len(n_events)):
            new_list = []
            for idx, entry in enumerate(sorted_indexs):
                if (entry >= np.sum(n_events[:i])) and (
                    entry < np.sum(n_events[: i + 1])
                ):
                    new_list.append(idx)
            out_list.append(new_list)
    else:
        for i in range(int(len(sorted_indexs) / n_events)):
            new_list = []
            for idx, entry in enumerate(sorted_indexs):
                if (entry >= i * n_events) and (entry < (i + 1) * n_events):
                    new_list.append(idx)
            out_list.append(new_list)
    return out_list


def index_data(data, indexes, wf_field="waveform"):
    new_baselines = lh5.Array(data["baseline"].nda[indexes])
    new_waveform_values = data[wf_field]["values"].nda[indexes]
    new_waveform_dts = data[wf_field]["dt"].nda[indexes]
    new_waveform_t0 = data[wf_field]["t0"].nda[indexes]
    new_waveform = lh5.WaveformTable(
        None, new_waveform_t0, "ns", new_waveform_dts, "ns", new_waveform_values
    )
    new_data = lh5.Table(col_dict={wf_field: new_waveform, "baseline": new_baselines})
    return new_data


def event_selection(
    raw_files,
    lh5_path,
    dsp_config,
    db_dict,
    peaks_kev,
    peak_idxs,
    kev_widths,
    cut_parameters=None,
    pulser_mask=None,
    energy_parameter="trapTmax",
    wf_field: str = "waveform",
    n_events=10000,
    threshold=1000,
    initial_energy="daqenergy",
    check_pulser=True,
):
    """
    Function for selecting events in peaks using raw files,
    to do this it uses the daqenergy to get a first rough selection
    then runs 1 dsp to get a more accurate energy estimate and apply cuts
    returns the indexes of the final events and the peak to which each index corresponds
    """

    if not isinstance(peak_idxs, list):
        peak_idxs = [peak_idxs]
    if not isinstance(kev_widths, list):
        kev_widths = [kev_widths]

    if lh5_path[-1] != "/":
        lh5_path += "/"

    raw_fields = [
        field.replace(lh5_path, "") for field in lh5.ls(raw_files[0], lh5_path)
    ]
    initial_fields = cts.get_keys(raw_fields, [initial_energy])
    initial_fields += ["timestamp"]

    df = lh5.read_as(lh5_path, raw_files, "pd", field_mask=initial_fields)
    df["initial_energy"] = df.eval(initial_energy)

    if pulser_mask is None and check_pulser is True:
        pulser_props = cts.find_pulser_properties(df, energy="initial_energy")
        if len(pulser_props) > 0:
            final_mask = None
            for entry in pulser_props:
                e_cut = (df.initial_energy.values < entry[0] + entry[1]) & (
                    df.initial_energy.values > entry[0] - entry[1]
                )
                if final_mask is None:
                    final_mask = e_cut
                else:
                    final_mask = final_mask | e_cut
            ids = final_mask
            log.debug(f"pulser found: {pulser_props}")
        else:
            log.debug("no_pulser")
            ids = np.zeros(len(df.initial_energy.values), dtype=bool)
        # Get events around peak using raw file values
    elif pulser_mask is not None:
        ids = pulser_mask
    else:
        ids = np.zeros(len(df.initial_energy.values), dtype=bool)

    initial_mask = (df["initial_energy"] > threshold) & (~ids)
    rough_energy = np.array(df["initial_energy"])[initial_mask]
    initial_idxs = np.where(initial_mask)[0]

    guess_kev = 2620 / np.nanpercentile(rough_energy, 99)
    euc_min = threshold / guess_kev * 0.6
    euc_max = 2620 / guess_kev * 1.1
    deuc = 1  # / guess_kev
    hist, bins, var = pgh.get_hist(rough_energy, range=(euc_min, euc_max), dx=deuc)
    detected_peaks_locs, detected_peaks_kev, roughpars = pgc.hpge_find_E_peaks(
        hist,
        bins,
        var,
        np.array([238.632, 583.191, 727.330, 860.564, 1620.5, 2103.53, 2614.553]),
    )
    log.debug(f"detected {detected_peaks_kev} keV peaks at {detected_peaks_locs}")

    masks = []
    for peak_idx in peak_idxs:
        peak = peaks_kev[peak_idx]
        kev_width = kev_widths[peak_idx]
        try:
            if peak not in detected_peaks_kev:
                raise ValueError
            detected_peak_idx = np.where(detected_peaks_kev == peak)[0]
            peak_loc = detected_peaks_locs[detected_peak_idx]
            log.info(f"{peak} peak found at {peak_loc}")
            rough_adc_to_kev = roughpars[0]
            e_lower_lim = peak_loc - (1.1 * kev_width[0]) / rough_adc_to_kev
            e_upper_lim = peak_loc + (1.1 * kev_width[1]) / rough_adc_to_kev
        except Exception:
            log.debug(f"{peak} peak not found attempting to use rough parameters")
            peak_loc = (peak - roughpars[1]) / roughpars[0]
            rough_adc_to_kev = roughpars[0]
            e_lower_lim = peak_loc - (1.5 * kev_width[0]) / rough_adc_to_kev
            e_upper_lim = peak_loc + (1.5 * kev_width[1]) / rough_adc_to_kev
        log.debug(f"lower_lim:{e_lower_lim}, upper_lim:{e_upper_lim}")
        e_mask = (rough_energy > e_lower_lim) & (rough_energy < e_upper_lim)
        e_idxs = initial_idxs[e_mask][: int(2.5 * n_events)]
        masks.append(e_idxs)
        log.debug(f"{len(e_idxs)} events found in energy range for {peak}")

    idx_list_lens = [len(masks[peak_idx]) for peak_idx in peak_idxs]

    sort_index = np.argsort(np.concatenate(masks))
    idx_list = get_wf_indexes(sort_index, idx_list_lens)
    idxs = np.array(sorted(np.concatenate(masks)))

    if len(idxs) == 0:
        raise ValueError("No events found in energy range")

    input_data = sto.read(f"{lh5_path}", raw_files, idx=idxs, n_rows=len(idxs))[0]

    if isinstance(dsp_config, str):
        with open(dsp_config) as r:
            dsp_config = json.load(r)

    dsp_config["outputs"] = cts.get_keys(
        dsp_config["outputs"], cut_parameters
    ) + [energy_parameter]

    log.debug("Processing data")
    tb_data = opt.run_one_dsp(input_data, dsp_config, db_dict=db_dict)

    ct_mask = np.full(len(tb_data), True, dtype=bool)
    if cut_parameters is not None:
        ct_mask = generate_cuts(tb_data, cut_parameters)
        log.debug("Cuts are calculated")

    final_events = []
    out_events = []
    for peak_idx in peak_idxs:
        peak = peaks_kev[peak_idx]
        kev_width = kev_widths[peak_idx]

        peak_ids = np.array(idx_list[peak_idx])
        peak_ct_mask = ct_mask[peak_ids]
        peak_ids = peak_ids[peak_ct_mask]

        energy = tb_data[energy_parameter].nda[peak_ids]

        hist, bins, var = pgh.get_hist(
            energy,
            range=(np.floor(np.nanmin(energy)), np.ceil(np.nanmax(energy))),
            dx=peak / (np.nanpercentile(energy, 50)),
        )
        peak_loc = pgh.get_bin_centers(bins)[np.nanargmax(hist)]

        mu, _, _ = pgc.hpge_fit_E_peak_tops(
            hist,
            bins,
            var,
            [peak_loc],
            n_to_fit=7,
        )[
            0
        ][0]

        if mu is None or np.isnan(mu):
            log.debug("Fit failed, using max guess")
            rough_adc_to_kev = peak / peak_loc
            e_lower_lim = peak_loc - (1.5 * kev_width[0]) / rough_adc_to_kev
            e_upper_lim = peak_loc + (1.5 * kev_width[1]) / rough_adc_to_kev
            hist, bins, var = pgh.get_hist(
                energy, range=(int(e_lower_lim), int(e_upper_lim)), dx=1
            )
            mu = pgh.get_bin_centers(bins)[np.nanargmax(hist)]

        updated_adc_to_kev = peak / mu
        e_lower_lim = mu - (kev_width[0]) / updated_adc_to_kev
        e_upper_lim = mu + (kev_width[1]) / updated_adc_to_kev
        log.info(f"lower lim is :{e_lower_lim}, upper lim is {e_upper_lim}")

        final_mask = (energy > e_lower_lim) & (energy < e_upper_lim)
        final_events.append(peak_ids[final_mask][:n_events])
        out_events.append(idxs[final_events[-1]])

        log.info(f"{len(peak_ids[final_mask][:n_events])} passed selections for {peak}")
        if len(peak_ids[final_mask]) < 0.5 * n_events:
            log.warning("Less than half number of specified events found")
        elif len(peak_ids[final_mask]) < 0.1 * n_events:
            log.error("Less than 10% number of specified events found")

    out_events = np.unique(np.concatenate(out_events))
    sort_index = np.argsort(np.concatenate(final_events))
    idx_list = get_wf_indexes(sort_index, [len(mask) for mask in final_events])
    return out_events, idx_list
