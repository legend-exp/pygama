"""Routines to evaluate the correlation between HPGe and SiPM signals."""
from __future__ import annotations

import awkward as ak
import numpy as np
import scipy
from numpy.typing import ArrayLike


def l200_combined_test_stat(
    t0: ak.Array,
    amp: ak.Array,
    geds_t0: ak.Array,
) -> ak.Array:
    """Combined L200 LAr veto classifier.

    Where combined means taking channel-specific parameters into account.

    `t0` and `amp` must be in the format of a 3-dimensional Awkward array,
    where the innermost dimension labels the SiPM pulse, the second one labels
    the SiPM channel and the outermost one labels the event.

    Parameters
    ----------
    t0
        arrival times of pulses in ns, split by channel.
    amp
        amplitude of pulses in p.e., split by channel.
    geds_t0
        t0 (ns) of the HPGe signal.
    """
    # flatten the data in the last axis (i.e. merge all channels together)
    # TODO: implement channel distinction
    t0 = ak.flatten(t0, axis=-1)
    amp = ak.flatten(amp, axis=-1)

    # subtract the HPGe t0 from the SiPM pulse t0s
    # HACK: remove 16 when units will be fixed
    rel_t0 = 16 * t0 - geds_t0

    return l200_test_stat(rel_t0, amp)


def l200_test_stat(relative_t0, amp):
    """Compute the test statistics.

    Parameters
    ----------
    relative_t0
        t0 (ns) of the SiPM pulses relative to the HPGe t0.
    amp
        amplitude in p.e. of the SiPM pulses.
    """
    return -ak.sum(ak.transform(_ak_l200_test_stat_terms, relative_t0, amp), axis=-1)


# need to define this function and use it with ak.transform() because scipy
# routines are not NumPy universal functions
def _ak_l200_test_stat_terms(layouts, **kwargs):
    """Awkward transform to compute the per-pulse terms of the test statistics.

    The two arguments are the pulse times `t0` and their amplitude `amp`. The
    function has to be invoked as ``ak.transform(_ak_l200_test_stat_terms, t0, amp,
    ...)``.
    """
    # sanity check
    assert len(layouts) == 2

    if not all([lay.is_numpy for lay in layouts]):
        return

    # these are the two supported arguments
    t0 = layouts[0].data
    amp = layouts[1].data

    # sanity check
    assert len(t0) == len(amp)

    # if there are no pulses return NaN
    if len(t0) == 0 or any(np.isnan(t0)):
        return ak.contents.NumpyArray([np.nan])

    # convert to integer number of pes
    n_pes = pulse_amp_round(amp)
    n_pe_tot = np.sum(n_pes)

    t_stat = n_pes * np.log(l200_tc_time_pdf(t0)) / n_pe_tot + np.log(
        l200_rc_amp_pdf(n_pe_tot)
    )

    return ak.contents.NumpyArray(t_stat)


def pulse_amp_round(amp: float | ArrayLike):
    """Get the most likely (integer) number of photo-electrons."""
    # promote all amps < 1 to 1. standard rounding to nearest for
    # amps > 1
    return ak.where(amp < 1, np.ceil(amp), np.floor(amp + 0.5))


def l200_tc_time_pdf(
    t0: float | ArrayLike,
    *,
    tau_singlet_ns: float = 6,
    tau_triplet_ns: float = 1100,
    sing2trip_ratio: float = 1 / 3,
    t0_res_ns: float = 35,
    t0_bias_ns: float = -80,
    bkg_prob: float = 0.42,
) -> float | ArrayLike:
    """The L200 experimental LAr scintillation pdf

    The theoretical scintillation pdf convoluted with a Normal distribution
    (experimental effects) and summed to a uniform distribution (uncorrelated
    pulses).

    Parameters
    ----------
    t0
        arrival times of the SiPM pulses in ns.
    tau_singlet_ns
        The lifetime of the LAr singlet state in ns.
    tau_triplet_ns
        The lifetime of the LAr triplet state in ns.
    sing2trip_ratio
        The singlet-to-triplet excitation probability ratio.
    t0_res_ns
        sigma (ns) of the normal distribution.
    t0_bias_ns
        mean (ns) of the normal distribution.
    bkg_prob
        probability for a pulse coming from some uncorrelated physics (uniform
        distribution).
    """
    return (
        # the triplet
        (1 - sing2trip_ratio)
        * scipy.stats.exponnorm.pdf(
            t0, tau_triplet_ns / t0_res_ns, loc=t0_bias_ns, scale=t0_res_ns
        )
        # the singlet
        + sing2trip_ratio
        * scipy.stats.exponnorm.pdf(
            t0, tau_singlet_ns / t0_res_ns, loc=t0_bias_ns, scale=t0_res_ns
        )
        # the random coincidences (uniform pdf for normalization)
        + bkg_prob * scipy.stats.uniform.pdf(t0, -1_000, 5_000)
    )


def l200_rc_amp_pdf(n):
    return np.exp(-n)
