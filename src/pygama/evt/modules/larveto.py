"""Routines to evaluate the correlation between HPGe and SiPM signals."""

from __future__ import annotations

from collections.abc import Sequence

import awkward as ak
import numpy as np
import scipy
from numpy.typing import ArrayLike


def l200_combined_test_stat(
    t0: ak.Array,
    amp: ak.Array,
    geds_t0: ak.Array,
    bkg_prob: float,
    dens_array: Sequence[float],
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
    bkg_prob
        probability for a pulse coming from some uncorrelated physics (uniform
        distribution). needed for the LAr scintillation time pdf.
    dens_array
        array of densities (probabilities) of uncorrelated number of photoelectrons in a 6µs window.
    """
    # flatten the data in the last axis (i.e. merge all channels together)
    # TODO: implement channel distinction
    t0 = ak.flatten(t0, axis=-1)
    amp = ak.flatten(amp, axis=-1)

    # subtract the HPGe t0 from the SiPM pulse t0s
    # HACK: remove 16 when units will be fixed
    rel_t0 = 16 * t0 - geds_t0

    return l200_test_stat(rel_t0, amp, bkg_prob, dens_array)


def l200_test_stat(relative_t0, amp, bkg_prob, dens_array):
    """Compute the test statistics.

    Parameters
    ----------
    relative_t0
        t0 (ns) of the SiPM pulses relative to the HPGe t0.
    amp
        amplitude in p.e. of the SiPM pulses.
    """
    # convert to integer number of pes
    n_pes = pulse_amp_round(amp)

    # compute total number of pes in event
    n_pe_tot = np.sum(n_pes, axis=-1)
    # if no pes in the event, use nan instead of zero (because of division later)
    n_pe_tot = np.where(n_pe_tot == 0, np.nan, n_pe_tot)

    # calculate the test statistic term related to the time distribution
    transform_function = transform_wrapper(bkg_prob)
    ts_time = -ak.sum(ak.transform(transform_function, relative_t0, amp), axis=-1)
    # calculate the amplitude contribution
    ts_amp = [l200_rc_amp_logpdf(n, dens_array) for n in n_pe_tot]

    # for events with no light, set the test statistic value to +inf
    t_stat = np.where(np.isnan(n_pe_tot), np.inf, ts_time / n_pe_tot + ts_amp)

    return t_stat


# need this to pass bkg_prob parameter with ak.transform()
def transform_wrapper(bkg_prob):
    def _transform(layouts, **kwargs):
        return _ak_l200_test_stat_time_term(layouts, bkg_prob=bkg_prob)

    return _transform


# need to define this function and use it with ak.transform() because scipy
# routines are not NumPy universal functions
def _ak_l200_test_stat_time_term(layouts, bkg_prob, **kwargs):
    """Awkward transform to compute the per-pulse terms of the test statistics.

    The two arguments are the pulse times `t0` relative to the HPGe trigger and
    their amplitude `amp`. The function has to be invoked as
    ``ak.transform(_ak_l200_test_stat_terms, t0, amp,
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

    # convert to integer number of pes
    n_pes = pulse_amp_round(amp)

    # calculate the time term of the test statistic
    ts_time = n_pes * np.log(l200_tc_time_pdf(t0, bkg_prob=bkg_prob))

    return ak.contents.NumpyArray(ts_time)


def pulse_amp_round(amp: float | ArrayLike):
    """Get the most likely (integer) number of photo-electrons."""
    # promote all amps < 1 to 1. standard rounding to nearest for amps > 1
    return ak.where(amp < 1, np.ceil(amp), np.floor(amp + 0.5))


def l200_tc_time_pdf(
    t0: float | ArrayLike,
    *,
    domain_ns: tuple[float] = (-1_000, 5_000),
    tau_singlet_ns: float = 6,
    tau_triplet_ns: float = 1100,
    sing2tot_ratio: float = 1 / 3,
    t0_res_ns: float = 35,
    t0_bias_ns: float = -80,
    bkg_prob: float = 0.42,
) -> float | ArrayLike:
    """The L200 experimental LAr scintillation pdf

    The theoretical scintillation pdf convoluted with a Normal distribution
    (experimental effects) and summed to a uniform distribution (uncorrelated
    pulses).

    This routine does not work with :class:`ak.Array`, since SciPy functions
    are not universal. See :func:`_ak_l200_test_stat_time_term` for an example
    Awkward transform that does the job.

    Parameters
    ----------
    t0
        arrival times of the SiPM pulses in ns.
    tau_singlet_ns
        The lifetime of the LAr singlet state in ns.
    tau_triplet_ns
        The lifetime of the LAr triplet state in ns.
    sing2tot_ratio
        The singlet-to-total excitation probability ratio.
    t0_res_ns
        sigma (ns) of the normal distribution.
    t0_bias_ns
        mean (ns) of the normal distribution.
    bkg_prob
        probability for a pulse coming from some uncorrelated physics (uniform
        distribution).
    """
    if np.any(t0 > domain_ns[1]) or np.any(t0 < domain_ns[0]):
        msg = f"{t0=} out of bounds for {domain_ns=}"
        raise ValueError(msg)

    return (
        # the triplet
        (1 - bkg_prob)
        * (
            (1 - sing2tot_ratio)
            * scipy.stats.exponnorm.pdf(
                t0, tau_triplet_ns / t0_res_ns, loc=t0_bias_ns, scale=t0_res_ns
            )
            # the singlet
            + sing2tot_ratio
            * scipy.stats.exponnorm.pdf(
                t0, tau_singlet_ns / t0_res_ns, loc=t0_bias_ns, scale=t0_res_ns
            )
        )
        # the random coincidences (uniform pdf)
        + bkg_prob
        * scipy.stats.uniform.pdf(t0, domain_ns[0], domain_ns[1] - domain_ns[0])
    )


def l200_rc_amp_logpdf(
    n, dens_array: Sequence[float] | None = None, slope: float = -1 / 15
):
    """The L200 experimental random coincidence (RC) amplitude pdf

    Parameters
    ----------
    n
        number of photoelectrons.
    dens_array
        density array of RC LAr energy histogram (total energy summed over all channels, in p.e.).
        derived from forced trigger data.
    slope
        slope for analytical continuation.

    """
    # analytical continuation must be decaying exponential function.
    assert slope < 0

    if dens_array is None:
        dens_array = [1]

    # up to 15 p.e., take the experimental values from the density.
    limit = min(len(dens_array) - 1, 15)
    if n <= limit:
        return np.log(dens_array[int(n)])
    # analytical continuation: log of an exponential function.
    else:
        return slope * (n - int(limit)) + np.log(dens_array[int(limit)])
