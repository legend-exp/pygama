"""
Helpers shared by the cross-talk (XTC) routines in :mod:`pygama.pargen.xtc`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import lgdo
import lh5
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.use("agg")

log = logging.getLogger(__name__)

#: Column each polarity is stored under in an xtc lh5 file.  The two names
#: are fixed by the production files the rest of LEGEND reads.
XTC_LH5_FIELD = {"neg": "xtalk_matrix_negative", "pos": "xtalk_matrix_positive"}

#: Colour-scale limits that make each matrix readable, in percent.
XTC_PLOT_RANGE = {"neg": (-0.3, 0.1), "pos": (-0.07, 0.3)}


class EventSelector:
    """Read one parameter from a set of LH5 files and apply event cuts to it.

    Reads *ene_dataset* (plus any flag fields named in *conditions*) from
    *table_path* across *files*, then keeps the events that are not NaN, that
    satisfy every flag condition, and that fall inside *energy_range*.

    The surviving row indices are exposed as :attr:`selected_idxs` so a
    selection made on one tier can be pushed into a selection on another: pass
    them as *idx* to restrict the second read to the same events.

    Parameters
    ----------
    table_path
        LH5 group holding the parameter, e.g. ``"ch1104000/hit/"``.
    files
        File, or list of files, to read.
    ene_dataset
        Name of the field the selection is applied to.
    conditions
        Mapping of flag field name to the value that field must equal for an
        event to be kept.  ``None`` applies no flag cut.
    energy_range
        ``(emin, emax)`` inclusive bounds on *ene_dataset*.  ``None`` applies
        no range cut.
    idx
        Row indices to restrict the read-back arrays to before cutting,
        typically the :attr:`selected_idxs` of an earlier selection on a
        different tier.  ``None`` considers every row.

    Attributes
    ----------
    energy_all
        Every value of *ene_dataset* in *files*, before any cut.
    energy_all_indexed
        ``energy_all[idx]``, or ``None`` when *idx* is ``None``.
    selected_energies
        The values surviving all cuts.
    selected_idxs
        Indices into :attr:`energy_all` of the surviving events.
    """

    def __init__(
        self,
        table_path: str,
        files: str | list,
        ene_dataset: str,
        conditions: dict | None = None,
        energy_range: tuple | None = None,
        idx: np.ndarray | None = None,
    ) -> None:
        flag_datasets = list(conditions.keys()) if conditions is not None else []
        all_fields = [ene_dataset, *flag_datasets]
        table = lh5.read(table_path, files, field_mask=all_fields)
        self.energy_all = table[ene_dataset].nda

        if idx is not None:
            self.energy_all_indexed = self.energy_all[idx]
            selection_array = ~np.isnan(self.energy_all_indexed)
        else:
            self.energy_all_indexed = None
            selection_array = ~np.isnan(self.energy_all)

        for flag in flag_datasets:
            flag_array = table[flag].nda[idx] if idx is not None else table[flag].nda
            condition = conditions.get(flag, True)
            selection_array &= flag_array == condition

        if energy_range is not None:
            emin, emax = energy_range
            energies = self.energy_all_indexed if idx is not None else self.energy_all
            selection_array &= (energies >= emin) & (energies <= emax)

        if idx is not None:
            self.selected_energies = self.energy_all_indexed[selection_array]
            self.selected_idxs = idx[selection_array]
        else:
            self.selected_energies = self.energy_all[selection_array]
            self.selected_idxs = np.arange(len(self.energy_all))[selection_array]

    def draw(
        self,
        fig_path: str | Path,
        y_scale: str = "log",
        plot_range: tuple | None = None,
        bins_count: int = 100,
    ) -> None:
        """Save a before/after histogram of the selection to *fig_path*.

        Parameters
        ----------
        fig_path
            File to write the figure to.
        y_scale
            Scale of the y axis, passed to :func:`matplotlib.pyplot.yscale`.
        plot_range
            ``(xmin, xmax)`` to restrict the histogram to.  ``None`` uses the
            full range of :attr:`energy_all`.
        bins_count
            Number of bin edges spanning the plot range.
        """
        plt.figure(figsize=(10, 6))

        if plot_range is not None:
            xmin, xmax = plot_range
            # Filtering
            data_all = self.energy_all[
                (self.energy_all >= xmin) & (self.energy_all <= xmax)
            ]
            data_sel = self.selected_energies[
                (self.selected_energies >= xmin) & (self.selected_energies <= xmax)
            ]
        else:
            xmin, xmax = np.min(self.energy_all), np.max(self.energy_all)
            data_all = self.energy_all
            data_sel = self.selected_energies

        # Calculate uniform bins strictly within this range
        bins = np.linspace(xmin, xmax, bins_count)

        # Plot the histograms using the filtered data and common bins
        plt.hist(
            data_all,
            bins=bins,
            alpha=0.5,
            label=f"Pre-selection({len(self.energy_all)})",
        )
        plt.hist(
            data_sel,
            bins=bins,
            alpha=0.5,
            label=f"Selected({len(self.selected_energies)})",
        )

        # Set the x-axis limits to match the requested range
        plt.xlim(xmin, xmax)

        plt.xlabel("Energy")
        plt.ylabel("Counts")
        plt.yscale(y_scale)
        plt.title("Energy Distribution Before and After Selection")
        plt.legend()
        plt.savefig(fig_path)
        plt.close()


def xtalk_element(
    e_trig: np.ndarray | float,
    e_response: np.ndarray | float,
    baseline_value: float,
) -> np.ndarray | float:
    """Cross-talk of a response channel relative to its trigger, in percent.

    The response amplitude has its baseline removed and is expressed as a
    fraction of the trigger energy::

        (e_response - baseline_value) / e_trig * 100

    Parameters
    ----------
    e_trig
        Trigger-channel energy.  Array, or a single value.
    e_response
        Response-channel amplitude at the same events.  Must match the type
        and length of *e_trig*.
    baseline_value
        Baseline of the response channel, subtracted from *e_response*.

    Returns
    -------
    Percentage cross-talk, elementwise when the inputs are arrays.
    """
    if not isinstance(baseline_value, (int, float, np.integer, np.floating)):
        msg = "baseline_value must be a numerical type (int or float)."
        raise TypeError(msg)

    if isinstance(e_trig, np.ndarray) and isinstance(e_response, np.ndarray):
        if len(e_trig) != len(e_response):
            msg = "e_trig and e_response must have the same length."
            raise ValueError(msg)
        return (e_response - baseline_value) / e_trig * 100

    if isinstance(e_trig, (int, float, np.integer, np.floating)) and isinstance(
        e_response, (int, float, np.integer, np.floating)
    ):
        return (e_response - baseline_value) / e_trig * 100

    msg = (
        "e_trig and e_response must either both be arrays of equal length "
        "or both be numerical values."
    )
    raise TypeError(msg)


class XTCMatrix:
    """The cross-talk matrices of one set of detectors, both polarities.

    Element ``[j1, j2]`` is the cross-talk seen in detector ``rawids[j2]``
    when detector ``rawids[j1]`` triggered, so rows are triggers and columns
    are responses.  The diagonal, and any pair that was never fitted, is NaN.

    Values are held **in percent**, the unit
    :func:`~pygama.pargen.xtc_utils.xtalk_element` produces, while the lh5
    file stores fractions the way the production xtc files do.
    :meth:`write_lh5` and :meth:`read_lh5` do that conversion for you; pass
    ``in_percent=False`` to either if you are working in fractions already.

    Parameters
    ----------
    rawids
        Detector ids in matrix-index order: ``rawids[j]`` is the detector at
        row and column ``j``.  Stored in the file as ``rawid_index``.
    mu, sigma
        ``{"neg": (N, N), "pos": (N, N)}`` of fitted peak positions and
        widths.  A missing polarity is filled with NaN.
    status
        ``{"neg": (N, N), "pos": (N, N)}`` of fit status codes, which is what
        tells a NaN that was never measured from one whose fit failed.  A
        missing polarity is filled with -1, "not recorded".
    status_codes
        Mapping of status name to the code used in *status*, carried into the
        file so a reader never has to hard-code the numbers.
    """

    polarities = ("neg", "pos")

    def __init__(
        self,
        rawids,
        mu: dict | None = None,
        sigma: dict | None = None,
        status: dict | None = None,
        status_codes: dict | None = None,
    ):
        self.rawids = np.asarray(rawids, dtype=np.int64)
        if self.rawids.ndim != 1:
            msg = f"rawids must be one-dimensional, got shape {self.rawids.shape}"
            raise ValueError(msg)

        self.status_codes = dict(status_codes) if status_codes else {}
        self.mu = self._as_matrices(mu, np.nan, float)
        self.sigma = self._as_matrices(sigma, np.nan, float)
        self.status = self._as_matrices(status, -1, np.int8)

    def _as_matrices(self, given: dict | None, fill, dtype) -> dict:
        """One ``(N, N)`` array per polarity, filled where none was given."""
        given = given or {}
        shape = (self.n_detectors, self.n_detectors)
        matrices = {}
        for polarity in self.polarities:
            value = given.get(polarity)
            if value is None:
                matrices[polarity] = np.full(shape, fill, dtype=dtype)
                continue
            value = np.asarray(value, dtype=dtype)
            if value.shape != shape:
                msg = (
                    f"{polarity} matrix has shape {value.shape}, but "
                    f"{len(self.rawids)} detectors need {shape}"
                )
                raise ValueError(msg)
            matrices[polarity] = value
        return matrices

    @property
    def n_detectors(self) -> int:
        """Detectors along each side of the matrix."""
        return len(self.rawids)

    def index_of(self, rawid: int) -> int:
        """Matrix index of *rawid*."""
        (found,) = np.nonzero(self.rawids == int(rawid))
        if len(found) != 1:
            msg = f"rawid {rawid} appears {len(found)} times in rawids"
            raise ValueError(msg)
        return int(found[0])

    def write_lh5(
        self,
        out_path: str | Path,
        group: str = "xtc",
        in_percent: bool = True,
    ) -> None:
        """Write both matrices to *out_path* as one lh5 table.

        The table carries ``rawid_index`` and the two production field names,
        plus a ``_sigma`` and a ``_status`` column for each -- extras the
        production files do not have, which a reader that wants only the two
        matrices simply ignores.

        Parameters
        ----------
        out_path
            File to write; parent directories are created and an existing
            file is replaced.
        group
            Table name inside the file.  Default ``"xtc"``, which is where
            the rest of LEGEND looks for it.
        in_percent
            Whether this object's values are in percent, in which case they
            are divided by 100 on the way out so the file holds fractions.
        """
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        scale = 0.01 if in_percent else 1.0
        col_dict = {"rawid_index": lgdo.Array(self.rawids)}
        for polarity in self.polarities:
            field = XTC_LH5_FIELD[polarity]
            col_dict[field] = lgdo.Array(self.mu[polarity] * scale)
            col_dict[f"{field}_sigma"] = lgdo.Array(self.sigma[polarity] * scale)
            col_dict[f"{field}_status"] = lgdo.Array(self.status[polarity])

        attrs = {}
        if self.status_codes:
            attrs["fit_status_codes"] = json.dumps(self.status_codes)

        if out_path.exists():
            log.info("replacing the existing %s", out_path)

        lh5.write(
            lgdo.Table(col_dict=col_dict, attrs=attrs),
            name=group,
            lh5_file=out_path,
            wo_mode="overwrite_file",
            compression="gzip",
        )
        log.info(
            "%sx%s cross-talk matrix written to %s:/%s",
            self.n_detectors,
            self.n_detectors,
            out_path,
            group,
        )

    @classmethod
    def read_lh5(
        cls,
        in_path: str | Path,
        group: str = "xtc",
        in_percent: bool = True,
    ) -> XTCMatrix:
        """Read a matrix back from an xtc lh5 file.

        Reads the production files as well as the ones :meth:`write_lh5`
        produces: the ``_sigma`` and ``_status`` columns are optional and
        come back filled with NaN and -1 when the file has none.

        Parameters
        ----------
        in_path
            File to read.
        group
            Table name inside the file.
        in_percent
            Whether the returned object should be in percent, in which case
            the file's fractions are multiplied by 100 on the way in.
        """
        table = lh5.read(group, Path(in_path))
        scale = 100.0 if in_percent else 1.0

        mu, sigma, status = {}, {}, {}
        for polarity in cls.polarities:
            field = XTC_LH5_FIELD[polarity]
            if field not in table.keys():
                continue
            mu[polarity] = table[field].nda * scale
            if f"{field}_sigma" in table.keys():
                sigma[polarity] = table[f"{field}_sigma"].nda * scale
            if f"{field}_status" in table.keys():
                status[polarity] = table[f"{field}_status"].nda

        codes = table.attrs.get("fit_status_codes")
        return cls(
            table["rawid_index"].nda,
            mu=mu,
            sigma=sigma,
            status=status,
            status_codes=json.loads(codes) if codes else None,
        )

    def plot(
        self,
        polarity: str,
        fig_path: str | Path,
        vmin: float | None = None,
        vmax: float | None = None,
        cmap=None,
        title: str | None = None,
    ) -> None:
        """Save a heatmap of one polarity's matrix to *fig_path*.

        Parameters
        ----------
        polarity
            ``"neg"`` or ``"pos"``.
        fig_path
            File to write the figure to.
        vmin, vmax
            Colour-scale limits, in the unit the matrix is held in.  ``None``
            takes the polarity's entry in :data:`XTC_PLOT_RANGE`.
        cmap
            Colormap, defaulting to reversed jet as in the original analysis.
        title
            Figure title.  ``None`` names the polarity.
        """
        if polarity not in self.polarities:
            msg = f"polarity must be one of {self.polarities}, got {polarity!r}"
            raise ValueError(msg)

        default_vmin, default_vmax = XTC_PLOT_RANGE[polarity]

        plt.figure(figsize=(8, 6))
        image = plt.imshow(
            self.mu[polarity],
            origin="lower",
            vmin=default_vmin if vmin is None else vmin,
            vmax=default_vmax if vmax is None else vmax,
            cmap=plt.cm.jet_r if cmap is None else cmap,
        )
        plt.colorbar(image, label="Cross-talk (%)")
        plt.xlabel("Response channel index")
        plt.ylabel("Trigger channel index")
        plt.title(title if title is not None else f"{polarity} cross-talk matrix")
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close()

    def summary(self) -> dict:
        """Count the matrix elements each fit status produced, per polarity.

        Replaces the per-pair bookkeeping the original analysis kept in a
        ``fail_dict``: the status matrices hold that information element by
        element, so all that is left to do is count it.
        """
        code_name = {code: name for name, code in self.status_codes.items()}
        counts = {}
        for polarity in self.polarities:
            values, totals = np.unique(self.status[polarity], return_counts=True)
            counts[polarity] = {
                code_name.get(int(v), f"code_{int(v)}"): int(t)
                for v, t in zip(values, totals)
            }
        return counts
