"""
Helpers shared by the cross-talk (XTC) routines in :mod:`pygama.pargen.xtc`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import lh5
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.use("agg")

log = logging.getLogger(__name__)


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
