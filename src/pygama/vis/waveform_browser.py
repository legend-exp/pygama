import glob
import itertools
import math
import os
import string
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler
from matplotlib.lines import Line2D

import pygama.lgdo.lh5_store as lh5
from pygama.dsp.processing_chain import build_processing_chain
from pygama.math.units import unit_registry as ureg


class WaveformBrowser:
    """
    The Waveform Browser is a tool meant for interacting with waveforms from
    LH5 files. This defines an interface for drawing waveforms from a file,
    drawing transformed waveforms defined using build_dsp style json files,
    drawing horizontal and vertical lines at the values of calculated
    parameters, and filling a legend with calculated parameters.
    """

    def __init__(self, files_in, lh5_group, base_path = '',
                 entry_list = None, entry_mask = None,
                 dsp_config = None, database = None,
                 aux_values = None,
                 lines = 'waveform', styles = None,
                 legend = None, legend_opts = None,
                 n_drawn = 1, x_unit = None, x_lim = None, y_lim = None,
                 norm = None, align=None,
                 buffer_len = 128, block_width = 8):
        """
        Parameters
        ----------
        files_in : str
            name of file or list of names to browse. Can use wildcards

        lh5_group : str
            name of LH5 group in file to browse

        base_path : str
            base path for file. See LH5Store

        entry_list : list-like or nested list-like (optional)
            List of event indices to draw. If it is a nested list, use local
            indices for each file, otherwise use global indices

        entry_mask : array-like or list of array-likes (optional)
            Boolean mask indicating which events to draw. If a nested list, use
            a mask for each file, else use a global mask. Cannot be used with
            entry_list...

        dsp_config : str (optional)
            name of DSP config json file containing a list of processors that
            can be applied to waveforms

        database : str or dict-like (optional)
            dict or JSON file with database of processing parameters

        aux_values : pandas dataframe (optional)
            table of auxiliary values that are one-to-one with the input
            waveforms that can be drawn or placed in the legend

        lines : str or [strs] (default 'waveform')
            name(s) of objects to draw 2D lines for. Waveforms will be drawn
            as a time-series. Scalar quantities will be drawn as horizontal
            or vertical lines, depending on units. Vectors will be drawn
            as multiple horizontal/vertical lines

        styles : (default None)
            line colors and other style parameters to cycle through when
            drawing waveforms. Can be given as:

              - dict of lists: e.g. {'color':['r', 'g', 'b'], 'linestyle':['-', '--', '.']}
              - name of predefined style; see matplotlib.style documentation
              - None: use current matplotlib rcparams style

            If a single style cycle is given, use for all lines; if a list is
            given, match to lines list.

        legend : str or [strs] (default None)
            Formatting string and values to include in the legend. This can
            be a list of values (one for each drawn object). If just a name
            is given, it will be auto-formatted to 3 digits. Otherwise,
            formatting strings in brackets can be used: ::

              "{energy:0.1f} keV, {timestamp:d} ns"

            Names will be searched in the input file, DSP processed parameters,
            or auxiliary data-table

        legend_opts : dict (default None)
            dict containing additional kwargs for matplotlib.legend

        n_drawn : int (default 1)
            number of events to draw simultaneously when calling DrawNext

        x_lim, y_lim : tuple-pair of float, pint.Quantity or str (default auto)
            range of x- or y-values and units passes as tuple.

              - None: Get range from first waveform drawn
              - pint.Quantity: set value and x-unit
              - float: get unit from first waveform drawn
              - str: convert to pint.Quanity (e.g. ('0*us', '10*us'))

        x_unit : pint.Unit or str (default auto)
            unit of x-axis

        norm : str (default None)
            name of parameter (probably energy) to use to normalize WFs
            useful when drawing multiple WFs

        align : str (default None)
            name of parameter to use for x-offset; useful, e.g., for aligning
            multiple waveforms at a particular timepoint

        buffer_len : int (default 16)
            number of waveforms to keep in memory at a time

        block_width : int (default 16)
            block width for processing chain
        """

        self.norm_par = norm
        self.align_par = align
        self.n_drawn = n_drawn
        self.next_entry = 0

        # data i/o initialization
        self.lh5_it = lh5.LH5Iterator(files_in,
                                      lh5_group,
                                      base_path=base_path,
                                      entry_list=entry_list,
                                      entry_mask=entry_mask,
                                      buffer_len=buffer_len )


        # Get the input buffer and read the first chunk
        self.lh5_in, _ = self.lh5_it.read(0)

        self.aux_vals = aux_values
        # Apply entry selection to aux_vals if needed
        if self.aux_vals is not None and len(self.aux_vals)>len(self.lh5_it):
            entries = []
            for i, f_entries in enumerate(self.lh5_it.entry_list):
                entry_offset = self.lh5_it.file_map[i-1] if i>0 else 0
                entries += [ entry_offset + entry for entry in f_entries ]
            self.aux_vals = self.aux_vals.iloc[entries].reset_index()

        # initialize objects to draw: dict from name to list of 2DLines
        if isinstance(lines, str): self.lines = { lines:[] }
        elif lines is None: self.lines = {}
        else: self.lines = { l:[] for l in lines }

        # styles
        if isinstance(styles, (list, tuple)):
            self.styles = [ None for _ in self.lines ]
            for i, sty in enumerate(styles):
                if isinstance(sty, str):
                    try:
                        self.styles[i] = plt.style.library[sty]['axes.prop_cycle']
                    except:
                        self.styles[i] = itertools.repeat(None)
                elif sty is None:
                    self.styles[i] = itertools.repeat(None)
                else:
                    self.styles[i] = cycler(**sty)
        else:
            if isinstance(styles, str):
                try:
                    self.styles = plt.style.library[styles]['axes.prop_cycle']
                except:
                    self.styles = itertools.repeat(None)
            elif styles is None:
                self.styles = itertools.repeat(None)
            else:
                self.styles = cycler(**styles)


        self.legend_format = [] # list of formatter strings
        self.legend_vals = {}  # Set up dict from names to lists of values

        if legend is None: legend = []
        elif isinstance(legend, str): legend = [legend]

        for entry in legend:
            legend_format = ""
            for st, name, form, cv in string.Formatter().parse(entry):
                if name is None:
                    # If name is none, this is the last batch of characters
                    legend_format += st
                    break

                if name=='':
                    raise KeyError("Cannot use empty formatter in "+entry)
                self.legend_vals[name] = []

                if form is None or form=='':
                    form = '~0.3P'
                cv = '' if cv is None or cv=='' else "!"+cv
                legend_format += f"{st}{{{name}:{form}{cv}}}"
            self.legend_format.append(legend_format)

        self.legend_kwargs = legend_opts if isinstance(legend_opts, dict) else {}

        # make processing chain and output buffer
        outputs = list(self.lines) + list(self.legend_vals)
        if isinstance(self.norm_par, str): outputs += [self.norm_par]
        if isinstance(self.align_par, str): outputs += [self.align_par]


        # Remove any values not found in aux_vals
        if self.aux_vals is not None:
            outputs = [ o for o in outputs if o not in self.aux_vals ]

        self.proc_chain, self.lh5_it.field_mask, self.lh5_out = build_processing_chain(self.lh5_in, dsp_config, db_dict=database, outputs=outputs, block_width=block_width)
        self.proc_chain.execute()

        # Check if all of our outputs can be found
        for name in outputs:
            if not name in self.lh5_out:
                raise KeyError("Could not find "+name+" in input lh5 file, DSP config file, or aux values")

        self.x_unit = ureg(x_unit) if x_unit else None
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.auto_x_lim = [np.inf, -np.inf]
        self.auto_y_lim = [np.inf, -np.inf]

        # Set limit and convert to x-unit if needed; also set x_unit if needed
        if self.x_lim is not None:
            self.x_lim = list(self.x_lim)
            for i in range(2):
                if isinstance(self.x_lim[i], str):
                    self.x_lim[i] = ureg.Quantity(x_lim[i])
                if isinstance(self.x_lim[i], ureg.Quantity):
                    if self.x_unit:
                        self.x_lim[i] = float(self.x_lim[i]/self.x_unit)
                    else:
                        self.x_unit = self.x_lim[i].u
                        self.x_lim[i] = self.x_lim[i].m

        # If we still have no x_unit get it from the first waveform we can find
        if self.x_unit is None:
            for wf in self.lh5_out.values():
                if not isinstance(wf, lh5.WaveformTable): continue
                self.x_unit = ureg(wf.dt_units)

        self.fig = None
        self.ax = None

    def new_figure(self):
        """Create a new figure and draw in it"""
        self.fig, self.ax = plt.subplots(1)

    def save_figure(self, f_out, *args, **kwargs):
        """ Write figure to file named f_out. See matplotlib.pyplot.savefig
        for args and kwargs"""
        self.fig.savefig(f_out)

    def set_figure(self, fig, ax=None):
        """Use an already existing figure and axis; make sure to set clear
        to False when drawing if you don't want to clear what's already there!
        Can give a WaveformBrowser object to use the fig/axis from that"""
        if isinstance(fig, WaveformBrowser):
            self.fig = fig.fig
            self.ax = fig.ax
        elif isinstance(fig, plt.Figure):
            self.fig = fig
            if ax is None:
                self.ax = fig.axes[0]
            elif isinstance(ax, plt.Axes):
                self.ax = ax
            else:
                raise TypeError("ax must be matplotlib.Axis")
        else:
            raise TypeError("fig must be matplotlib.Figure or WaveformBrowser")

    def clear_data(self):
        """ Reset the currently stored data """
        for line_data in self.lines.values(): line_data.clear()
        for leg_data in self.legend_vals.values(): leg_data.clear()
        self.auto_x_lim = [np.inf, -np.inf]
        self.auto_y_lim = [np.inf, -np.inf]
        self.n_stored = 0

    def find_entry(self, entry, append=True, safe=False):
        """
        Find the requested data associated with entry in input files and
        place store it internally without drawing it.

        Parameters
        ----------
        entry : int or [ints]
            index of entry or list of entries to find
        append : bool (default True)
            if False, clear previously found data before finding more
        safe : bool (default False)
            if False, throw an exception for out of range entries
        """
        if not append: self.clear_data()
        if hasattr(entry, '__iter__'):
            for idx in entry: self.find_entry(idx)
            return

        if entry > len(self.lh5_it):
            if safe: raise IndexError
            else: return

        # Get our current position in the I/O buffers; update if needed
        i_tb = entry - self.lh5_it.current_entry
        if not ( self.lh5_it.n_rows > i_tb >= 0 ):
            self.lh5_it.read(entry)
            self.proc_chain.execute()
            i_tb = 0

        # get scaling factor/time shift if used
        if self.norm_par is None:
            norm = 1.
        elif isinstance(self.norm_par, str):
            norm = self.lh5_out[self.norm_par].nda[entry]
        else:
            norm = self.norm_par[entry]

        if self.align_par is None:
            ref_time = 0
        elif isinstance(self.align_par, str):
            data = self.lh5_out.get(self.align_par, None)
            if isinstance(data, lh5.Array):
                ref_time = data.nda[i_tb]
                unit = data.attrs.get('units', None)
                if unit and unit in ureg and ureg.is_compatible_with(unit, self.x_unit):
                    ref_time *= float(ureg(unit)/self.x_unit)
            elif data is None:
                ref_time = self.aux_vals[self.align_par][entry]
            else:
                raise

        leg_handle = None

        # lines
        lim = math.sqrt(sys.float_info.max) # limits for v/h lines
        for name, lines in self.lines.items():
            # Get the data; note this is implicitly copying it!
            data = self.lh5_out.get(name, None)
            if isinstance(data, lh5.WaveformTable):
                y = data.values.nda[i_tb,:]/norm - ref_time
                dt = data.dt.nda[i_tb] * float(ureg(data.dt_units)/self.x_unit)
                t0 = data.t0.nda[i_tb] * float(ureg(data.t0_units)/self.x_unit)
                x = np.linspace(t0, t0+dt*(data.wf_len-1), data.wf_len)
                lines.append(Line2D(x, y))
                self._update_auto_limit(x, y)

            elif isinstance(data, lh5.Array):
                val = data.nda[i_tb]
                unit = data.attrs.get('units', None)
                if unit and unit in ureg and ureg.is_compatible_with(unit, self.x_unit):
                    # Vertical line
                    val = val*float(ureg(unit)/self.x_unit) - ref_time
                    lines.append(Line2D([val]*2, [-lim, lim]))
                    self._update_auto_limit(val, None)
                else:
                    # Horizontal line
                    lines.append(Line2D([-lim, lim], [val/norm]*2))
                    self._update_auto_limit(None, val)

            elif data is None:
                # Check for data in auxiliary table. It's unitless so I guess just do an hline...
                val = self.aux_vals[name][entry]/norm
                lines.append(Line2D([-lim, lim], [val]*2))
                self._update_auto_limit(None, val)

            else:
                raise TypeError("Cannot draw "+name+". WaveformBrowser does not support drawing lines for data of type " + str(data.__class__))

        # legend data
        for name, vals in self.legend_vals.items():
            data = self.lh5_out.get(name, None)

            if not data:
                data = ureg.Quantity(self.aux_vals[name][entry])
            elif isinstance(data, lh5.Array):
                unit = data.attrs.get('units', None)
                if unit and unit in ureg:
                    data = data.nda[i_tb]*ureg(unit)
                else:
                    data = ureg.Quantity(data.nda[i_tb])
            else:
                raise TypeError("WaveformBrowser does not adding legend entries for data of type " + data.__class__)

            vals.append(data)

        self.n_stored += 1
        self.next_entry = entry + 1

    def draw_current(self, clear=True):
        """
        Draw the waveforms and data currently held internally by this class.
        """
        # Make figure/axis if needed
        if not (self.ax and self.fig and plt.fignum_exists(self.fig.number)):
            self.new_figure()

        if clear:
            self.ax.clear()

        x_lim = self.x_lim if self.x_lim else self.auto_x_lim
        y_lim = self.y_lim
        if not y_lim:
            y_range = self.auto_y_lim[1] - self.auto_y_lim[0]
            y_lim = [self.auto_y_lim[0] - 0.05*y_range, self.auto_y_lim[1] + 0.05*y_range]
        self.ax.set_xlim(*x_lim)
        self.ax.set_ylim(*y_lim)

        leg_handles = []
        leg_labels = []
        if not isinstance(self.styles, list):
            styles = self.styles

        # draw lines
        for i, lines in enumerate(self.lines.values()):
            if isinstance(self.styles, list):
                styles = self.styles[i]
            if styles is None:
                styles = cycler(plt.rcparams)

            for line, sty in zip(lines, styles):
                if sty is not None: line.update(sty)
                if line.get_figure() is not None: line.remove()
                line.set_transform(self.ax.transData)
                self.ax.add_line(line)

                leg_handles.append(line)

        # Get legend entries
        try:
            leg_cycle = cycler(**self.legend_vals)
        except Exception:
            for form in self.legend_format:
                for i in range(self.n_stored):
                    leg_labels.append(form)
        else:
            for form in self.legend_format:
                for leg_dat in leg_cycle:
                    leg_labels.append(form.format(**leg_dat))

        # Draw legend
        self.ax.set_xlabel(self.x_unit)
        self.ax.xaxis.set_label_coords(0.98, -0.05)
        if self.x_lim:
            self.ax.set_xlim(*self.x_lim)
        if len(leg_labels)>0:
            if not clear:
                old_leg = self.ax.get_legend()
                if old_leg:
                    leg_handles = old_leg.get_lines() + leg_handles
                    leg_labels = [t.get_text() for t in old_leg.get_texts()] + leg_labels
            self.ax.legend(leg_handles, leg_labels, **self.legend_kwargs)

    def _update_auto_limit(self, x, y):
        # Helper to update the automatic limits
        y_where = {}
        if isinstance(y, np.ndarray) and self.y_lim is not None:
            y_where['where'] = ((y>=self.y_lim[0]) & (y<=self.y_lim[1]))
        x_where = {}
        if isinstance(x, np.ndarray) and self.x_lim is not None:
            x_where['where'] = ((x>=self.x_lim[0]) & (x<=self.x_lim[1]))
        if x is not None:
            self.auto_x_lim[0] = np.amin(x, **y_where, initial=self.auto_x_lim[0])
            self.auto_x_lim[1] = np.amax(x, **y_where, initial=self.auto_x_lim[1])
        if y is not None:
            self.auto_y_lim[0] = np.amin(y, **y_where, initial=self.auto_y_lim[0])
            self.auto_y_lim[1] = np.amax(y, **y_where, initial=self.auto_y_lim[1])

    def draw_entry(self, entry, append=False, clear=True, safe=False):
        """
        Draw specified entry in the current figure/axes

        Parameters
        ----------
        entry : int or [ints]
            entry or list of entries to draw
        append : bool (default False)
            if True, do not clear previously drawn entries before drawing more
        clear : bool (default True)
            if True, clear previously drawn objects in the axes before drawing
        safe : bool (default False)
            if False, throw an exception for out of range entries
        """
        self.find_entry(entry, append)
        self.draw_current(clear)

    def find_next(self, n_wfs = None, append = False):
        """Find the next n_wfs waveforms (default self.n_drawn). See find_entry"""
        if not n_wfs: n_wfs = self.n_drawn
        entries = (self.next_entry, self.next_entry+n_wfs)
        self.find_entry(range(*entries), append, safe=True)
        return entries

    def draw_next(self, n_wfs = None, append = False, clear = True):
        """Draw the next n_wfs waveforms (default self.n_drawn). See draw_next"""
        entries = self.find_next(append)
        self.draw_current(clear)
        return entries

    def reset(self):
        """ Reset to the start of the file for draw_next """
        self.clear_data()
        self.next_entry = 0

    def __iter__(self):
        while self.next_entry < len(self.lh5_it):
            yield self.draw_next()
