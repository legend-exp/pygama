from pygama.io.raw_to_dsp import build_processing_chain
from pygama import lh5
import pygama.dsp.units as units

import glob, os, itertools, contextlib, string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler

class WaveformBrowser:
    """
    The Waveform Browser is a tool meant for interacting with waveforms from
    LH5 files. This defines an interface for drawing waveforms from a file,
    drawing transformed waveforms defined using raw_to_dsp style json files,
    drawing horizontal and verticle lines at the values of calculated
    parameters, and filling a legend with calculated parameters.
    """

    def __init__(self, files_in, lh5_group, dsp_config = None, database = None,
                 n_drawn = 1, x_unit = 'ns', x_lim=None,
                 waveforms = 'waveform', wf_styles = None, lines = None,
                 legend = None, legend_opts = None, norm = None, align=None,
                 selection = None,
                 buffer_len = 128, block_width = 8, verbosity=1):
        """Constructor for WaveformBrowser:
        - file_in: name of file or list of names to browse. Can use wildcards
        - lh5_group: name of LH5 group in file to browse
        - dsp_config (optional): name of DSP config json file containing transforms available to draw
        - database (optional): dict with database of processing parameters
        - n_drawn (default 1): number of events to draw simultaneously when calling DrawNext
        - x_unit (default ns): unit for x-axis
        - x_lim (default auto): range of x-values passes as tuple
        - waveforms (default 'waveform'): name of wf or list of wf names to draw
        - wf_styles (default None): waveform colors and other style parameters to cycle through when drawing waveforms. Can be given as:
            dict of lists: e.g. {'color':['r', 'g', 'b'], 'linestyle':['-', '--', '.']}
            name of predefined style; see matplotlib.style documentation
            None: use current matplotlib style
          If a single style cycle is given, use for all lines; if a list is given, match to waveforms list.
        - lines (default None): name of parameter or list of parameters to draw hlines and vlines for
        - legend (default None): formatting string and values to include in the legend. This can be a list of values (one for each waveform in waveforms). The values can be given as a tuple whose first entry is a formatting string and subsequent entries are the values to place in the formatting string. When building a formatting string, if a name is given in the {}s, it is assumed to be a parameter from the DSP config file. An example is:
          ("{:0.1f} keV", energy)
        - legend_opts (default None): dict containing kwargs for formatting the legend
        - norm (default None): name of parameter (probably energy) to use to normalize WFs; useful when drawing multiple
        - align (default None): name of time parameter to set as 0 time; useful for aligning multiple waveforms
        - selection (optional): selection of events to draw. Can be either a list of event indices or a numpy array mask (ala pandas).
        - buffer_len (default 128): number of waveforms to keep in memory at a time
        - block_width (default 8): block width for processing chain
        """
        self.verbosity = verbosity

        # data i/o initialization
        self.lh5_st = lh5.Store(keep_open=True)
        if isinstance(files_in, str): files_in = [files_in]
        
        # Expand wildcards and map out the files
        self.lh5_files = [f for f_wc in files_in for f in sorted(glob.glob(os.path.expandvars(f_wc)))]
        self.lh5_group = lh5_group
        # file map is cumulative lenght of files up to file n. By doing searchsorted left, we can get the file for a given wf index
        self.file_map = np.array([self.lh5_st.read_n_rows(lh5_group, f) for f in self.lh5_files], 'int64')
        np.cumsum(self.file_map, out=self.file_map)

        # Get the input buffer and read the first chunk
        self.lh5_in = self.lh5_st.get_buffer(self.lh5_group, self.lh5_files[0], buffer_len)
        self.lh5_st.read_object(self.lh5_group, self.lh5_files[0], start_row=0, n_rows=buffer_len, obj_buf=self.lh5_in)
        self.buffer_len = buffer_len
        self.current_file = None
        self.current_chunk = None

        # initialize stuff for iteration
        self.selection = selection
        self.index_it = None
        self.reset()
        self.n_drawn = n_drawn

        # initialize list of objects to draw
        if isinstance(waveforms, str): self.wf_names = [waveforms]
        elif waveforms is None: self.wf_names = []
        else: self.wf_names = list(waveforms)
        self.wf_data = [ [] for _ in self.wf_names ]

        # wf_styles
        if isinstance(wf_styles, list) or isinstance(wf_styles, tuple):
            self.wf_styles = [ None for _ in self.wf_data ]
            for i, sty in enumerate(wf_styles):
                if isinstance(sty, str):
                    try:
                        self.wf_styles[i] = plt.style.library[sty]['axes.prop_cycle']
                    except:
                        self.wf_styles[i] = itertools.repeat(None)
                elif sty is None:
                    self.wf_styles[i] = itertools.repeat(None)
                else:
                    self.wf_styles[i] = cycler(**sty)
        else:
            if isinstance(wf_styles, str):
                try:
                    self.wf_styles = plt.style.library[wf_styles]['axes.prop_cycle']
                except:
                    self.wf_styles = itertools.repeat(None)
            elif wf_styles is None:
                self.wf_styles = itertools.repeat(None)
            else:
                self.wf_styles = cycler(**wf_styles)
        
        if lines is None: self.line_names = []
        elif isinstance(lines, list): self.line_names = lines
        elif isinstance(lines, tuple):  self.line_names = list(lines)
        else: self.line_names = [lines]
        self.line_data = [ [] for _ in self.line_names ]
        
        if legend is None: legend = []
        elif not isinstance(legend, list): legend = [legend]

        # Set up the legend format strings and collect input values
        self.legend_input = []
        self.legend_format = []
        for entry in legend:
            legend_input = []
            legend_format = ''
            if not isinstance(entry, tuple):
                entry = (entry,)

            for val in entry:
                if isinstance(val, str):
                    for st, name, form, cv in string.Formatter().parse(val):
                        legend_format += st
                        if name is not None:
                            legend_format += '{'
                            legend_input.append(name)
                            if form is not None and form != '':
                                legend_format += ':' + form
                            if cv is not None and cv != '':
                                legend_format += '!' + cv
                            legend_format += '}'
                else:
                    # find any {}s to fill from the formatter
                    idxs = [i for i, inp in enumerate(legend_input) if isinstance(inp,str) and inp=='']
                    if idxs: # if we found a {}. it's already in the formatter
                        legend_input[idxs[0]] = val
                    else: # otherwise add to formatter
                        legend_input.append(val)
                        if legend_format!='': legend_format += ', '
                        if isinstance(val, pd.Series):
                            legend_format += val.name + ' = {:.3g}'
                        elif isinstance(val, np.ndarray):
                            legend_format += '{:.3g}'
            self.legend_input.append(legend_input)
            self.legend_format.append(legend_format)

        self.legend_data = [ [] for _ in self.legend_input ]
        self.legend_kwargs = legend_opts if legend_opts else {}
        
        self.norm_par = norm
        self.align_par = align

        self.x_unit = units.unit_parser.parse_unit(x_unit)
        self.x_lim = x_lim

        # make processing chain and output buffer
        outputs = self.wf_names + \
                  [name for name in self.line_names if isinstance(name, str)] + \
                  [name for name in self.legend_input  if isinstance(name, str)]
        if isinstance(self.norm_par, str): outputs += [self.norm_par]
        if isinstance(self.align_par, str): outputs += [self.align_par] 
        
        self.proc_chain, self.lh5_out = build_processing_chain(self.lh5_in, dsp_config, db_dict=database, outputs=outputs, verbosity=self.verbosity, block_width=block_width)
        
        self.fig = None
        self.ax = None        
    
    def new_figure(self):
        """Create a new figure and draw in it"""
        self.fig, self.ax = plt.subplots(1)

    def set_figure(self, fig, ax=None):
        """Use an already existing figure and axis; make sure to set clear to False when drawing if you don't want to clear what's already there! Can give a WaveformBrowser object to use the fig/axis from that"""
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
        for wf_set in self.wf_data: wf_set.clear()
        for line_set in self.line_data: line_set.clear()
        for leg_data in self.legend_data: leg_data.clear()
        
    def find_entry(self, entry, append=True):
        """
        Find the requested data associated with entry in input files and
        place it in self.wf_data, self.line_data and self.legend_data. Set
        append to False to clear these buffers before fetching the entry/ies.
        Can give a list/tuple to find multiple entries.
        """
        if not append: self.clear_data()
        if isinstance(entry, list) or isinstance(entry, tuple):
            for idx in entry: self.find_entry(idx)
            return
        
        # figure out which file we are reading from and the chunk/index within the file, using the file map
        file_no = np.searchsorted(self.file_map, entry, 'left')
        if file_no>len(self.lh5_files):
            raise IndexError
        # get chunk and index within this chunk
        file_beg = self.file_map[file_no-1] if file_no>0 else 0
        chunk, index = divmod(entry - file_beg, self.buffer_len)
        
        # Update the chunk as needed
        if file_no != self.current_file or chunk != self.current_chunk:
            self.current_chunk = chunk
            self.current_file = file_no
            self.lh5_in, n_read = self.lh5_st.read_object(self.lh5_group,
                                                          self.lh5_files[file_no],
                                                          start_row=chunk*self.buffer_len,
                                                          n_rows=len(self.lh5_in),
                                                          obj_buf=self.lh5_in)
            self.proc_chain.execute(0, n_read)
            

        # get scaling factor/time shift if used
        if self.norm_par is None:
            norm = 1.
        elif isinstance(self.norm_par, str):
            norm = self.lh5_out[self.norm_par].nda[index]
        else:
            norm = self.norm_par[entry]

        if self.align_par is None:
            ref_time = 0
        elif isinstance(self.align_par, str):
            unit = self.lh5_out[self.align_par].attrs['units']
            dt = units.convert(1, units.unit_parser.parse_unit(unit), self.x_unit)
            ref_time = self.lh5_out[self.align_par].nda[index]*dt
        else:
            ref_time = self.align_par[entry]
            
        leg_handle = None

        #waveforms
        for wf_name, wf_data in zip(self.wf_names, self.wf_data):
            # Get the data; note this is implicitly copying it!
            y = self.lh5_out[wf_name].nda[index]/norm
            if self.x_unit.derivation == self.proc_chain._clk.derivation:
                # this is a WF
                dt = units.convert(1, self.proc_chain._clk, self.x_unit)
                x = np.linspace(-ref_time, len(y)*dt-ref_time, len(y), 'f')
            elif self.x_unit.derivation == (1/self.proc_chain._clk).derivation:
                # this is a FT
                f_nyq = units.convert(1, 0.5/self.proc_chain._clk, self.x_unit)
                x = np.linspace(0, f_nyq, len(y), 'f')
            wf_data.append((x, y))
                    
        # lines
        for line_name, line_data in zip(self.line_names, self.line_data):
            try: # if unit is time, do vline
                unit = units.unit_parser.parse_unit(self.lh5_out[line_name].attrs['units'])
                val = self.lh5_out[line_name].nda[index]*unit
                val -= ref_time*self.x_unit
                
            except: # else do hline
                val = self.lh5_out[line_name].nda[index]/norm

            line_data.append(val)

        # legend data
        for legend_input, legend_data in zip(self.legend_input, self.legend_data):
            leg_vals = []
            for val in legend_input:
                if isinstance(val, str):
                    leg_vals.append(self.lh5_out[val].nda[index])
                else:
                    leg_vals.append(val[entry])
            legend_data.append(leg_vals)
    
    def draw_current(self, clear=True):
        """
        Draw the waveforms and data currently held internally by this class.
        """
        # Make figure/axis if needed
        if not (self.ax and self.fig and plt.fignum_exists(self.fig.number)):
            self.new_figure()
        
        if clear:
            self.ax.clear()

        leg_handles = []
        leg_labels = []
        if not isinstance(self.wf_styles, list):
            wf_styles = self.wf_styles
            
        # draw waveforms
        for i, wf_set in enumerate(self.wf_data):
            if isinstance(self.wf_styles, list):
                wf_styles = self.wf_styles[i]
            if wf_styles is None:
                wf_styles = itertools.repeat(None)
            
            for wf, sty in zip(wf_set, wf_styles):
                if sty is None:
                    wf_line, = self.ax.plot(*wf, '-')
                else:
                    wf_line, = self.ax.plot(*wf, **sty)
                leg_handles.append(wf_line)

        # draw legend
        for legend_data, legend_format in zip(self.legend_data, self.legend_format):
            for legend_vals in legend_data:
                leg_labels.append(legend_format.format(*legend_vals))

        # draw hlines and vlines
        for lines in self.line_data:
            for val in lines:
                if isinstance(val, units.unit):
                    self.ax.axvline(units.convert(1, val, self.x_unit))
                else:
                    self.ax.axhline(val)
        
        self.ax.set_xlabel(self.x_unit.label)
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
        
                
    def draw_entry(self, entry, append=False, clear=True):
        """Draw specified entries from file. Entry_list can be either a single value or list of values representing the index of an event within all files. If append is True, previously drawn entries will be drawn along with this one. If clear is False, the axis will not be cleared before drawing. Return the axis object"""
        self.find_entry(entry, append)
        self.draw_current(clear)

    def find_next(self, n_wfs = None, append = False):
        """Find the next n_wfs (default to self.n_drawn) waveforms indicated by self.selection and place them in self.wf_data, self.line_data and self.legend_data. If append is True, do not clear these buffers first."""
        if not n_wfs: n_wfs = self.n_drawn

        wf_indices = [ i_wf for _, i_wf in zip(range(n_wfs), self.index_it) ]
        if len(wf_indices) < n_wfs: self.eof=True
        self.find_entry(wf_indices, append)

        return wf_indices
        
    def draw_next(self, n_wfs = None, append = False, clear = True):
        """Draw the next n_wfs waveforms on the same axis. If a selection was set, only draw waveforms from that selection. Return a list of waveform indices drawn and the axis object:
           n_wfs: number of waveforms to draw (default is self.n_wfs)
           append: set to True to prevent clearing of axis before drawing
           clear: set False to draw on an already defined axis
        """
        wf_indices = self.find_next(n_wfs, append)
        self.draw_current(clear)
        
        return wf_indices
            
    def reset(self):
        """ Reset to the start of the file for draw_next """
        self.eof = False
        try:
            if self.selection is None:
                self.index_it = iter(range(0, self.file_map[-1]))
            elif isinstance(self.selection, list) or isinstance(self.selection, tuple): # index list
                self.index_it = iter(self.selection)
            elif isinstance(self.selection, np.ndarray): # numpy boolean mask
                self.index_it = iter(np.nonzero(self.selection)[0] )
            elif isinstance(self.selection, pd.Series): # pandas series mask
                self.index_it = iter(np.nonzero(self.selection.values)[0] )
            else:
                raise Exception
        except:
            print("Not sure what to do with selection", self.selection, "("+str(self.selection.__class__)+")")
        
    def __iter__(self):
        self.reset()
        return self
    
    def __next__(self):
        """ Call draw_next... """
        if self.eof:
            raise StopIteration
        return self.draw_next()
