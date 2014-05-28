#!/usr/bin/env python

# CURRENTLY BEING UPDATED

"""
A module to plot root histograms in a number of styles.
The goal is to remove the time investment needed to make
good-looking plots by providing versatile templates.
Now using matplotlib to generate the plots, because it
tends to cause less headaches. 

This can be imported as a module in order to use the 
output styles with other forms of input. 
"""
__author__  = 'Brad Axen'
__version__ = '0.9.0'
__status__  = 'Prototype'

import sys

import argparse
import math
import shlex
import style
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from copy import deepcopy
from matplotlib.colors import LogNorm,ListedColormap
from scipy.optimize import leastsq            
from scipy.special import erf

class PlotError(Exception):
    pass
            
class Hist:
    """Encapsulates the information from a ROOT histogram for plotting in mpl.
    This class is ONLY designed to handle the plotting aspects. It is not 
    full featured in other respects. 

    Arguments
    lefts: The starting x-values for the bins (list or np array).
    widths: The widths of the bins (list or np array).
    values: The 'height' of the bins (list or np array).
    yerr: Errors on the height of the bins (list or np array).
    
    Keyword Arguments
    exclude: Remove bins which intersect the specified interval (a,b).
    scale: Multiply the heights by the specified constant.
    norm: Multiply the heights by a constant so that the integral is norm.
    
    Notes
    All remaining kwargs are passed to matplotlib errorbar/bar when plotted.
    """
    def __init__(self, lefts, widths, values, yerr=None, label=None, color=style.def_grey, emph="neutral", **kwargs):
        self.lefts = np.array(lefts)
        self.widths = np.array(widths)
        self.values = np.array(values)
        self.yerr = np.array(yerr) if yerr is not None else None
        self.color = color
        self.emph = emph
        if label is not None and label.lower() == "_none":
            self.label = None
        else:
            self.label = label
        # Handle options which modify the data.
        if 'exclude' in kwargs: 
            interval = kwargs.pop('exclude')
            indices = np.where( (self.lefts + self.widths <= interval[0]) | (self.lefts >= interval[1]) )[0]
            self.lefts = self.lefts[indices]
            self.widths = self.widths[indices]
            self.values = self.values[indices]
            self.yerr = self.yerr[indices] if yerr is not None else None
        if 'scale' in kwargs:
            factor = kwargs.pop('scale')
            self.values = factor * self.values
            self.yerr = factor * self.yerr if yerr is not None else None
        if 'norm' in kwargs:
            factor = kwargs.pop('norm') / np.sum(self.values)
            self.values = factor * self.values
            self.yerr = factor * self.yerr if yerr is not None else None
        # Remaining options will be passed to mpl when plotting.
        self.options = kwargs
        return

    @classmethod
    def from_bin_edges(cls, bin_edges, values, yerr=None, **kwargs):
        bin_edges = np.array(bin_edges)
        return cls(bin_edges[:-1], np.diff(bin_edges), values, yerr, **kwargs)

    @classmethod
    def from_centers(cls, centers, values, yerr=None, **kwargs):
        centers = np.array(centers)
        diffs = np.diff(centers)
        diffs = np.append(diffs, diffs[-1])
        return cls(centers-(diffs/2.0), diffs, values, yerr, **kwargs)

    @classmethod
    def from_hist(cls, hist, **options):
        """Create a Hist from a ROOT TH1.

        Can use a TH1 or a projection/average of a TH2, specified in kwargs.
        """
        # For 1D hists derived from 2D hists
        if 'rebin' in options:
            hist.Rebin(options.pop('rebin'))
        if 'projectionx' in options:
            arg = options.pop('projectionx')
            bin1 = hist.GetYaxis().FindBin(arg[0])
            bin2 = hist.GetYaxis().FindBin(arg[1])
            hist = hist.ProjectionX(hist.GetName()+"_px",bin1,bin2,"e")
        if 'projectiony' in options:
            arg = options.pop('projectiony')
            bin1 = hist.GetXaxis().FindBin(arg[0])
            bin2 = hist.GetXaxis().FindBin(arg[1])
            hist = hist.ProjectionY(hist.GetName()+"_py",bin1,bin2,"e")
        if 'averagex' in options:
            arg = options.pop('averagex')
            bin1 = hist.GetYaxis().FindBin(arg[0])
            bin2 = hist.GetYaxis().FindBin(arg[1])
            avg_hist = hist.ProjectionX(hist.GetName()+"_px",bin1,bin1,"e")
            for i in range(1,avg_hist.GetNbinsX() + 1):
                avg_hist.SetBinContent(i,0)
                avg_hist.SetBinError(i,0)
            tmp_weights = avg_hist.Clone(hist.GetName()+"_weights")
            for i in range(bin1,bin2):
                tmp_hist = hist.ProjectionX(hist.GetName()+"_px",i,i+1,"e")
                for j in range(1,avg_hist.GetNbinsX() + 1):
                    tmp_weights.SetBinContent(j,tmp_weights.GetBinContent(j) + (tmp_hist.GetBinError(j) ** -2))
                    tmp_hist.SetBinContent(j,tmp_hist.GetBinContent(j) * (tmp_hist.GetBinError(j) ** -2))
                    tmp_hist.SetBinError(j,tmp_hist.GetBinError(j) * (tmp_hist.GetBinError(j) ** -2))
                avg_hist.Add(tmp_hist)
            avg_hist.Divide(tmp_weights)
            hist = avg_hist
        if 'averagey' in options:
            arg = options.pop('averagey')
            bin1 = hist.GetXaxis().FindBin(arg[0])
            bin2 = hist.GetXaxis().FindBin(arg[1])
            avg_hist = hist.ProjectionY(hist.GetName()+"_py",bin1,bin1,"e")
            for i in range(1,avg_hist.GetNbinsX() + 1):
                avg_hist.SetBinContent(i,0)
                avg_hist.SetBinError(i,0)
            tmp_weights = avg_hist.Clone(hist.GetName()+"_weights")
            for i in range(bin1,bin2):
                tmp_hist = hist.ProjectionY(hist.GetName()+"_py",i,i+1,"e")
                for j in range(1,avg_hist.GetNbinsX() + 1):
                    tmp_weights.SetBinContent(j,tmp_weights.GetBinContent(j) + (tmp_hist.GetBinError(j) ** -2))
                    tmp_hist.SetBinContent(j,tmp_hist.GetBinContent(j) * (tmp_hist.GetBinError(j) ** -2))
                    tmp_hist.SetBinError(j,tmp_hist.GetBinError(j) * (tmp_hist.GetBinError(j) ** -2))
                avg_hist.Add(tmp_hist)
            avg_hist.Divide(tmp_weights)
            hist = avg_hist
        if 'error' in options and options.pop('error'):
            for i in range(1, hist.GetNbinsX()+1):
                hist.SetBinContent(i,hist.GetBinError(i)/hist.GetBinContent(i))
                hist.SetBinError(i,0.0)
        nbins = hist.GetNbinsX()
        bin_edges = np.fromiter((hist.GetBinLowEdge(i) for i in xrange(1,nbins+2)), np.float, nbins+1)
        values = np.fromiter((hist.GetBinContent(i) for i in xrange(1,nbins+1)), np.float, nbins)
        yerr = np.fromiter((hist.GetBinError(i) for i in xrange(1,nbins+1)), np.float, nbins)
        return cls.from_bin_edges(bin_edges,values,yerr,**options)

    def divide(self, hist2, label=None):
        h = deepcopy(self)
        h.values = h.values / hist2.values
        h.yerr = h.values * np.sqrt( (self.yerr/self.values)**2 + (hist2.yerr/hist2.values)**2 )
        h.label = label
        return h

    def plot_bars(self, ax, bottom=None, log=False):
        """Plot histogram as a filled bar graph. (Like 'HistStack')"""
        if bottom is None and log:
            bottom = np.ones_like(self.values) * min(self.values[self.values > 0 ]) * .1
        return ax.bar(self.lefts, self.values, self.widths, color=self.color, label=self.label,edgecolor=self.color, bottom=bottom, log=log, **self.options)[0]

    def plot_lines(self, ax):
        """Plot histogram as stepped lines with errorbars. (Like 'E0')"""
        return ax.errorbar(self.lefts + (self.widths*.5), self.values, yerr=self.yerr, xerr=.5*self.widths, marker='o', color=self.color, label=self.label, capsize=0, ls='', **self.options)

    def plot_points(self, ax):
        """Plot histogram as stepped lines with errorbars. (Like 'E0')"""
        return ax.errorbar(self.lefts + (self.widths*.5), self.values, yerr=self.yerr, marker='o', color=self.color, label=self.label, ls='', capsize=0, **self.options)

    def plot_noerror(self, ax):
        """Plot histogram as stepped lines without errorbars. (Like 'hist')"""
        tmp_lefts = deepcopy(self.lefts)
        tmp_lefts = np.append(tmp_lefts, self.lefts[-1] + self.widths[-1])
        tmp_values = deepcopy(self.values)
        tmp_values = np.append(tmp_values, self.values[-1])
        return ax.plot(tmp_lefts, tmp_values, color=self.color, drawstyle='steps-post', label=self.label, **self.options)


class Hist2D:
    """Encapsulates the information from a ROOT 2D histogram for plotting in mpl.
    This class is ONLY designed to handle the plotting aspects. It is not 
    full featured in other respects. 

    Arguments
    xbin_edges: The x-values for the bin edges (list or np array).
    ybin_edges: The y-values for the bin edgess (list or np array).
    values: The 'height' of the bins (list[list] or 2D np array).
    
    Keyword Arguments
    scale: Multiply the heights by the specified constant.
    norm: Multiply the heights by a constant so that the integral is norm.
    
    Notes
    All remaining kwargs are passed to matplotlib errorbar/bar when plotted.
    The x(y)bin_edges should have len nbinsx(y) + 1
    Errors are currently not supported. (How would they fit on the plot?)
    If errors are important, consider using the projection/average options of Hist.
    Bin ranges with gaps are currently not supported.
    """
    def __init__(self, xbin_edges, ybin_edges, values, label=None, color='jet', emph="neutral", **kwargs):
        self.xbins = np.array(xbin_edges)
        self.ybins = np.array(ybin_edges)
        self._xgrid, self._ygrid = np.meshgrid(self.xbins, self.ybins)
        self.values = np.array(values)
        self.color = color
        self.emph = emph
        self.label = label
        # Handle options which modify the data.
        if 'scale' in kwargs:
            factor = kwargs.pop('scale')
            self.values = factor * self.values
            self.err = factor * self.err if err is not None else None
        if 'norm' in kwargs:
            factor = kwargs.pop('norm') / np.sum(self.values)
            self.values = factor * self.values
            self.err = factor * self.err if err is not None else None
        # Remaining options will be passed to mpl when plotting.
        self.options = kwargs
        return

    @classmethod
    def from_hist(cls, hist, **options):
        """Create a Hist2D from a ROOT TH2."""
        if 'rebin' in options:
            hist.Rebin(options.pop('rebin'))
        nbinsx = hist.GetNbinsX()
        nbinsy = hist.GetNbinsY()
        xbin_edges = np.fromiter((hist.GetXaxis().GetBinLowEdge(i) for i in xrange(1,nbinsx+2)), np.float, nbinsx+1)
        ybin_edges = np.fromiter((hist.GetYaxis().GetBinLowEdge(i) for i in xrange(1,nbinsy+2)), np.float, nbinsy+1)
        values = [[hist.GetBinContent(i,j) for i in xrange(1,nbinsx+1)] for j in xrange(1,nbinsy+1)]
        values = np.array(values)
        return cls(xbin_edges, ybin_edges, values, **options)

    def divide(self, hist2, label=None):
        h = deepcopy(self)
        h.values = h.values / hist2.values
        h.err = h.values * np.sqrt( (self.err/self.values)**2 + (hist2.err/hist2.values)**2 )
        h.label = label
        return h

    def plot_colors(self, log=False):
        """Plot 2D histogram as a grid of colors."""
        # still needs to be implemented
        return plt.pcolor(self._xgrid, self._ygrid, self.values, cmap=self.color, **self.options)


class HistStack:
    """Container of histograms that plots them as a stacked bar graph."""
    def __init__(self, *hists):
        self.hists = hists
    
    def plot_bars(self, ax, bottom=None, log=False):
        """Plot all histograms as stacked bar plots."""
        handles = []
        for hist in self.hists:
            handles.append(hist.plot_bars(ax, bottom, log))
            if bottom is None:
                bottom = deepcopy(hist.values)
            else:
                bottom += hist.values
        return handles

    def sum_hist(self, label=None):
        """Return a histogram which is equal to the sum of all histograms in the stack."""
        values = [h.values for h in self.hists]
        errors = [h.yerr for h in self.hists]
        hist = deepcopy(self.hists[0])
        hist.values = sum(values)
        hist.yerr = np.sqrt(sum([error*error for error in errors]))
        hist.label = label
        hist.color = style.def_grey
        return hist


def tex_escape(s):
    # For now, assume if someone is using "$" then they are correctly formatting their own latex
    if "$" in s:
        return s
    for symbol in ["\\", "_", "%"]:
        s = s.replace(symbol, "\\" + symbol)
    s = s.replace("<", r"\textless\ ")
    s = s.replace(">", r"\textgreater\ ")
    return s

# Fitting Functions
def scurve(v,x):
    # v = [mu,sigma,N]
    return .5*v[2]*(1 + erf((x - v[0])/(1.4142*v[1])))


def fit_scurve(xs, counts):
    err = lambda v, x, y: (scurve(v,x) - y)
    v0 = [(xs[0] + xs[1])/2, .01, 255]
    v, success = leastsq(err, v0, args=(xs,counts))    
    return v


#Fitting dictionary:
fits = {'scurve':[scurve,fit_scurve]}



def tick_format_log(x, p):
    if x == 0:
        return '0'
    exp = np.log10(x)
    if np.abs(exp-int(exp)) < .00000001:
        return r"10$^{\mathsf{%d}}$" % exp
    elif (x - int(x)) < .00000001:
        return str(int(x))
    else:
        return str(x)

def tick_format_linear(x, p):
    if (x - int(x)) < .00000001:
        return str(int(x))
    else:
        return str(x)
    return str(x)


def hist_arithmetic(hname, fname, root_file):
    """ Interprets a string as a histogram name and returns a ROOT hist.
    
    Notes:
    Accepts "arithmetic" syntax, like hist1+hist2. It will recursively handle
    these operations. Currently supported operations are: +,-, and -/.
    -/ is shorthand for percantage difference.
    """
    if '-/' in hname: #shorthand to do %difference
        hist = hist_arithmetic(hname.split("-/")[0], fname, root_file)
        dhist = hist_arithmetic(hname[hname.index('-/')+2:],root_file)
        hist.Add(dhist,-1)
        hist.Divide(dhist)
        for i in range(1,hist.GetNbinsX()+1):
            hist.SetBinContent(i,abs(hist.GetBinContent(i)))
        return hist        
    elif '+' in hname:
        hist = hist_arithmetic(hname.split("+")[0], fname, root_file)
        hist.Add(hist_arithmetic(hname[hname.index('+')+1:],root_file))
        return hist
    elif '-' in hname:
        hist = hist_arithmetic(hname.split("-")[0], fname, root_file)
        hist.Add(hist_arithmetic(hname[hname.index('-')+1:], fname, root_file),-1)
        return hist
    else:
        hist = root_file.Get(hname)
        if not hist:
            print "Could not find %s in %s." % (hname, fname)
            raise PlotError
        return hist
        


def expand_targets(targets, suffix=".root"):
    """ Expands list of target arguments into two lists: file names and hist names. 

    Notes:
    Supports a number of syntaxes which are described in help.
    """
    files = [x for x in targets if suffix in x]
    hnames = [x for x in targets if suffix not in x]
    if len(files) == 1:
        files = [files[0] for i in range(len(hnames))]
    elif len(hnames) == 1:
        hnames = [hnames[0] for i in range(len(files))]
    elif len(files) > 1 and len(hnames) > 1:
        # Most general case but not often necessary.
        tmp_files, files, hnames = [],[],[]
        for i,target in enumerate(targets):
            if suffix not in target and i == 0:
                print "Unable to interpret targets. Check the help documentation."
                raise PlotError
            if suffix in target:
                tmp_files.append(target)
            if suffix not in target:
                files.append(tmp_files[-1])
                hnames.append(target)
    else:
        print "Invalid set of targets. Check the help documentation."
        raise PlotError
    return files,hnames


def get_hists(args):
    """ Returns a list of Hist or Hist2D given the input arguments.
    Notes:
    This version handles ROOT files and ROOT histograms. An alternate
    version of this function can be written and provided as an argument
    to main in order to use the plotting system with other input types.
    """
    from ROOT import TFile
    files, hnames = expand_targets(args.targets)
    if not args.labels:
        labels = [None for x in files]
    else:
        labels = args.labels
        if len(labels) != len(hnames):
            print "The number of labels doesn't match the number of hists. Use _None to not specify a label for a given histogram."
            raise PlotError
    if not args.colors:
        colors = [style.def_grey for x in files]
    else:
        colors = args.colors
        if len(labels) != len(hnames):
            print "The number of colors doesn't match the number of hists."
            raise PlotError

    gopts = {}
    if args.rebin: gopts["rebin"] = args.rebin
    if args.norm: gopts["norm"] = args.norm
    if args.error: gopts["error"] = args.error
    if args.scale: gopts["scale"] = args.scale
    if args.exclude: gopts["exclude"] = args.exclude
    if args.projectionx: gopts["projectionx"] = args.projectionx
    if args.projectiony: gopts["projectiony"] = args.projectiony    
    if args.averagex: gopts["averagex"] = args.averagex
    if args.averagey: gopts["averagey"] = args.averagey
    if args.emph: gopts["emph"] = args.emph

    hists = []
    if args.sub in ['color2D']:
        for i in range(len(hnames)):
            root_file = TFile.Open(args.inDir + files[i])
            rhist = hist_arithmetic(hnames[i], args.inDir + files[i], root_file)
            hists.append(Hist2D.from_hist(rhist, label=labels[i], **opts))            
            root_file.Close()
    else:
        for i in range(len(hnames)):
            opts = gopts.copy()
            for opt in args.options:
                if opt["pos"] == i:
                    opts = dict(gopts.items() + opt.items())
                    del opts["pos"]
                    break
            root_file = TFile.Open(args.inDir + files[i])
            rhist = hist_arithmetic(hnames[i], args.inDir + files[i], root_file)
            hists.append(Hist.from_hist(rhist, label=labels[i], color=colors[i], **opts))
            root_file.Close()
    return hists


def plot_lines(args, ax, colors, cindex, hists):
    """ Plot line-type arguments. """
    if args.line:
        ax.axhline(args.line[0], color=colors[cindex], linestyle='--', label=args.line[1])
        cindex += 1
    elif args.avline is not None:
        avh = deepcopy(hists[args.avline])
        avh.values = np.ones_like(avh.values) * np.average(avh.values)
        avh.label = None
        avh.options['linestyle'] = '--'
        avh.plot_noerror(ax)
    if args.box:
        ax.axhspan(args.box[0], args.box[1], color=colors[cindex], zorder=-10)
        cindex += 1
    return cindex

def setup_figure(args, ax):
    """ Apply figure-wide arguments. """
    if args.title:
        ax.set_title(tex_escape(args.title))
    if args.logy:
        ax.set_yscale('log')
    if args.logx:
        ax.set_xscale('log')
    if args.min is not None:
        ax.set_ylim(bottom=args.min)
    if args.max is not None:
        ax.set_ylim(top=args.max)
    if args.limits:
        ax.set_xlim(args.limits[0],args.limits[1])
    if args.ylabel:
        ax.set_ylabel(args.ylabel, y=1, va='top')        
    if args.func == color2D or (args.xlabel and not args.ratio):
        ax.set_xlabel(tex_escape(args.xlabel), x=1, ha='right')        
    if not args.logx:
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    if not args.logy:
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    return


def setup_ratio(args, ax, ax_ratio):
    """ Setup additional ratio subplot and apply ratio specific arguments."""
    main_ticks = ax.yaxis.get_major_ticks()
    main_ticks[0].label1.set_visible(False)
    ax.yaxis.set_label_coords(-0.12,1)
    ax_ratio.yaxis.set_label_coords(-0.12,.5)
    if args.logx:
        ax_ratio.set_xscale('log')
    if args.xlabel:
        ax_ratio.set_xlabel(tex_escape(args.xlabel), x=1, ha='right')
    if args.rlabel:
        ax_ratio.set_ylabel(args.rlabel)
    if args.limits:
        ax_ratio.set_xlim(args.limits[0],args.limits[1])
    if args.rmin is not None:
        ax_ratio.set_ylim(bottom=args.rmin)
    if args.rmax is not None:
        ax_ratio.set_ylim(top=args.rmax)
    ax_ratio.yaxis.grid(True)
    xmin, xmax, ymin, ymax = ax_ratio.axis()
    ax_ratio.yaxis.set_major_locator(ticker.MaxNLocator(3))
    ax_ratio.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    if not args.logx:
        ax_ratio.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    return


def compare(hists, args):
    """ Plot hists as side-by-side histograms with options given in arguments. """
    if args.total:
        total_h = deepcopy(hists[0])
        if args.line:
            total_h.values = np.sum( h.values for h in hists)
            total_h.values = total_h.values + (np.ones_like(total_h.values)* args.line[0])
        else:
            total_h.values = np.sum((h.values for h in hists))
        total_h.yerr = np.sqrt(np.sum( h.yerr*h.yerr for h in hists))
        total_h.label = 'Total'
        hists.insert(0,total_h)
    if args.totalsq:
        total_h = deepcopy(hists[0])
        if args.line:
            total_h.values = np.sum( h.values*h.values for h in hists)
            total_h.values = total_h.values + (np.ones_like(total_h.values)* (args.line[0]*args.line[0]))
            total_h.values = np.sqrt(total_h.values)
        else:
            total_h.values = np.sqrt(np.sum( h.values*h.values for h in hists))
        total_h.yerr = np.zeros_like(total_h.yerr)
        total_h.label = 'Total'
        hists.insert(0,total_h)
    
    neutral = style.get_colors("neutral");
    if not args.colors:
        if args.colorscheme:
            neutral = style.get_colors(args.colorscheme,len(hists));
            bold    = style.get_colors(args.colorscheme,len(hists));
            light   = style.get_colors(args.colorscheme,len(hists));
        else:
            neutral = style.get_colors("neutral");
            bold    = style.get_colors("bold");
            light   = style.get_colors("light");
        for i,hist in enumerate(hists):
            if "bold" == hist.emph:
                hist.color = bold[i%len(bold)]
            elif "light" == hist.emph:
                hist.color = light[i%len(bold)]
            else:
                hist.color = neutral[i%len(bold)]

    fig = plt.figure()
    if args.ratio:
        gs = gridspec.GridSpec(2,1,height_ratios=[3,1])
        ax = plt.subplot(gs[0])
        ax_ratio = plt.subplot(gs[1], sharex=ax)
        rhists = [hist.divide(hists[-1]) for hist in hists[:-1]]
        for rhist in rhists:
            rhist.plot_lines(ax_ratio)
        plt.subplots_adjust(hspace=0)
        plt.setp(ax.get_xticklabels(), visible=False)
        setup_ratio(args, ax, ax_ratio)
    else:
        ax = fig.add_subplot(111)

    if args.alpha:
        for hist in hists:
            hist.options['alpha'] = args.alpha

    if args.fit:
        for hist in hists:
            v = fits[args.fit[0]][1](hist.lefts + hist.widths/2.0, hist.values)
            params = tuple([v[i] for i in xrange(args.fit[1].count('%'))])
            hist.label += (args.fit[1] % params)
            x = np.linspace(hist.lefts[0],hist.lefts[-1]+hist.widths[-1],200)
            ax.plot(x,fits[args.fit[0]][0](v,x), color = hist.color)

    if args.noerror:
        for hist in hists:
            hist.plot_noerror(ax)
    elif args.points:
        for hist in hists:
            hist.plot_points(ax)
    else:
        for hist in hists:
            hist.plot_lines(ax)


    fig.subplots_adjust(bottom=.12, left=.14)
    plot_lines(args, ax, neutral, len(hists), hists)
    setup_figure(args, ax)
    ax.legend(frameon=False,loc=args.loc)
    
    if "." not in args.name:
        fig.savefig(args.outDir+args.name+".pdf", transparent=args.transparent)
        print "Saving figure: %s.pdf" % args.outDir+args.name
    else:
        fig.savefig(args.outDir+args.name, transparent=args.transparent)
        print "Saving figure: %s" % args.outDir+args.name

    plt.close(fig)
    return


def stack(hists, args):
    """ Plot a stack of histograms and optional data with options given in arguments. """
    data = [hist for (i,hist) in enumerate(hists) if i in args.data]
    stack = [hist for (i,hist) in enumerate(hists) if i not in args.data]
    all_hists = data + stack # reordered hists

    neutral = style.get_colors("neutral");
    if not args.colors:
        if args.colorscheme:
            neutral = style.get_colors(args.colorscheme,len(hists));
            bold    = style.get_colors(args.colorscheme,len(hists));
            light   = style.get_colors(args.colorscheme,len(hists));
            
            ndata = style.get_colors("neutral");
            bdata = style.get_colors("bold");   
            ldata = style.get_colors("light");  
            for i,hist in enumerate(stack):
                if "bold" == hist.emph:
                    hist.color = bold[i%len(bold)]
                elif "light" == hist.emph:
                    hist.color = light[i%len(light)]
                else:
                    hist.color = neutral[i%len(neutral)]
            for i,hist in enumerate(data):
                hist.color = bdata[i%len(bold)]
        else:
            neutral = style.get_colors("neutral");
            bold    = style.get_colors("bold");
            light   = style.get_colors("light");
            if len(data) == 0:
                neutral = neutral[1:]
                bold = bold[1:]
                light = light[1:]
            for i,hist in enumerate(all_hists):
                if hist in data:
                    hist.color = bold[i%len(bold)]
                elif "bold" == hist.emph:
                    hist.color = bold[i%len(bold)]
                elif "light" == hist.emph:
                    hist.color = light[i%len(light)]
                else:
                    hist.color = neutral[i%len(neutral)]

    hs = HistStack(*stack)
    fig = plt.figure()

    if args.ratio:
        gs = gridspec.GridSpec(2,1,height_ratios=[3,1])
        ax = plt.subplot(gs[0])
        ax_ratio = plt.subplot(gs[1])
        sum_MC = hs.sum_hist()
        rhists = [hist.divide(sum_MC) for hist in data]
        for rhist in rhists:
            rhist.plot_lines(ax_ratio)
        plt.subplots_adjust(hspace=0)
        plt.setp(ax.get_xticklabels(), visible=False)
        setup_ratio(args, ax, ax_ratio)
    else:
        ax = fig.add_subplot(111)

    if args.alpha:
        for hist in hists:
            hist.options['alpha'] = args.alpha

    if args.noerror:
        for hist in data:
            hist.plot_noerror(ax)
    elif args.points:
        for hist in data:
            hist.plot_points(ax)
    else:
        for hist in data:
            hist.plot_lines(ax)
    hs.plot_bars(ax, log = args.logy)

    fig.subplots_adjust(bottom=.12, left=.14)
    plot_lines(args, ax, neutral, len(all_hists), hists)
    setup_figure(args, ax)
    ax.legend(frameon=False, loc=args.loc)

    if "." not in args.name:
        fig.savefig(args.outDir+args.name+".pdf", transparent=args.transparent)
        print "Saving figure: %s.pdf" % args.outDir+args.name
    else:
        fig.savefig(args.outDir+args.name, transparent=args.transparent)
        print "Saving figure: %s" % args.outDir+args.name
    return


def color2D(hists, args):
    """ Plot a 2D histogram as a colorbar graph. """
    if len(hists) > 1:
        print "color2D can only plot 1 2D histogram."
        return
    hist = hists[0]
    if args.logz:
        hist.options['norm'] = LogNorm(vmin=args.zmin, vmax=args.zmax)
    if args.zmax:
        hist.options['vmax'] = args.zmax
    if args.zmin:
        hist.options['vmin'] = args.zmin
    hist.color = args.colorscheme
    fig,ax = plt.subplots()
    setup_figure(args, ax)

    hist.plot_colors()
    cb = plt.colorbar()
    if args.zlabel:
        cb.set_label(args.zlabel)
    cb.update_ticks()

    
    if "." not in args.name:
        fig.savefig(args.outDir+args.name+".pdf", transparent=args.transparent)
        print "Saving figure: %s.pdf" % args.outDir+args.name
    else:
        fig.savefig(args.outDir+args.name, transparent=args.transparent)
        print "Saving figure: %s" % args.outDir+args.name
    return


# The following functions define argument types and return 
# argparse specific errors if the argument provided is not adaptable.

def opt_dict(string):
    opts = {}
    for (i,val) in enumerate(string.split(",")):
        if i == 0:
            try:
                opts["pos"] = int(val)
            except ValueError:
                raise argparse.ArgumentTypeError("First entry for an --hopts argument should be the position of the hist.")
        elif "=" not in val:
            raise argparse.ArgumentTypeError("Options should have the form opt=val.")
        elif val.split("=")[0] not in hopts.keys():
            raise argparse.ArgumentTypeError("Invalid histogram option: %s" % val.split("=")[0])
        else:
            opts[val.split("=")[0]] = hopts[val.split("=")[0]](val.split("=")[1])
    return opts


def list_arg(string,type):
    return [type(x) for x in string.split(",")]


def interval_arg(string):
    msg = 'Invalid argument for interval, use x:y e.g. 1.23:3.21'
    if ':' not in string:
        raise argparse.ArgumentTypeError(msg)
    try:
        x = float(string.split(":")[0])
        y = float(string.split(":")[1])
    except ValueError:
        raise argparse.ArgumentTypeError(msg)
    return x,y


def interval_arg_2D(string):
    msg = 'Invalid argument for interval, use x1:x2,y1:y2 e.g. 1.23:3.21,1.01:1.02'
    if ':' not in string:
        raise argparse.ArgumentTypeError(msg)
    if ',' not in string:
        raise argparse.ArgumentTypeError(msg)
    try:
        xs = string.split(",")[0]
        ys = string.split(",")[1]
        x1 = float(xs.split(":")[0])
        x2 = float(xs.split(":")[1])
        y1 = float(ys.split(":")[0])
        y2 = float(ys.split(":")[1])
    except ValueError:
        raise argparse.ArgumentTypeError(msg)
    return x1,x2,y1,y2


def emph_str(string):
    msg = 'Invalid argument for emphasis, choose from light,neutral,bold'
    if string.lower() not in ['bold','neutral','light']:
        raise argparse.ArgumentTypeError(msg)
    return string.lower()


def rep_arg(string):
    msg = 'Invalid argument for replacement, use a=b'
    if '=' not in string:
        raise argparse.ArgumentTypeError(msg)
    a = string.split("=")[0]
    b = string.split("=")[1]
    return a,b


def bool_arg(string):
    msg = 'Unrecognized option for True/False.'
    if string.lower() in ['true','t','yes','1']:
        return True
    elif string.lower() in ['false','f','no','0']:
        return False
    else:
        raise argparse.ArgumentTypeError(msg)


def line_arg(string):
    msg = 'Unrecognized option for line argument, use height,label.'
    if ',' in string:
        height = float(string.split(',')[0])
        label  = string.split(',')[1]
        return height,label
    else:
        raise argparse.ArgumentTypeError(msg)

def fit_arg(string):
    msg = 'Unrecognized option for fit argument, use function_name(,format_string).\n The choices for fits are %s, and the optional format string \n should be a python format string for the fit parameters.' % (str(fits.keys()))
    if ',' in string:
        key, format = string.split(',')
    else:
        key = string
        format = ''
    if key not in fits.keys():
        raise argparse.ArgumentTypeError(msg)
    return key, format

def color_arg(string):
    msg = 'Unrecognized option for color argument (%s). Use any string that can be passed to matplotlib as a color, such as "#FFFFFF" or "black". Can also use "colorscheme,number_in_group,position_in_group".' % string
    if string.count(',') == 2:
        scheme, n, i = string.split(',')
        try:
            n, i = int(n), int(i)
        except ValueError:
            raise argparse.ArgumentTypeError(msg)
        return style.get_colors(scheme,n)[i]
    else:
        try:
            tmp = ListedColormap([string,string])
            tmp(0)
        except ValueError:
            raise argparse.ArgumentTypeError(msg)
        return string

        

# This dictionary contains all options which can be applied to individual histograms, and the 
# accompanying argument type. These are handled by get_hists.    

hopts = {'rebin':int,'projectionx':interval_arg,'projectiony':interval_arg,'projectionx':interval_arg,'projectiony':interval_arg,'averagex':interval_arg,'averagey':interval_arg,'exclude':interval_arg,'emph':emph_str,'norm':float,'scale':float,'fill':bool_arg,'error':bool_arg}


def gen_parser(add_hgroup):
    """ Build and return the default argument parser. 

    Notes:
    add_hgroup = True/False specifies whether to add the
    ROOT histogram options, which are only available when
    using the default get_hists.
    """
    commonArgs = argparse.ArgumentParser(add_help=False)
    # Input and Output options
    iogroup = commonArgs.add_argument_group('Specify Input and Output')
    iogroup.add_argument('name', help='The savename for the generated plot.')
    iogroup.add_argument('targets', help='Specify histograms to add to the plot.', nargs='*')
    iogroup.add_argument('-i', '--input',  dest='inDir',  help='Input directory with .root files.',default='')
    iogroup.add_argument('-o', '--output', dest='outDir', help='Output directory for saved plots.',default='')
    iogroup.add_argument('-l', '--labels', dest='labels', help='Labels to use in the plot legend.',nargs='*')
    iogroup.add_argument('-c', '--colors', dest='colors', help='Specific colors for each histogram.',nargs='*', type=color_arg)
    
    # Canvas based options
    cgroup = commonArgs.add_argument_group('Specify Canvas Options')
    cgroup.add_argument('--title',  dest='title',  help='Title to add to plot.',default='')
    cgroup.add_argument('--logy',    dest='logy',    help='Set log for Y axis.', action="store_true")
    cgroup.add_argument('--logx',    dest='logx',    help='Set log for X axis.', action="store_true")
    cgroup.add_argument('--colorscheme', dest='colorscheme', help='Override default color scheme.',
                            choices=["BrBG","RdBu","Blues","Greens","Oranges"])
    cgroup.add_argument('--fit', dest='fit', help='Fit each histogram to the specified function.', type=fit_arg)
    cgroup.add_argument('--loc',    dest='loc',    help='Location of the plot legend.', default='best')
    cgroup.add_argument('--max',    dest='max',    help='Upper bound for plot.', type=float)
    cgroup.add_argument('--min',    dest='min',    help='Lower bound for plot.', type=float)
    cgroup.add_argument('--alpha',  dest='alpha',  help='Set transparency for points/lines on plot.', type=float)
    cgroup.add_argument('--limits', dest='limits', help='Override X limits for plot.',type=interval_arg)
    cgroup.add_argument('--xlabel', dest='xlabel', help='Override label for x-axis.')
    cgroup.add_argument('--ylabel', dest='ylabel', help='Override label for y-axis.')
    cgroup.add_argument('--yields', dest='yields', help='Write yields for each histogram.', action="store_true")
    cgroup.add_argument('--line',   dest='line',   help='Place a horizontal line at the specified value.', type=line_arg)
    cgroup.add_argument('--box',   dest='box',   help='Place a box which streches across the x range and has y values given by y1:y2.', type=interval_arg)
    cgroup.add_argument('--avline', dest='avline', help='Place a horizontal line at the average of the specified histogram.', type=int)
    cgroup.add_argument('--noerror', dest='noerror', help='Plot histograms as lines without errors.', action="store_true")
    cgroup.add_argument('--points', dest='points', help='Plot histograms as points on centers.', action="store_true")
    cgroup.add_argument('--transparent', dest='transparent', help='Save plots with transparent background.', action="store_true")
    

    if add_hgroup:
        hgroup = commonArgs.add_argument_group('Specify Single Histogram Options')
        # Histogram Based Options, Local
        hgroup.add_argument('--hopts', dest='options', help='A comma separated list of keyworded options for a specified histogram. Specify the histogram by its position in the list (starting from 0). Valid options are the same as the global histogram options below. Ex: 0,rebin=2,exclude=0:0',nargs='*',type=opt_dict, default={})

        hgroupg = commonArgs.add_argument_group('Specify Global Histogram Options')
        # Histogram Based Options, Global
        hgroupg.add_argument('--rebin',      dest='rebin',   help='Number of bins to combine in rebin', type=int)
        hgroupg.add_argument('--norm',       dest='norm',   help='Renormalize histogram to value.', type=float)
        hgroupg.add_argument('--scale',      dest='scale',   help='Scale histogram by value.', type=float)
        hgroupg.add_argument('--projectionx',dest='projectionx',    help='Projection to take from a 2D histogram.',type=interval_arg)
        hgroupg.add_argument('--projectiony',dest='projectiony',    help='Projection to take from a 2D histogram.',type=interval_arg)
        hgroupg.add_argument('--averagex',   dest='averagex',    help='Error weighted average over the specified interval on a 2D histogram.',type=interval_arg)
        hgroupg.add_argument('--averagey',   dest='averagey',    help='Error weighted average over the specified interval on a 2D histogram.',type=interval_arg)
        hgroupg.add_argument('--exclude',    dest='exclude', help='Do not plot a range of the histograms. Specify as start:end.',type=interval_arg)
        hgroupg.add_argument('--exclude2D',  dest='exclude2D', help='Do not plot a range of the histograms. Specify as start:end,start:end for x,y.',type=interval_arg_2D)
        hgroupg.add_argument('--emph',       dest='emph',    help='Specify emphasis for histograms. Choose from bold,neutral,light',type=emph_str)
        hgroupg.add_argument('--error',      dest='error',   help='Plot the errors on the histogram as the central values.',type=bool_arg)

    # Ratio arguments, currently for stack and compare
    ratioArgs = argparse.ArgumentParser(add_help=False)
    rgroup = ratioArgs.add_argument_group('Specify Ratio Subplot Options')
    rgroup.add_argument('--ratio',  dest='ratio', help='Draw a ratio subplot.', action="store_true")
    rgroup.add_argument('--rlabel', dest='rlabel', help='Ratio axis label.', default = 'Ratio')
    rgroup.add_argument('--rmin',   dest='rmin', help='Ratio axis minimum.', type=float)
    rgroup.add_argument('--rmax',   dest='rmax', help='Ratio axis maximum.', type=float)
    
    usage = "quickPlot.py %s targets %s[--labels] [canvas options] [histogram options]\nTargets can be a single file and a list of histograms or a list of files and a single histogram.\nFor example:\n    file1.root hist1 hist2 hist3\n    hist1 file1.root file2.root file3.root\n\nFor the more general case, targets can be specified as: \n    file1.root hist1 hist2 file2.root hist3 hist4 ..."

    parser = argparse.ArgumentParser(description='Quick plotting utility for ROOT histograms.')
    subparsers = parser.add_subparsers(title='Plot Templates',dest='sub')

    pStack = subparsers.add_parser('stack',help='Stack a set of histograms and (optional) compare to data.',parents=[commonArgs,ratioArgs],usage=(usage % ("stack","[--data] ")))
    pStack.add_argument('-d', '--data', dest='data', help='Plot the specified hists as data, separate from the stack. Specify by position in the list, starting from 0.', nargs='*', type=int,default=[])
    pStack.set_defaults(func=stack)

    pCompare = subparsers.add_parser('compare',help='Compare a set of histograms by plotting side-by-side.',parents=[commonArgs,ratioArgs],usage=(usage % ("compare","")))
    pCompare.add_argument('--total', dest='total',     help='Add a histogram which is the total of the others.', action="store_true")
    pCompare.add_argument('--totalsq', dest='totalsq',     help='Add a histogram which is the total (sum of squares) of the others.', action="store_true")
    pCompare.set_defaults(func=compare)
    
    pColor2D = subparsers.add_parser('color2D',help='Plot one 2D histogram using a color plot.',parents=[commonArgs],usage=(usage % ("compare2D","")))
    pColor2D.add_argument('--logz', dest='logz', help='Set log for Z axis.', action='store_true')
    pColor2D.add_argument('--zmax',    dest='zmax',    help='Upper bound for plot.', type=float)
    pColor2D.add_argument('--zmin',    dest='zmin',    help='Lower bound for plot.', type=float)
    pColor2D.add_argument('--zlabel', dest='zlabel', help='Override label for z-axis.')
    pColor2D.set_defaults(func=color2D)

    usage = 'quickPlot.py batch filename\nGenerates a batch of plots using each line of the file as a separate set of arguments.' 
    pBatch = subparsers.add_parser('batch',help='Run a batch of plots from an input file, each line should contain a full command.',usage=usage)
    pBatch.add_argument('file', help='The file with the batch commands.')
    pBatch.add_argument('--replace', dest='rep', type=rep_arg, help='A list of a=b to replace a with b in the file.',nargs='*',default=[])
    return parser


def main(get_hists=get_hists, add_hgroup=True):
    """ Generate plots from command line arguments.
    
    Notes: 
    More detailed information about what this does is available in the
    command line help functions.
    Override the defaults to make the program work with other types of
    input. You should most likely use add_hgroup=False if you specify
    a custom get_hists. (The handling of the add_hgroup arguments is
    done in the get_hists function.)
    """
    parser = gen_parser(add_hgroup)
    args = parser.parse_args()

    if args.sub != "batch":
        hists = get_hists(args)
        args.func(hists, args)
    else:
        bFile = open(args.file, 'r')
        for line in bFile:
            for a,b in args.rep:
                line = line.replace(a,b)
            if line.strip() == "":
                continue
            if line.strip()[0] == '#' or line.strip()[:1] == "//":
                continue
            if 'quickplot' in line.split(" ")[0].lower():
                input_args = shlex.split(line)[1:]
            else:
                input_args = shlex.split(line)
            try:
                new_args = parser.parse_args(input_args)
                hists = get_hists(new_args)
                new_args.func(hists, new_args)
            except PlotError:
                continue
            except ValueError:
                continue
            except argparse.ArgumentTypeError, err:
                print err
                continue


if __name__ == "__main__":
    main()
