import pix
import sys
from numpy import histogram, std
import quickplot as qp
from itertools import izip

def get_hists(args):
    files, hnames = qp.expand_targets(args.targets, suffix='.csv')
    if not args.labels:
        labels = [None for x in files]
    else:
        labels = args.labels
        if len(labels) != len(hnames):
            print "The number of labels doesn't match the number of hists."
            sys.exit(0)
    # No histogram specific options for now.
    hists = []
    for fname,hname,label in izip(files,hnames,labels):
        try:
            pixels = pix.PixelLibrary.from_file(fname)
            if args.sub in ['color2D']:
                key = hname
                if key == 'thresh':
                    values = pixels.get_thresh_grid()
                else:
                    values = pixels.get_data_grid(key)
                if len(values) < 18:
                    ybins = range(1,len(values)+2)
                else:
                    ybins = range(19)
                hists.append(qp.Hist2D(range(65),ybins, values))
            elif ',' in hname:
                key, col_index = hname.split(',')
                col_index = int(col_index)
                if key == 'thresh':
                    heights = pixels.get_thresh_col(col_index)
                else:
                    heights = pixels.get_data_col(key, col_index)                    
                hists.append(qp.Hist(range(len(heights)),[1 for i in xrange(len(heights))], heights, label=label))
            else:
                key = hname
                if key == 'thresh':
                    values = pixels.get_thresh_all()
                else:
                    values = pixels.get_data_all(key)
                dev = std(values)
                if label is None:
                    label = '$\sigma = \mathsf{%i}$' % dev
                else:
                    label += ', $\sigma = \mathsf{%i}$' % dev
                heights, bin_edges = histogram(values, bins=50, range=args.limits)
                hists.append(qp.Hist.from_bin_edges(bin_edges, heights, label=label))
                

        except ValueError:
            print "Invalid histogram specification: %s %s %s" % (fname, hname, label)
            sys.exit(0)
    return hists


if __name__ == "__main__":
    qp.main(get_hists, False)
