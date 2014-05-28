#!/usr/bin/env python
"""
Color lists for the lines/points on plots.
"""

import matplotlib

def_grey = '#737373'
#             grey        red       green      blue       orange      
neutral  = ['#737373', '#FF3F5D', '#5BCB61', '#4999D8', '#FFA44E', '#A95AAE', '#DC6753', '#E670B6']
bold     = ['#020202', '#FF0025', '#009441', '#0253AD', '#FF7300', '#710095', '#B0011B', '#C50097']
light    = ['#CCCCCC', '#FDA9AC', '#D5E8A5', '#B3D1EE', '#F9D1AD', '#DDACD6', '#E3B7A7', '#F4BADB']

brownblue = [['#D8B365', '#F5F5F5', '#5AB4AC'],
                ['#A6611A', '#DFC27D', '#80CDC1', '#018571'],
                ['#A6611A', '#DFC27D', '#F5F5F5', '#80CDC1', '#018571'],
                ['#8C510A', '#D8B365', '#F6E8C3', '#C7EAE5', '#5AB4AC', '#01665E'],
                ['#8C510A', '#D8B365', '#F6E8C3', '#F5F5F5', '#C7EAE5', '#5AB4AC', '#01665E'],
                ['#8C510A', '#BF812D', '#DFC27D', '#F6E8C3', '#C7EAE5', '#80CDC1', '#35978F', '#01665E'],
                ['#8C510A', '#BF812D', '#DFC27D', '#F6E8C3', '#F5F5F5', '#C7EAE5', '#80CDC1', '#35978F', '#01665E'],
                ['#543005', '#8C510A', '#BF812D', '#DFC27D', '#F6E8C3', '#C7EAE5', '#80CDC1', '#35978F', '#01665E', '#003C30'],
                ['#543005', '#8C510A', '#BF812D', '#DFC27D', '#F6E8C3', '#F5F5F5', '#C7EAE5', '#80CDC1', '#35978F', '#01665E', '#003C30']]

redblue   = [['#EF8A62', '#F7F7F7', '#67A9CF'],
                ['#CA0020', '#F4A582', '#92C5DE', '#0571B0'],
                ['#CA0020', '#F4A582', '#F7F7F7', '#92C5DE', '#0571B0'],
                ['#B2182B', '#EF8A62', '#FDDBC7', '#D1E5F0', '#67A9CF', '#2166AC'],
                ['#B2182B', '#EF8A62', '#FDDBC7', '#F7F7F7', '#D1E5F0', '#67A9CF', '#2166AC'],
                ['#B2182B', '#D6604D', '#F4A582', '#FDDBC7', '#D1E5F0', '#92C5DE', '#4393C3', '#2166AC'],
                ['#B2182B', '#D6604D', '#F4A582', '#FDDBC7', '#F7F7F7', '#D1E5F0', '#92C5DE', '#4393C3', '#2166AC'],
                ['#67001F', '#B2182B', '#D6604D', '#F4A582', '#FDDBC7', '#D1E5F0', '#92C5DE', '#4393C3', '#2166AC', '#053061'],
                ['#67001F', '#B2182B', '#D6604D', '#F4A582', '#FDDBC7', '#F7F7F7', '#D1E5F0', '#92C5DE', '#4393C3', '#2166AC', '#053061']]

blues     = [['#DEEBF7', '#9ECAE1', '#3182BD'],
                ['#EFF3FF', '#BDD7E7', '#6BAED6', '#2171B5'],
                ['#EFF3FF', '#BDD7E7', '#6BAED6', '#3182BD', '#08519C'],
                ['#EFF3FF', '#C6DBEF', '#9ECAE1', '#6BAED6', '#3182BD', '#08519C',],
                ['#EFF3FF', '#C6DBEF', '#9ECAE1', '#6BAED6', '#4292C6', '#2171B5', '#084594'],
                ['#F7FBFF', '#DEEBF7', '#C6DBEF', '#9ECAE1', '#6BAED6', '#4292C6', '#2171B5', '#084594'],
                ['#F7FBFF', '#DEEBF7', '#C6DBEF', '#9ECAE1', '#6BAED6', '#4292C6', '#2171B5', '#08519C', '#08306B']]

greens    = [['#E5F5E0', '#A1D99B', '#31A354'],
                ['#EDF8E9', '#BAE4B3', '#74C476', '#238B45'],
                ['#EDF8E9', '#BAE4B3', '#74C476', '#31A354', '#006D2C'],
                ['#EDF8E9', '#C7E9C0', '#A1D99B', '#74C476', '#31A354', '#006D2C'],
                ['#EDF8E9', '#C7E9C0', '#A1D99B', '#74C476', '#41AB5D', '#238B45', '#005A32'],
                ['#F7FCF5', '#E5F5E0', '#C7E9C0', '#A1D99B', '#74C476', '#41AB5D', '#238B45', '#005A32'],
                ['#F7FCF5', '#E5F5E0', '#C7E9C0', '#A1D99B', '#74C476', '#41AB5D', '#238B45', '#006D2C', '#00441B']]

oranges   = [['#FEE6CE', '#FDAE6B', '#E6550D'],
                ['#FEEDDE', '#FDBE85', '#FD8D3C', '#D94701'],
                ['#FEEDDE', '#FDBE85', '#FD8D3C', '#E6550D', '#A63603'],
                ['#FEEDDE', '#FDD0A2', '#FDAE6B', '#FD8D3C', '#E6550D', '#A63603'],
                ['#FEEDDE', '#FDD0A2', '#FDAE6B', '#FD8D3C', '#F16913', '#D94801', '#8C2D04'],
                ['#FFF5EB', '#FEE6CE', '#FDD0A2', '#FDAE6B', '#FD8D3C', '#F16913', '#D94801', '#8C2D04'],
                ['#FFF5EB', '#FEE6CE', '#FDD0A2', '#FDAE6B', '#FD8D3C', '#F16913', '#D94801', '#A63603', '#7F2704']]

colorlists = {"neutral":neutral,"light":light, "bold":bold, "oranges": oranges,
              "greens":greens, "blues":blues, "brbg":brownblue, "rdbu":redblue}


def get_colors(name, n=3):
    if name.lower() in ["neutral","light","bold"]:
        return colorlists[name.lower()]
    else:
        return colorlists[name.lower()][n-3]

matplotlib.rcParams['backend'] = 'Agg'
#matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.size'] = 20
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Helvetica'

matplotlib.rcParams['axes.edgecolor'    ] = '#737373'
matplotlib.rcParams['grid.color'        ] = '#B3B3B3'
matplotlib.rcParams['grid.linestyle'    ] = '--'
matplotlib.rcParams['grid.linewidth'    ] = 2
matplotlib.rcParams['ytick.major.size'  ] = 8	     
matplotlib.rcParams['ytick.minor.size'  ] = 4	     
matplotlib.rcParams['xtick.major.size'  ] = 8	     
matplotlib.rcParams['xtick.minor.size'  ] = 4

matplotlib.rcParams['lines.linewidth'] = 2.0
matplotlib.rcParams['lines.markersize'] = 4.0
matplotlib.rcParams['legend.numpoints'] = 1

#matplotlib.rcParams['text.latex.preamble'] ='\usepackage{sfmath}'
