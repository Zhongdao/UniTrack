###################################################################
# File Name: radar.py
# Author: Zhongdao Wang
# mail: wcd17@mails.tsinghua.edu.cn
# Created Time: Tue Apr 27 18:30:24 2021
###################################################################

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

# Libraries
import random
import pandas as pd
import math
from math import pi
import pdb
import seaborn as sb; sb.set()
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
matplotlib.rc('xtick',labelsize=8)
matplotlib.rc('ytick',labelsize=8)
matplotlib.rc('axes',labelsize=10)
matplotlib.rc('axes',titlesize=12)
class ScalarFormatterNoLeadingZero(ScalarFormatter):
    def pprint_val(self, x):
        s = ScalarFormatter.pprint_val(self, x)
        return s.replace("0.",".")

sb.set_style('ticks', {'xtick.major.size':8, 'ytick.major.size':8})
cmap = sb.color_palette('hls', 16)
cmap = np.random.permutation(cmap)
# Set data

df_ = pd.read_csv('ssl.csv')

scol = ['method','IMAGENET','DCF','VOS','MOTIDF','MOTSIDF','PTIDF']

df = df_[scol]
df = df.iloc[:-1]
for sc in scol[1:]:
    df[sc] /= 1000.
df1 = df.iloc[:-4]
df2 = df.iloc[-4:]

rmatrix = df[scol[1:]].corr(method='pearson')
rhomatrix = df[scol[1:]].corr(method='spearman')
print(rmatrix)
print(rhomatrix)
#df[scol[1:]] = df[scol[1:]].apply(lambda x: x * 1.0/(1000-x))

#df[scol[1:]] = df[scol[1:]].apply(np.log)
#df[scol[1:]] = df[scol[1:]].apply(lambda x: x.astype(np.float))
#print(df[scol[1:]].corr(method='pearson'))

ylims = [[0.58,0.64], [0.58,0.63], [0.68,0.76], [0.66,0.71], [0.72,0.75]]
xtick = [0.6, 0.65, 0.7, 0.75]
titles = ['SOT', 'VOS', 'MOT', 'MOTS', 'PoseTrack']
markers=['P', 'v', 'X', 'd', '^','p', '<', '>','d','*',',','D','x','_','3','|']
fig, axs = plt.subplots(ncols=5, figsize=(14,2), sharex=True)
for i in range(2, len(scol)):
    sb.scatterplot(x='IMAGENET', y=scol[i], style='method', hue='method', palette=cmap[:-4], data=df1,markers=markers[:-4], s=100,ax=axs[i-2] )
    sb.scatterplot(x='IMAGENET', y=scol[i], style='method', hue='method', palette=cmap[-4:], data=df2,markers=markers[-4:], s=100,ax=axs[i-2], linewidth=2)
    sb.regplot(x='IMAGENET', y=scol[i], data=df, ax=axs[i-2], scatter_kws={'s':0.0})
    axs[i-2].legend_.remove()
    axs[i-2].set_ylim(ylims[i-2])
    axs[i-2].set_ylabel('')
    axs[i-2].grid()
    axs[i-2].set_xlabel('ImageNet Top-1 Acc.')
    axs[i-2].yaxis.set_major_formatter(ScalarFormatterNoLeadingZero())
    axs[i-2].set_xticks(xtick)
    axs[i-2].set_title(titles[i-2])
    xbeg,xend = axs[i-2].get_xlim()
    ybeg,yend = ylims[i-2]
    axs[i-2].text(xbeg + (xend-xbeg)*0.05 , ybeg + (yend-ybeg)*0.05, \
            r'$r$ = {:0.2f}'.format(rmatrix['IMAGENET'][scol[i]]), color='black', size=10, usetex=True)
    axs[i-2].text(xbeg + (xend-xbeg)*0.05 , ybeg + (yend-ybeg)*0.15, \
            r'$\rho$ = {:0.2f}'.format(rhomatrix['IMAGENET'][scol[i]]), color='black', size=10, usetex=True)
    if i == 6:
        axs[i-2].set_yticks([0.72,0.73,0.74,0.75])
    if i == 2:
        axs[i-2].set_ylabel('AUC/J-mean/IDF-1')
lll = axs[-1].legend(loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0, ncol=2, fontsize='small')
plt.subplots_adjust(right=0.7, left=0.05, wspace=0.2, top=0.8, bottom=0.2, hspace=0.1)
plt.show()



