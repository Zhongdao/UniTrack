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
cmap = sb.color_palette('hls', 14)
cmap = np.random.permutation(cmap)
# Set data

#kk = 'Segmentation'
#dd = 'Seg. mean IOU'
#xtick = [0.26, 0.28, 0.30]
#xlim = [0.25, 0.30]

#kk = 'VOC (Frozen)'
#dd = 'Det mAP (Fronzen)'
#xtick = [0.50, 0.52, 0.54]
#xlim = [0.49, 0.55]


#kk = 'VOC (Finetune)'
#dd = 'Det mAP (Finetune)'
#xtick = [0.46, 0.48, 0.50, 0.52, 0.54]
#xlim = [0.44, 0.55]


kk = 'Surface Normal'
dd = '(-) Surf. Norm. Mean Err.'
xtick = [0.28, 0.32, 0.36, 0.40]
xlim = [0.28, 0.40]

xtick = [-40, -36, -32, -28]
xlim = [-40, -28]

df_ = pd.read_csv('ssl_others.csv')
scol = ['method', kk ,'DCF','VOS','MOTIDF','MOTSIDF','PTIDF']

df = df_[scol]
#df = df.iloc[:-1]
for sc in scol[1:]:
    df[sc] /= 1000.
df[kk] /= 10.
df['Surface Normal'] *= -100 
df1 = df.iloc[:-2]
df2 = df.iloc[-2:]

rmatrix = df[scol[1:]].corr(method='pearson')
rhomatrix = df[scol[1:]].corr(method='spearman')
print(rmatrix)
print(rhomatrix)
#df[scol[1:]] = df[scol[1:]].apply(lambda x: x * 1.0/(1000-x))

#df[scol[1:]] = df[scol[1:]].apply(np.log)
#df[scol[1:]] = df[scol[1:]].apply(lambda x: x.astype(np.float))
#print(df[scol[1:]].corr(method='pearson'))

ylims = [[0.58,0.64], [0.58,0.63], [0.68,0.76], [0.66,0.71], [0.72,0.75]]
titles = ['SOT', 'VOS', 'MOT', 'MOTS', 'PoseTrack']
markers=['P', 'v', 'X', 'd', '^','p', '<', '>','d','*',',','D','x','_']
fig, axs = plt.subplots(ncols=5, figsize=(14,2.5), sharex=True)
for i in range(2, len(scol)):
    sb.scatterplot(x=kk, y=scol[i], style='method', hue='method', palette=cmap[:-2], data=df1,markers=markers[:-2], s=100,ax=axs[i-2] )
    sb.scatterplot(x=kk, y=scol[i], style='method', hue='method', palette=cmap[-2:], data=df2,markers=markers[-2:], s=100,ax=axs[i-2], linewidth=2)
    sb.regplot(x=kk, y=scol[i], data=df, ax=axs[i-2], scatter_kws={'s':0.0})
    axs[i-2].legend_.remove()
    axs[i-2].set_ylim(ylims[i-2])
    axs[i-2].set_ylabel('')
    axs[i-2].grid()
    axs[i-2].set_xlabel(dd)
    axs[i-2].yaxis.set_major_formatter(ScalarFormatterNoLeadingZero())
    axs[i-2].set_xlim(xlim)
    axs[i-2].set_xticks(xtick)
    axs[i-2].set_title(titles[i-2])
    xbeg,xend = axs[i-2].get_xlim()
    ybeg,yend = ylims[i-2]
    axs[i-2].text(xbeg + (xend-xbeg)*0.05 , ybeg + (yend-ybeg)*0.05, \
            r'$r$ = {:0.2f}'.format(rmatrix[kk][scol[i]]), color='black', size=10, usetex=True)
    axs[i-2].text(xbeg + (xend-xbeg)*0.05 , ybeg + (yend-ybeg)*0.15, \
            r'$\rho$ = {:0.2f}'.format(rhomatrix[kk][scol[i]]), color='black', size=10, usetex=True)
    if i == 6:
        axs[i-2].set_yticks([0.72,0.73,0.74,0.75])
    if i == 2:
        axs[i-2].set_ylabel('AUC/J-mean/IDF-1')
lll = axs[-1].legend(loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0, ncol=2, fontsize='small')
plt.subplots_adjust(right=0.7, left=0.05, wspace=0.2, top=0.8, bottom=0.2, hspace=0.1)
plt.savefig('/home/wangzd/sur.pdf')
plt.show()


