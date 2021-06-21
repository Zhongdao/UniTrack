#!/usr/bin/env python
# coding=utf-8
import pdb
import matplotlib 
from matplotlib import pyplot as plt
import pandas as pds
import seaborn as sns; sns.set()

sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
df = pds.read_csv('./sde.csv')
#cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)
cmap = sns.color_palette("hls", 4)
df["MOTA"] /= 1000
df["MOTA19"] /=1000
f, (ax1,ax2) = plt.subplots(1, 2, figsize=(7.5, 3.5))
sns.scatterplot(x="FPS", y="MOTA", hue='Embedding',style='Detection', data=df, palette=cmap,  legend='brief', s=80,
        markers=['P', 'v', 'X', 'd', '^','p', 's'], ax=ax1)
sns.scatterplot(x="FPS19", y="MOTA19", hue='Embedding',style='Detection', data=df, palette=cmap,  legend='brief', s=80,
        markers=['P', 'v', 'X', 'd', '^','p', 's'], ax=ax2)
ax1.set_xlim(0, 25)
ax1.set_ylim(0.45,0.7)
ax1.grid()
ax2.set_xlim(0, 25)
ax2.set_ylim(0.32,0.48)
ax2.grid()

l1 = ax1.legend(labelspacing=0.1)
l2 = ax2.legend(labelspacing=0.1)
l1.texts[0].set_fontsize(8)
l1.texts[0].set_position((-30,0))
l1.texts[5].set_position((-30,0))
l2.texts[0].set_fontsize(8)
l2.texts[0].set_position((-30,0))
l2.texts[5].set_position((-30,0))
ax1.set_xlabel('(a) FPS@usual case')
ax1.set_ylabel('MOTA')

ax2.set_xlabel('(b) FPS@crowd case')
ax2.set_ylabel('MOTA')
#
ax1.vlines(11,0.45,0.7, colors='gray', linestyles='dashed', linewidth=2)
ax1.hlines(0.648,0,25, colors='gray', linestyles='dashed', linewidth=2)
ax2.vlines(7,0.32,0.48, colors='gray', linestyles='dashed', linewidth=2)
ax2.hlines(0.43,0,25, colors='gray', linestyles='dashed', linewidth=2)
plt.show()

