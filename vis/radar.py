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
import matplotlib.pyplot as plt
import pandas as pd
from math import pi,cos,sin
import pdb
 
# Set data

df_ = pd.read_excel('ssl.xlsx')

scol = ['method','DCF','VOS','MOTIDF','PTIDF','MOTSIDF']
df = df_[scol]
for sc in scol[1:]:
    sortd = df[sc].argsort()
    for i,idx in enumerate(sortd):
        df[sc][idx] = i


# ------- PART 1: Define a function that do a plot for one line of the dataset!
 
def make_spider( row, title, color):

    # number of variable
    categories=list(df)[1:]
    categories=['SOT','VOS','MOT','PoseT', 'MOTS']
    N = len(categories)

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    ax = plt.subplot(2,8,row, polar=True, )

    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], categories, color='grey', size=0)
    rots = [0, 288,36, -36, 72]
    for i,c in enumerate(categories):
        p = 18 if i in [2,3] else 17
        ax.text(angles[i], p, c,horizontalalignment='center',verticalalignment='center', rotation=rots[i], color='grey', size=12)

    # Draw ylabels
    ax.set_rlabel_position(5)
    plt.yticks([5,10,15,20],[])
    plt.ylim(0,20)

    values=df.loc[0].drop('method').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, color='gray', linewidth=2, linestyle='dashed')
    
    # Ind1
    values=df.loc[row].drop('method').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, color=color, linewidth=2, linestyle='solid')
    ax.fill(angles, values, color=color, alpha=0.4)




    # Add a title
    plt.title(title, size=20, color=color, y=1.1)

    
# ------- PART 2: Apply the function to all individuals
# initialize the figure
my_dpi=96
plt.figure(figsize=(1000/my_dpi, 1000/my_dpi), dpi=my_dpi)
 
# Create a color palette:
my_palette = plt.cm.get_cmap("tab10", 10)
 
# Loop to plot
for row in range(1, len(df.index)):
    make_spider( row=row, title=df['method'][row], color=my_palette(row%10))
plt.show()
