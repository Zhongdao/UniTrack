###################################################################
# File Name: setup.py
# Author: Zhongdao Wang
# mail: wcd17@mails.tsinghua.edu.cn
# Created Time: Wed Jul  7 20:23:56 2021
###################################################################

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os

if __name__ == '__main__':
    os.chdir('./tracker/sot/lib/eval_toolkit/pysot/utils/')
    os.system('python setup.py build_ext --inplace')
