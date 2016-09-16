"""
mix images in all classes to a single folder and generate corresponding label
Author: qi liu
"""

import os,glob,re
import sys
import csv
import pandas as pd

inpath = '/home/mirabot/Documents/deeplearning/zoophoto/zoosorted/'
outpath = '/home/mirabot/Documents/deeplearning/zoophoto/zoosorted/mixed/'
subdir = [dI for dI in os.listdir(inpath) if os.path.isdir(os.path.join(inpath,dI))]

nfile = 0

# create new directory for all files
os.mkdir(outpath, 0755)

# label list
label = []
labeln = 0
for sdir in subdir:
    dirs = inpath+sdir
    for filename in glob.glob(dirs+'/*.JPG'):
        new_name = outpath+'image'+str(nfile)+'.JPG'
        os.rename(filename, new_name)
        nfile = nfile + 1
        # append label to file id
        label.append(labeln)
    labeln = labeln + 1
# save label
labelFile = open(inpath+"label.csv",'wb')
df = pd.DataFrame(label, columns=["label"])
df.to_csv(labelFile, index=False)