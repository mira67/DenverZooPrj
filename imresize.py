"""
resize image
Author: qi liu
"""

#!/usr/bin/python
from PIL import Image
import os, sys

input_dir  = '/home/mirabot/googledrive/deeplearning/zoophoto/all_sorted/wildlife/'
output_dir  = '/home/mirabot/googledrive/deeplearning/zoophoto/all_sorted/resized/wildlife/'
dirs = os.listdir(input_dir)

def resize():
    for item in dirs:
        if os.path.isfile(input_dir+item):
            im = Image.open(input_dir+item)
            f, e = os.path.splitext(input_dir+item)
            #print item
            imResize = im.resize((64,64), Image.ANTIALIAS)
            imResize.save(output_dir + item, 'JPEG', quality=100)

resize()
print "Done!"
