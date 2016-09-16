"""
resize image
Author: qi liu
"""

#!/usr/bin/python
from PIL import Image
import os, sys

input_dir  = '/home/mirabot/Documents/deeplearning/zoophoto/zoosorted/mixed/'
output_dir  = '/home/mirabot/Documents/deeplearning/zoophoto/zoosorted/mixed_resize/'
dirs = os.listdir(input_dir)

def resize():
    for item in dirs:
        if os.path.isfile(input_dir+item):
            im = Image.open(input_dir+item)
            f, e = os.path.splitext(input_dir+item)
            
            imResize = im.resize((200,200), Image.ANTIALIAS)
            imResize.save(output_dir + item, 'JPEG', quality=90)

resize()