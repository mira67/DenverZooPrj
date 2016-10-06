from PIL import Image
import os 
import threading
from multiprocessing import cpu_count


"""
READ ME

This program will dump all of the croped images into the current file
I would recomand putting it into a new empty folder

folder is where it will get the images to crop from.
It will get every file from that folder and trat it like an image

top_crop ajusts the amount of pixsels that will be cut off of the bottom.
bottom_crop ajusts the amount of pixels that will be cut off of the bottom.


SIDE NOTES

This program is multithreaded to speed up the process.
It will probobly eat up your cpu for a bit.

I keep all of the file names the same to see if you have duplicits.

"""

folder="/media/wire_wolf/LTS/Q_Liu_Photos/not_croped/"

top_crop=32
bottom_crop=64

def crop(image):
    img=Image.open(folder+image)
    width = img.size[0]
    height = img.size[1]
    box=(0,top_crop,width,height-bottom_crop)
    img3=img.crop(box)
    img3.save(image)


def cropping():
    while len(bit_list) != 0:
        image = bit_list.pop()
        crop(image)

bit_list = os.listdir(folder)

threads = []

# you can change cpu_count() to be an intiger value for the amount of threads you want
# egsample   for x in range(2):
# if you want it to be the max for your CPU don't change it 
for x in range(cpu_count()):
    t = threading.Thread(target=cropping)
    threads.append(t)
    t.start()

#single threded code. It took forever for it to run.
"""
for bitmap in os.listdir(folder):
    crop(bitmap)
"""