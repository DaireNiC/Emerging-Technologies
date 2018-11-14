# Reading in compressed files
import gzip
import numpy as np
# PIL to save np array as image
from PIL import Image
# read as sequence of bytes
# mimincs open command in python - but deals with compression
with gzip.open('data/t10k-images-idx3-ubyte.gz', 'rb') as f:
    file_content = f.read()

# i = 16
# j = 800
#
#  i = i+j --> 816
#  j +
#
# for ():
# 16:800
#
# i = 16
# j = 800
#
# i+ 16 + j
# 16 + 16 + 800 --> 832
#
# 800 + 800 + 16 --> 1616
# j + 800 + 16
image = ~np.array(list(file_content[800: 1584])).reshape(28,28).astype(np.uint8)
im = Image.fromarray(image)
im.save("test_00001.png")
# save each 784 bytes as a png image
