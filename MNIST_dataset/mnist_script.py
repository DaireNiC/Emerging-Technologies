# Adapted from :
# (1): https://www.youtube.com/watch?v=6xar6bxD80g
# (2): https://github.com/ianmcloughlin/jupyter-teaching-notebooks/blob/master/mnist.ipynb

# Reading in compressed files
import gzip
import numpy as np
# PIL to save np array as image
from PIL import Image
# read as sequence of bytes
# mimincs open command in python - but deals with compression
import os,codecs


# https://docs.python.org/3/library/enum.html
from enum import Enum
class MNIST(Enum):
    IMAGES = 2051
    LABELS = 2049
    TRAIN = 'train'
    TEST = 'test'

# PROVIDE YOUR DIRECTORY WITH THE EXTRACTED FILES HERE
datapath = './data/'

files = os.listdir(datapath)


#convert byte to int
def byte_to_int(b):
    return int.from_bytes(b, byteorder='big')

# The MNIST data set is in binary format and must be proessed as detailed inthe dataset's docs :http://yann.lecun.com/exdb/mnist/
def process_mnist():
    # get every file in the data directory

    data_dict = {}
    for file in files:
        print('Beginning rocessing on: ', file)
        # Unzip the .gz files
        with gzip.open(datapath + file, 'rb') as f:
            file_content = f.read()

        # MAGIC NUMBER : determines whether the it's labels or images we're reading
        type = byte_to_int(file_content[:4])
        print(type)
         # 4-7: LENGTH OF THE ARRAY (DIMENSION 0)
        length = byte_to_int(file_content[4:8])

        # if dealing with the images
        if (type == MNIST.IMAGES.value):
            category = MNIST.IMAGES.name
            num_rows = byte_to_int(file_content[8:12])  # Num of rows  (DIMENSION 1)
            num_cols = byte_to_int(file_content[12:16])  # num of cols  (DIMENSION 2)

            # read the pixel values as integers, offset account for meta data
            parsed = np.frombuffer(file_content,dtype = np.uint8, offset = 16)
            # Reshape the array [NO_OF_SAMPLES x HEIGHT x WIDTH]
            parsed = parsed.reshape(length,num_rows,num_cols)
        # otherwise dealing with label set
        elif(type == MNIST.LABELS.value):
            category = MNIST.LABELS.name
            parsed = np.frombuffer(file_content, dtype=np.uint8, offset=8)
            # reshape the array as number of samples
            parsed = parsed.reshape(length)
        # separate test data from training data
        if (length==10000):
            set = 'test'
        elif (length==60000):
            set = 'train'
        data_dict[set+'_'+category] = parsed  # Save the parsed data into a dict for convenience
        for key,value in data_dict.items():
            print(key)
    return data_dict


def save_mnist_png(mnist_data):
    # path where image will be saved to
    image_dir = './mnist_images/'

    sets = ['train','test']

    for set in sets:   # FOR TRAIN AND TEST SET
        images = mnist_data[set+'_' + MNIST.IMAGES.name]   # IMAGES
        labels = mnist_data[set+'_'+ MNIST.LABELS.name]   # LABELS
        no_of_samples = images.shape[0]     # NUBMER OF SAMPLES
        for indx in range (10):  # FOR EVERY SAMPLE
            print(set, indx)
            image = Image.fromarray(images[indx])         # GET IMAGE
            label = labels[indx]            # GET LABEL
            if not os.path.exists(image_dir+set+'/'+str(label)+'/'):    # IF DIRECTORIES DO NOT EXIST THEN
                os.makedirs (image_dir+set+'/'+str(label)+'/')       # CREATE TRAIN/TEST DIRECTORY AND CLASS SPECIFIC SUBDIRECTORY
            filenumber = len(os.listdir(image_dir+set+'/'+str(label)+'/'))  # NUMBER OF FILES IN THE DIRECTORY FOR NAMING THE FILE
            image.save(image_dir+set+'/'+str(label)+'/%05d.png'%(filenumber))  # SAVE THE IMAGE WITH PROPER NAME







data = process_mnist()

save_mnist_png(data)

#
# image = ~np.array(list(file_content[800: 1584])).reshape(28,28).astype(np.uint8)
# im = Image.fromarray(image)
# im.save("test_00001.png")
# save each 784 bytes as a png image
