# Adapted from :
# (1): https://www.youtube.com/watch?v=6xar6bxD80g
# (2): https://github.com/ianmcloughlin/jupyter-teaching-notebooks/blob/master/mnist.ipynb
# (3): https://medium.com/@mannasiladittya/converting-mnist-data-in-idx-format-to-python-numpy-array-5cb9126f99f1

# Reading in compressed files
import gzip
import numpy as np
# PIL to save np array as image
from PIL import Image
# read as sequence of bytes
# mimincs open command in python - but deals with compression
import os,codecs


# https://docs.python.org/3/library/enum.html
# enum for constants relating to MNIST data

from enum import Enum
class MNIST(Enum):
    IMAGES = 2051
    LABELS = 2049
    TRAIN = 'train'
    TEST = 'test'
    TEST_SIZE = 10000
    TRAIN_SIZE = 60000

# provide your directory with the extracted files here
datapath = './data/'

files = os.listdir(datapath)


#convert byte to int
def byte_to_int(b):
    return int.from_bytes(b, byteorder='big')

# The MNIST data set is in binary format and must be proessed as detailed inthe dataset's docs :http://yann.lecun.com/exdb/mnist/
def process_mnist():
    # handy structure to hold the data we want to parse
    data_dict = {}
    # get every file in the data directory
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
        # separate test data from training data - we know which is which from mnist doc detailing set size
        if (length == MNIST.TEST_SIZE.value):
            set = 'test'
        elif (length==MNIST.TRAIN_SIZE.value):
            set = 'train'
        data_dict[set+'_'+category] = parsed  # Save the parsed data into a dict for convenience
        for key,value in data_dict.items():
            print(key)
    return data_dict


def save_mnist_png(mnist_data):
    # path where image will be saved to
    image_dir = './mnist_images/'

    sets = ['train','test']

    for set in sets:   # for train and test set
        images = mnist_data[set+'_' + MNIST.IMAGES.name]
        labels = mnist_data[set+'_'+ MNIST.LABELS.name]
        # num of samples
        no_of_samples = images.shape[0]
         # for every sample
        for indx in range (100): # using 100 as a test - change to num of samples
            print(set, indx)
            # get image
            image = Image.fromarray(images[indx])
            # get label
            label = labels[indx]
              # if directories do not exist then
            if not os.path.exists(image_dir+set+'/'+str(label)+'/'):
                 # create train/test directory and class specific subdirectory
                os.makedirs (image_dir+set+'/'+str(label)+'/')
            filenumber = len(os.listdir(image_dir+set+'/'+str(label)+'/'))  # number of files in the directory for naming the file
            # save the image with proper name
            image.save(image_dir+set+'/'+str(label)+'/%05d.png'%(filenumber))


data = process_mnist()
save_mnist_png(data)
