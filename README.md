


<img src="https://raw.githubusercontent.com/DaireNiC/Emerging-Techologies/master/media/emerging_tech.gif?token=ATSi0EMXwwN1uEVUuDSDRU23EBbcYBetks5cCtJDwA%3D%3D" width="100%"/>

## Emerging Technologies

> This repos contains a collection of Jupyter notebooks and python
> scripts relating to emerging technologies in the field of computer and
> data science.
> 

|| Contents ||
|--|--|--|
|1  | [Numpy Random Notebook](https://github.com/DaireNiC/Emerging-Techologies/blob/master/numpy_random.ipynb) | Exploring probability distributions & numpy.rand package
|2 | [Iris Dataset Notebook](https://github.com/DaireNiC/Emerging-Techologies/blob/master/iris_dastaset.ipynb) | Researching and visualizing the Iris dataset with Pandas, Keras & more
|3| [MNIST Dataset Notebook](https://github.com/DaireNiC/Emerging-Techologies/blob/master/MNIST_dataset/MNIST.ipynb) | How to read and interpret the MNIST dataset
|5 |[ MNSIT Dataset Reader Python Script](https://github.com/DaireNiC/Emerging-Techologies/blob/master/MNIST_dataset/mnist_script.py) | Python implementation of reading and storing the MNIST dataset
|4 | [Digit recognizer Notebook](https://github.com/DaireNiC/Emerging-Techologies/blob/master/MNIST_dataset/digit_recognition.ipynb) | Building Neural Networks to recognize hand drawn digits
|5 |[ Digit recognizer Notebook Python Script](https://github.com/DaireNiC/Emerging-Techologies/blob/master/MNIST_dataset/digitrec.py) | Command Line tool that takes an image file containing a handwritten digit and identifies the digit using a supervisedlearning algorithm and the MNIST dataset


## Using Jupyter Notebooks
To run and view the .ipynb files correctly, Jupyter Notebook/Jupyter lab must be installed. 

- Before installing ensure Python 3.3 or greater, or Python 2.7 is installed on your machine
- One approach to installing Python & Jupyter is using [Anaconda](https://www.anaconda.com/), get it from [here](https://www.anaconda.com/download/).

With Anaconda and Python installed, execute the following

    conda install jupyter 


With Juypter successfully installed , you can now download and run .ipynb files.

### Run the Notebook
#### Option 1: 
- Click on notebook 
- View raw
- Paste into text editor
- Save with .ipynb extension
- Execute
	- `jupyter notebook $what_you_saved_it_as.ipynb`
#### Option 2:
- Clone this repo
- Execute
	- 	 `jupyter notebook $my_notebook_name.ipynb`

## Extra Goodies
#### Things that I dedicated a little extra time to thoughout the development of this project

Command Line Tool
- The command line tool for the digit recogniser uses the argparse python library
- I added the option to  allow the user to enter an image via the command line 
- Also added option to draw your own digit
	- Used the Python libraries [Pillow]() & [Tkinter](https://wiki.python.org/moin/TkInter) to process user input
- Well structured code
	- I made use of the Black linter to ensure my python code conforms to Pep8 standards

 Consistent Work
- Evident in git history, added to the project at each and every week.

Notebook Explanations & Presentation
- I tried to have fun creating and writing the Jupyter Notebooks in this project. Explanations are in simple terms and aim to be succinct at all times. Hopefully they may someday help another developer begin to get a handle on some of the concepts I've explored as part of this module.  
- I also added some of my own hand drawn sketches, photoshop creations and carefully selected gifs to help clarify some of the tools and ideas discussed in the notebooks. 

Trial & Error
- I tried various approaches to each of the tasks required. This is evident in how I discuss using different ML algorithims, testing to see which yields the best accuracy - and prodding at why that might be. Also shown in my approach to using data vis libraries such a matplot and seaborn wherever possible to get a better understanding of the each concept.
