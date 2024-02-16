#%%
import numpy as np
import tensorflow as tf
from tensorflow import keras 
import os 
import matplotlib.pyplot as plt 
import cv2 
from skimage.transform import resize 
import pathlib 

from keras.datasets import cifar10
from tensorflow.keras.applications.resnet50 import ResNet50 
from tensorflow.keras.applications.resnet50 import preprocess_input 
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers 
from tensorflow.keras.layers import Upsampling2D, Dense, Flatten, BatchNormalization, Dropout 
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, SGD
# %%
